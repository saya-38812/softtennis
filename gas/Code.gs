/**
 * サーブノート 管理用スプレッドシートの GAS（Google Apps Script）
 *
 * 役割：
 *   Webアプリ（Vercel）から「動画をアップロードしました」という連絡を受け取り、
 *   管理用スプレッドシートの「sessions」シートに1行追加する。
 *   この行が、Colab にとっての「解析待ちリスト」になる。
 *
 * 使い方：
 *   1. 管理用スプレッドシートを開き、[拡張機能] → [Apps Script] を開く
 *   2. このファイルの中身を Code.gs に丸ごと貼り付けて保存
 *   3. [デプロイ] → [新しいデプロイ] → 種類「ウェブアプリ」
 *        - 次のユーザーとして実行：自分（運営者）
 *        - アクセスできるユーザー：全員
 *   4. 発行された URL（…/exec）を、アプリの .env.local の NEXT_PUBLIC_GAS_URL に設定
 *
 * ※ コードを書き換えたら、[デプロイ] → [デプロイを管理] → 編集 →「新バージョン」で
 *   デプロイし直さないと、URL の先は古いコードのまま動き続けるので注意。
 */

// ============================================================
// 設定
// ============================================================

/** データを書き込むシートの名前 */
const SHEET_NAME = 'sessions';

/**
 * シートの列（左から順番）
 * ここを変えると列の並びが変わるので、Colab 側の読み込み処理もあわせて直すこと。
 */
const COLUMNS = [
  'received_at',      // 受け付けた日時
  'session_id',       // アプリ内のセッションID（1記録につき1つ）
  'requester_email',  // 依頼者のメールアドレス（Google のログイン情報から確認したもの）
  'mode',             // serve_tracking（IN/FAULT 記録）/ form_only（フォームだけ解析）
  'source',           // practice（練習画面）/ upload（アップロード画面）
  'recorded_at',      // 練習した日時（アプリでセッションを作った日時）
  'drive_file_id',    // Drive の動画ファイルID（Colab が動画を取り出すのに使う）
  'drive_folder_id',  // 依頼者の「サーブノート」フォルダID（解析結果の返却先）
  'video_name',       // 動画のファイル名
  'serve_count',      // サーブ本数
  'in_count',         // IN の本数
  'fault_count',      // FAULT の本数
  'serves_json',      // サーブ一覧（[{"result":"IN","timestamp_sec":1.5}, ...] の JSON）
  'status',           // 解析待ち / 解析中 / 完了 / エラー
  'analyzed_at',      // 解析が終わった日時（Colab が書く）
  'result_json',      // 解析結果（Colab が書く）
  'error_message',    // エラー内容（Colab が書く）
];

/** ステータスの値（Colab と共通で使う） */
const STATUS_PENDING = '解析待ち';

// ============================================================
// Web アプリの入口
// ============================================================

/**
 * POST リクエストを受け取る関数（GAS の決まった名前）
 *
 * アプリからは JSON 文字列が送られてくる。
 * ※ アプリ側は Content-Type を text/plain にして送る。
 *   application/json にすると、ブラウザが事前確認（プリフライト）の OPTIONS リクエストを送るが、
 *   GAS はそれに応答できず通信が失敗してしまうため。
 */
function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    const result = handleUploadNotice_(body);
    return jsonResponse_({ ok: true, ...result });
  } catch (err) {
    // エラーの内容をアプリに返す（アプリ側で画面に表示できるように）
    console.error(err);
    return jsonResponse_({ ok: false, error: String(err && err.message ? err.message : err) });
  }
}

/** 動作確認用：ブラウザで URL を開くと「動いています」と返す */
function doGet() {
  return jsonResponse_({ ok: true, message: 'サーブノート GAS は動いています' });
}

// ============================================================
// メイン処理
// ============================================================

/**
 * 「動画をアップロードしました」の連絡を処理して、シートに書き込む
 *
 * URL を知っていれば誰でも POST できてしまうので、次の2つを確認してから書き込む：
 *   1. アクセストークンを Google に問い合わせて、本物の Google ユーザーかを確認
 *      （メールアドレスは、送られてきた値ではなく Google から返ってきた値を使う）
 *   2. 送られてきた動画ファイルを、運営者（このスクリプトの実行者）が本当に開けるかを確認
 *      （＝運営者に共有されたフォルダの中にある動画だけを受け付ける）
 */
function handleUploadNotice_(body) {
  // ---- 入力チェック ----
  const required = ['access_token', 'session_id', 'drive_file_id', 'drive_folder_id'];
  required.forEach(function (key) {
    if (!body[key]) throw new Error('必要な項目がありません: ' + key);
  });

  // ---- 1. 本物の Google ユーザーか確認 ----
  const email = fetchVerifiedEmail_(body.access_token);

  // ---- 2. 運営者が開ける動画か確認 ----
  let videoName;
  try {
    videoName = DriveApp.getFileById(body.drive_file_id).getName();
  } catch (err) {
    throw new Error('動画ファイルを確認できませんでした（フォルダがコーチに共有されていない可能性があります）');
  }

  // ---- 書き込む1行分のデータを作る ----
  const serves = Array.isArray(body.serves) ? body.serves : [];
  const record = {
    received_at: new Date(),
    session_id: body.session_id,
    requester_email: email,
    mode: body.mode || 'serve_tracking',
    source: body.source || 'practice',
    recorded_at: body.recorded_at ? new Date(body.recorded_at) : '',
    drive_file_id: body.drive_file_id,
    drive_folder_id: body.drive_folder_id,
    video_name: videoName,
    serve_count: serves.length,
    in_count: serves.filter(function (s) { return s.result === 'IN'; }).length,
    fault_count: serves.filter(function (s) { return s.result === 'FAULT'; }).length,
    serves_json: JSON.stringify(serves),
    status: STATUS_PENDING,
    analyzed_at: '',
    result_json: '',
    error_message: '',
  };

  // ---- シートに書き込む ----
  // LockService：2人が同時に送ってきても、書き込みが混ざらないよう1人ずつ順番に処理する
  const lock = LockService.getScriptLock();
  lock.waitLock(10 * 1000); // 最大10秒待つ
  try {
    const sheet = getOrCreateSheet_();
    const row = COLUMNS.map(function (col) { return record[col]; });

    // 同じセッションIDの行がすでにあれば上書き（通信エラーでアプリが再送しても、行が増えないように）
    const existingRow = findRowBySessionId_(sheet, body.session_id);
    if (existingRow > 0) {
      sheet.getRange(existingRow, 1, 1, row.length).setValues([row]);
      return { updated: true };
    }
    sheet.appendRow(row);
    return { updated: false };
  } finally {
    lock.releaseLock(); // 処理が終わったら（エラーでも）必ずロックを外す
  }
}

// ============================================================
// 補助関数（名前の最後の _ は「このファイルの中だけで使う」という GAS の慣習）
// ============================================================

/**
 * アクセストークンを Google に問い合わせて、メールアドレスを取得する
 * トークンが偽物・期限切れなら Google がエラーを返すので、ここで弾ける。
 */
function fetchVerifiedEmail_(accessToken) {
  const res = UrlFetchApp.fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
    headers: { Authorization: 'Bearer ' + accessToken },
    muteHttpExceptions: true, // エラーでも例外にせず、ステータスコードで判定する
  });
  if (res.getResponseCode() !== 200) {
    throw new Error('Google のログイン情報を確認できませんでした。もう一度ログインしてください。');
  }
  const info = JSON.parse(res.getContentText());
  if (!info.email) throw new Error('メールアドレスを取得できませんでした');
  return info.email;
}

/** 「sessions」シートを取得する。なければ見出し行付きで作る */
function getOrCreateSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sheet = ss.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = ss.insertSheet(SHEET_NAME);
    sheet.appendRow(COLUMNS);
    sheet.setFrozenRows(1); // 見出し行を固定して、スクロールしても見えるようにする
  }
  return sheet;
}

/** セッションIDが一致する行の番号を返す（見つからなければ -1） */
function findRowBySessionId_(sheet, sessionId) {
  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return -1; // 見出し行しかない
  const col = COLUMNS.indexOf('session_id') + 1; // getRange は 1 始まり
  const ids = sheet.getRange(2, col, lastRow - 1, 1).getValues();
  for (let i = 0; i < ids.length; i++) {
    if (ids[i][0] === sessionId) return i + 2; // 2行目から始まっているので +2
  }
  return -1;
}

/** オブジェクトを JSON のレスポンスにする */
function jsonResponse_(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * 【初回だけ手動で実行する】権限の許可とシートの準備
 * GAS エディタの上部で関数「setup」を選んで ▶実行 を押すと、
 * スプレッドシート・Drive・外部通信の権限を許可する画面が出る。
 */
function setup() {
  getOrCreateSheet_();
  // DriveApp と UrlFetchApp を一度呼んで、権限の許可画面を出させる
  DriveApp.getRootFolder();
  UrlFetchApp.fetch('https://www.googleapis.com/discovery/v1/apis', { muteHttpExceptions: true });
  console.log('準備完了：sessions シートを確認してください');
}
