/**
 * Google Drive API を扱うモジュール
 *
 * 今はまだ「サーブノート」フォルダの作成だけ。
 * 動画のアップロードやコーチへの共有は、次のステップでここに追加していく。
 *
 * Drive API はただの Web API なので、fetch で URL を呼ぶだけで使える。
 * そのとき、ヘッダーに「Authorization: Bearer アクセストークン」を付けるのがポイント。
 */

/** 依頼者の Drive に作るフォルダの名前 */
export const APP_FOLDER_NAME = "サーブノート";

/** Drive でフォルダを表す特別な種類（MIMEタイプ） */
const FOLDER_MIME_TYPE = "application/vnd.google-apps.folder";

/** Drive API の基本URL */
const DRIVE_API = "https://www.googleapis.com/drive/v3";

/** Drive 上のフォルダ情報 */
export interface DriveFolder {
  id: string;
  name: string;
  /** ブラウザで開くためのURL */
  webViewLink?: string;
}

/**
 * Drive API を呼び出す共通処理
 * エラーのときに、原因がわかるメッセージを付けて投げ直す。
 */
async function driveFetch<T>(
  accessToken: string,
  url: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      ...init.headers,
      Authorization: `Bearer ${accessToken}`,
    },
  });

  if (res.status === 401) {
    // 401 = 認証エラー。たいていはトークンの期限切れ
    throw new Error("Googleのログインが切れました。もう一度ログインしてください。");
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Google Drive の操作に失敗しました（${res.status}）: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ------------------------------------------------------------
// 運営者への共有
// ------------------------------------------------------------

/**
 * 共有先の運営者のメールアドレス（.env.local の NEXT_PUBLIC_OPERATOR_EMAIL）
 * Colab は運営者のアカウントで動くので、依頼者のフォルダを運営者が見られるようにする必要がある。
 */
const OPERATOR_EMAIL = (process.env.NEXT_PUBLIC_OPERATOR_EMAIL ?? "").trim().toLowerCase();

/** 共有処理の結果（画面に状態を出すのに使う） */
export type ShareStatus =
  | "shared" // 共有できた（または共有済みだった）
  | "self" // ログインしているのが運営者本人なので共有不要
  | "not_configured"; // 運営者のメールアドレスが未設定

/**
 * フォルダを運営者に「編集者」として共有する。
 *
 * なぜ「編集者（writer）」なのか：
 *   Colab で解析したあと、IN/FAULT ごとに切り出した動画やレポートを
 *   依頼者のフォルダに返却する予定なので、書き込み権限が必要。
 *
 * 毎回ログインのたびに呼ばれるので、すでに共有済みなら何もしない（二重に共有しない）。
 *
 * @param signedInEmail ログインしている人のメールアドレス
 */
export async function shareFolderWithOperator(
  accessToken: string,
  folderId: string,
  signedInEmail: string
): Promise<ShareStatus> {
  if (!OPERATOR_EMAIL) return "not_configured";

  // 運営者本人がログインしている場合、自分のフォルダを自分に共有することはできない
  if (signedInEmail.trim().toLowerCase() === OPERATOR_EMAIL) return "self";

  // ---- 1. すでに共有済みか確認する ----
  const existing = await driveFetch<{ permissions: { emailAddress?: string }[] }>(
    accessToken,
    `${DRIVE_API}/files/${folderId}/permissions?fields=permissions(emailAddress)`
  );
  const alreadyShared = existing.permissions.some(
    (p) => p.emailAddress?.toLowerCase() === OPERATOR_EMAIL
  );
  if (alreadyShared) return "shared";

  // ---- 2. 共有する ----
  // sendNotificationEmail=false：共有のたびに運営者へ通知メールが飛ばないようにする
  // （依頼者が増えるとメールが大量に届いてしまうため。共有されたフォルダは Drive の「共有アイテム」で見られる）
  await driveFetch(
    accessToken,
    `${DRIVE_API}/files/${folderId}/permissions?sendNotificationEmail=false`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type: "user", // 特定の人に共有する
        role: "writer", // 編集者
        emailAddress: OPERATOR_EMAIL,
      }),
    }
  );
  return "shared";
}

// ------------------------------------------------------------
// 動画のアップロード
// ------------------------------------------------------------

/** アップロードした動画ファイルの情報 */
export interface DriveFile {
  id: string;
  name: string;
  webViewLink?: string;
}

/**
 * 動画の種類（MIMEタイプ）から拡張子を決める。
 * 録画データは "video/webm;codecs=vp9" のように付属情報が付くことがあるので、";" より前だけを見る。
 */
function extensionFor(mimeType: string): string {
  const base = mimeType.split(";")[0];
  if (base === "video/mp4") return "mp4";
  if (base === "video/quicktime") return "mov"; // iPhone で撮った動画
  return "webm";
}

/** Date を "2026-11-11_0905" のようなファイル名用の文字列にする（端末の現地時刻） */
function toFileTimestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `_${pad(date.getHours())}${pad(date.getMinutes())}`
  );
}

/**
 * 動画を Drive のフォルダにアップロードする（レジュマブルアップロード方式）
 *
 * 動画は数十〜数百MBになるので、「レジュマブル（再開可能）アップロード」を使う。
 * 手順は2段階：
 *   1. 「これからこういうファイルを送ります」と伝えて、送り先URL（セッションURL）をもらう
 *   2. そのURLに動画の中身を送る
 * 2. では fetch ではなく XMLHttpRequest を使う。
 * fetch は送信の進み具合（何%送れたか）を取れないが、XMLHttpRequest は取れるため。
 *
 * @param sessionId  アプリ内のセッションID。appProperties（アプリ専用の隠しメモ欄）に入れておき、
 *                   後で Colab やスプシから「どの記録の動画か」を探せるようにする
 * @param onProgress 0〜100 の進み具合を受け取る関数（進捗バー用）
 */
export async function uploadVideoToDrive(params: {
  accessToken: string;
  folderId: string;
  video: Blob;
  sessionId: string;
  recordedAt: Date;
  onProgress?: (percent: number) => void;
}): Promise<DriveFile> {
  const { accessToken, folderId, video, sessionId, recordedAt, onProgress } = params;
  const mimeType = video.type.split(";")[0] || "video/webm";
  const fileName = `サーブ練習_${toFileTimestamp(recordedAt)}.${extensionFor(mimeType)}`;

  // ---- 1. アップロードの予約（セッションURLをもらう） ----
  const initRes = await fetch(
    "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,webViewLink",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": mimeType,
        "X-Upload-Content-Length": String(video.size),
      },
      body: JSON.stringify({
        name: fileName,
        mimeType,
        parents: [folderId], // 「サーブノート」フォルダの中に入れる
        appProperties: { servenoteSessionId: sessionId },
      }),
    }
  );
  if (initRes.status === 401) {
    throw new Error("Googleのログインが切れました。もう一度ログインしてください。");
  }
  if (!initRes.ok) {
    throw new Error(`アップロードの準備に失敗しました（${initRes.status}）: ${await initRes.text()}`);
  }
  // 送り先URLは、レスポンス本文ではなく Location ヘッダーに入っている
  const uploadUrl = initRes.headers.get("Location");
  if (!uploadUrl) {
    throw new Error("アップロード先のURLを取得できませんでした");
  }

  // ---- 2. 動画の中身を送る ----
  return new Promise<DriveFile>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", uploadUrl);
    xhr.setRequestHeader("Content-Type", mimeType);

    // 送信の進み具合が変わるたびに呼ばれる
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    // 通信が終わったとき（成功・失敗どちらでも呼ばれるので status で判定する）
    xhr.onload = () => {
      if (xhr.status === 200 || xhr.status === 201) {
        resolve(JSON.parse(xhr.responseText) as DriveFile);
      } else {
        reject(new Error(`アップロードに失敗しました（${xhr.status}）`));
      }
    };
    // 電波が切れたなど、通信自体ができなかったとき
    xhr.onerror = () => {
      reject(new Error("通信エラーでアップロードできませんでした。電波の良い場所で再度お試しください。"));
    };
    xhr.send(video);
  });
}

/**
 * 「サーブノート」フォルダを探して、なければ作る。
 *
 * 毎回作ってしまうと、同じ名前のフォルダがどんどん増えてしまう。
 * そのため「まず探す → 見つからなければ作る」という順番にしている。
 *
 * ※ drive.file 権限では「このアプリが作ったファイル」しか見えないので、
 *   依頼者が自分で同じ名前のフォルダを作っていても、そちらは見つからない（＝混ざらない）。
 */
export async function ensureAppFolder(accessToken: string): Promise<DriveFolder> {
  // ---- 1. 既存のフォルダを探す ----
  // q は Drive の検索条件。「マイドライブ直下にある・ゴミ箱に入っていない・この名前のフォルダ」
  const query = [
    `name = '${APP_FOLDER_NAME}'`,
    `mimeType = '${FOLDER_MIME_TYPE}'`,
    `'root' in parents`,
    `trashed = false`,
  ].join(" and ");

  // URLSearchParams を使うと、日本語や記号を URL 用に正しく変換してくれる
  const params = new URLSearchParams({
    q: query,
    fields: "files(id, name, webViewLink)", // 必要な項目だけ返してもらう（通信量の節約）
    spaces: "drive",
  });
  const found = await driveFetch<{ files: DriveFolder[] }>(
    accessToken,
    `${DRIVE_API}/files?${params.toString()}`
  );
  if (found.files.length > 0) {
    return found.files[0];
  }

  // ---- 2. 見つからなければ新しく作る ----
  return driveFetch<DriveFolder>(
    accessToken,
    `${DRIVE_API}/files?fields=id,name,webViewLink`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: APP_FOLDER_NAME,
        mimeType: FOLDER_MIME_TYPE,
        parents: ["root"], // マイドライブ直下に作る
      }),
    }
  );
}
