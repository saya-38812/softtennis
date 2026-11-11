/**
 * ブラウザ内（localStorage）に練習セッションを保存するモジュール
 *
 * 以前は IN / FAULT をタップするたびに FastAPI へ送信していたが、
 * 新しい構成ではサーバーを持たないため、記録はまずブラウザ内にためておく。
 * （最後に Drive / GAS へまとめて送る処理は、次のステップで追加する）
 *
 * なぜ localStorage を使うのか：
 * - React の state だけだと、ページを再読み込みしたときに記録が消えてしまう
 * - サーブの記録は数KB程度なので、localStorage（約5MBまで）で十分に収まる
 * ※ 動画は大きすぎるので localStorage には入れない（SessionVideoContext で持つ）
 */

import type { ServeResult, TrackerSession } from "@/lib/api";

// localStorage に保存するときのキー名（他のデータとぶつからないようにアプリ名を付ける）
const STORAGE_KEY = "servenote.sessions";

// ------------------------------------------------------------
// 内部用：全セッションの読み書き
// ------------------------------------------------------------

/**
 * 保存されている全セッションを読み込む。
 * localStorage が使えない環境（プライベートモードなど）や、
 * データが壊れている場合でもアプリが落ちないよう、空配列を返す。
 */
function loadAll(): TrackerSession[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    // 配列でなければ壊れたデータとみなす
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.warn("セッションの読み込みに失敗しました", e);
    return [];
  }
}

/** 全セッションを保存する */
function saveAll(sessions: TrackerSession[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  } catch (e) {
    // 容量オーバーなどで保存できなかった場合は、呼び出し元に知らせる
    throw new Error("記録をブラウザに保存できませんでした");
  }
}

/**
 * サーブ一覧から IN / FAULT などの件数を数え直す。
 * 件数を別々に足し引きするとズレる原因になるので、
 * 毎回 serves 配列から計算し直す（＝データの正しさを1か所に集める）。
 */
function recount(session: TrackerSession): TrackerSession {
  const serves = session.serves ?? [];
  return {
    ...session,
    total_attempts: serves.length,
    in_count: serves.filter((s) => s.result === "IN").length,
    out_count: serves.filter((s) => s.result === "OUT").length,
    fault_count: serves.filter((s) => s.result === "FAULT").length,
  };
}

/**
 * 指定IDのセッションを書き換えて保存する共通処理。
 * update には「古いセッションを受け取って新しいセッションを返す関数」を渡す。
 */
function updateSession(
  sessionId: string,
  update: (session: TrackerSession) => TrackerSession
): TrackerSession {
  const sessions = loadAll();
  const index = sessions.findIndex((s) => s.id === sessionId);
  if (index === -1) {
    throw new Error("セッションが見つかりません");
  }
  const updated = recount(update(sessions[index]));
  sessions[index] = updated;
  saveAll(sessions);
  return updated;
}

/** Date を端末の現地時刻で "YYYY-MM-DD" 形式の文字列にする */
function toLocalDateString(date: Date): string {
  const year = date.getFullYear();
  // getMonth() は 0 始まり（1月 = 0）なので +1 する。padStart で "1" → "01" にそろえる
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

// ------------------------------------------------------------
// 外部に公開する関数
// ------------------------------------------------------------

/** 新しい練習セッションを作成して、そのIDを返す */
export function createLocalSession(): TrackerSession {
  const now = new Date();
  const session: TrackerSession = {
    // crypto.randomUUID() はブラウザ標準の「ほぼ重複しないID」を作る関数
    id: crypto.randomUUID(),
    // 例: "2026-11-11"
    // ※ toISOString() は世界標準時(UTC)なので、日本の朝9時前だと前日になってしまう。
    //   そのため端末の現地時刻から年月日を組み立てる。
    date: toLocalDateString(now),
    created_at: now.toISOString(),
    total_attempts: 0,
    in_count: 0,
    out_count: 0,
    fault_count: 0,
    serves: [],
  };
  const sessions = loadAll();
  sessions.push(session);
  saveAll(sessions);
  return session;
}

/** IDを指定してセッションを1件取得する（見つからなければ null） */
export function getLocalSession(sessionId: string): TrackerSession | null {
  return loadAll().find((s) => s.id === sessionId) ?? null;
}

/** 保存されている全セッションを新しい順で返す（履歴画面用） */
export function listLocalSessions(): TrackerSession[] {
  return loadAll().sort((a, b) =>
    (b.created_at ?? "").localeCompare(a.created_at ?? "")
  );
}

/**
 * サーブを1本記録する
 * @param timestampSec 録画開始から何秒後にタップしたか（後で動画を切り出すのに使う）
 */
export function addLocalServe(
  sessionId: string,
  result: ServeResult,
  timestampSec: number
): TrackerSession {
  return updateSession(sessionId, (session) => ({
    ...session,
    serves: [
      ...(session.serves ?? []),
      {
        id: crypto.randomUUID(),
        result,
        timestamp: new Date().toISOString(), // タップした実際の時刻
        timestamp_sec: timestampSec, // 動画内での位置（秒）
      },
    ],
  }));
}

/**
 * 記録済みのサーブ一覧から、セッションを1件まとめて作る（アップロード画面用）。
 *
 * アップロード画面では、途中で画面を閉じたときに「空のセッション」が履歴に残らないよう、
 * タップ中は画面の中だけで記録を持ち、「記録を完了」を押したときに1回で保存する。
 * 巻き戻してからタップすると順番がバラバラになるので、ここで動画内の時刻順に並べ替える。
 */
export function createLocalSessionFromServes(
  serves: { result: ServeResult; timestampSec: number }[]
): TrackerSession {
  const session = createLocalSession();
  // 元の配列を壊さないよう、コピー（[...serves]）してから時刻の早い順に並べ替える
  const sorted = [...serves].sort((a, b) => a.timestampSec - b.timestampSec);
  return updateSession(session.id, (s) => ({
    ...s,
    source: "upload", // 「アップロードした動画から記録した」ことの目印
    serves: sorted.map((serve) => ({
      id: crypto.randomUUID(),
      result: serve.result,
      timestamp: new Date().toISOString(),
      timestamp_sec: serve.timestampSec,
    })),
  }));
}

/**
 * 「フォームだけ解析」用のセッションを作る（アップロード画面用）。
 *
 * IN/FAULT は記録しないので serves は空のまま。
 * 動画はサーブ1本分を想定し、打った瞬間（インパクト）は後で Colab 側が自動で見つける。
 */
export function createFormOnlySession(): TrackerSession {
  const session = createLocalSession();
  return updateSession(session.id, (s) => ({
    ...s,
    source: "upload",
    mode: "form_only",
  }));
}

/** 「フォームだけ解析」のセッションかどうか */
export function isFormOnly(session: TrackerSession): boolean {
  return session.mode === "form_only";
}

/**
 * Drive へのアップロードが終わったことをセッションに記録する。
 * これがあれば「アップロード済み」と表示でき、同じ動画を二重に送るのも防げる。
 */
export function markLocalSessionUploaded(
  sessionId: string,
  file: { id: string; name: string; webViewLink?: string }
): TrackerSession {
  return updateSession(sessionId, (session) => ({
    ...session,
    drive_file: { ...file, uploaded_at: new Date().toISOString() },
  }));
}

/** 管理用スプシ（GAS）へ解析を依頼できたことを記録する */
export function markLocalSessionRequested(sessionId: string): TrackerSession {
  return updateSession(sessionId, (session) => ({
    ...session,
    analysis_requested_at: new Date().toISOString(),
  }));
}

/** セッションを1件削除する（履歴画面の「削除」ボタン用） */
export function deleteLocalSession(sessionId: string): void {
  // 指定ID以外を残す ＝ 指定IDだけを消す
  const remaining = loadAll().filter((s) => s.id !== sessionId);
  saveAll(remaining);
}

/** 保存されている全セッションを削除する（履歴画面の「全削除」ボタン用） */
export function clearAllLocalSessions(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    throw new Error("記録を削除できませんでした");
  }
}

/**
 * 動作確認用のサンプルセッションを1件作る。
 * PC などカメラが使えない環境でも、履歴や結果画面の見た目を確認できるようにするため。
 */
export function createSampleSession(): TrackerSession {
  const session = createLocalSession();
  // IN と FAULT が混ざった10本分の記録（5秒おきにサーブした想定）
  const results: ServeResult[] = [
    "IN", "IN", "FAULT", "IN", "FAULT",
    "IN", "IN", "IN", "FAULT", "IN",
  ];
  let updated = session;
  results.forEach((result, index) => {
    updated = addLocalServe(session.id, result, (index + 1) * 5);
  });
  return updated;
}

/** 最後に記録したサーブを1本取り消す（Undo ボタン用） */
export function undoLocalServe(sessionId: string): TrackerSession {
  return updateSession(sessionId, (session) => ({
    ...session,
    // slice(0, -1) で「最後の1件を除いた配列」を作る
    serves: (session.serves ?? []).slice(0, -1),
  }));
}
