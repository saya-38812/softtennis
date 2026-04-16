/**
 * MVP API: セッション・サーブ・フォーム解析
 */

const getApiBase = () => {
  return process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
};

export type ServeResult = "IN" | "OUT" | "FAULT";

export interface ServeAnalysis {
  scores: Record<string, number>;
  focus_label: string;
  ai_text: string;
  practice: string;
}

export interface TrackerSession {
  id: string;
  date: string;
  created_at?: string;
  video_path?: string | null;
  /** 骨格トラッキング付き解析動画（/outputs/ 配下のファイル名） */
  tracking_video?: string | null;
  /** 出所（"upload" = アップロード画面から） */
  source?: string | null;
  total_attempts: number;
  in_count: number;
  out_count: number;
  fault_count: number;
  serves?: {
    id: string;
    result: ServeResult;
    timestamp: string;
    timestamp_sec?: number;
  }[];
  analysis?: Record<string, ServeAnalysis>;
}

export interface SessionStartResponse {
  session_id: string;
  date: string;
}

export async function startSession(): Promise<SessionStartResponse> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/start-session`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function recordServe(
  sessionId: string,
  result: ServeResult,
  timestamp_sec?: number
): Promise<{ ok: boolean; serve: unknown }> {
  const base = getApiBase();
  const body: { session_id: string; result: ServeResult; timestamp_sec?: number } = {
    session_id: sessionId,
    result,
  };
  if (typeof timestamp_sec === "number") body.timestamp_sec = timestamp_sec;
  const res = await fetch(`${base}/api/serves`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function undoServe(sessionId: string): Promise<{ ok: boolean }> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/serves/undo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function endSession(sessionId: string): Promise<TrackerSession> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/sessions/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSession(sessionId: string): Promise<TrackerSession> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/sessions/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listSessions(): Promise<TrackerSession[]> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/sessions`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<{ ok: boolean }> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** 全セッション・動画・出力を削除 */
export async function resetAll(): Promise<{ status: string; message: string }> {
  const base = getApiBase();
  const res = await fetch(`${base}/reset-all`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** サンプルセッションを追加（録画なしで結果画面を確認する用）。Next.js API経由でバックエンドに転送。 */
export async function seedSample(): Promise<{ created: string[]; message: string }> {
  const res = await fetch("/api/seed", { method: "POST" });
  if (!res.ok) {
    const text = await res.text();
    let msg = text;
    try {
      const j = JSON.parse(text);
      if (j.message) msg = j.message;
      else if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      // use text as is
    }
    throw new Error(msg);
  }
  return res.json();
}

/**
 * フォーム解析: 動画＋サーブイベントを送信し、各サーブの解析結果を取得。
 */
export async function runFormAnalysis(
  sessionId: string,
  videoBlob: Blob,
  serveEvents: { id: string; result: ServeResult; timestamp_sec: number }[]
): Promise<TrackerSession> {
  const base = getApiBase();
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("serve_events", JSON.stringify(serveEvents));
  form.append("video", videoBlob, "recording.webm");
  const res = await fetch(`${base}/api/analysis`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
