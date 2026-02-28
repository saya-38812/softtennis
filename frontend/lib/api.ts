const getApiBase = () => {
  return process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
};

export interface AnalyzeResponse {
  score: number;
  feedback: string[];
  metrics: {
    elbow_angle: number;
    body_sway: number;
    impact_height: number;
  };
}

/** SSE 解析結果（POST /analyze の type: result） */
export interface AnalyzeStreamResult {
  type: "result";
  status: string;
  session_id: string;
  scores: Record<string, number>;
  normalized_scores: Record<string, number>;
  improvement: Record<string, number> | null;
  improvement_message: string;
  ai_text: string;
  practice: string[] | string;
  user_image: string;
  user_video: string;
  ideal_image: string;
  focus_label: string;
  comparison: {
    label_ideal?: string;
    label_user?: string;
    value_ideal?: number;
    value_user?: number;
    action_tip?: string;
  };
  count: number;
}

export type AnalyzeStreamEvent =
  | { type: "progress"; percent: number }
  | { type: "error"; message: string }
  | { type: "result"; result: AnalyzeStreamResult };

/**
 * 動画を POST /analyze に送り、SSE で進捗と結果を受け取る。
 * onEvent で progress / error / result を逐次受け取る。
 */
export async function analyzeServeStreaming(
  video: File,
  onEvent: (ev: AnalyzeStreamEvent) => void
): Promise<AnalyzeStreamResult | null> {
  const base = getApiBase();
  const formData = new FormData();
  formData.append("file", video);

  const res = await fetch(`${base}/analyze`, {
    method: "POST",
    body: formData,
    headers: {},
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const dec = new TextDecoder();
  let buffer = "";
  let result: AnalyzeStreamResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += dec.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const event of events) {
      const line = event.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        const data = JSON.parse(line.slice(6)) as {
          type: string;
          percent?: number;
          message?: string;
          [key: string]: unknown;
        };
        if (data.type === "progress" && typeof data.percent === "number") {
          onEvent({ type: "progress", percent: data.percent });
        } else if (data.type === "error") {
          onEvent({ type: "error", message: data.message ?? "解析に失敗しました" });
          return null;
        } else if (data.type === "result") {
          result = data as unknown as AnalyzeStreamResult;
          onEvent({ type: "result", result });
          return result;
        }
      } catch {
        // ignore parse error
      }
    }
  }

  return result;
}

export async function analyzeServe(video: File): Promise<AnalyzeResponse> {
  const base = getApiBase();
  const formData = new FormData();
  formData.append("video", video);

  const res = await fetch(`${base}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json();
}

// --- Serve Tracker MVP ---

export type ServeResult = "IN" | "OUT" | "FAULT";

export interface TrackerSession {
  id: string;
  date: string;
  total_attempts: number;
  in_count: number;
  out_count: number;
  fault_count: number;
  avg_speed: number | null;
  max_speed: number | null;
  serves?: { id: string; result: ServeResult; speed?: number; timestamp: string }[];
}

export interface SessionStartResponse {
  session_id: string;
  date: string;
}

export async function startTrackerSession(): Promise<SessionStartResponse> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/session/start`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function recordServe(sessionId: string, result: ServeResult): Promise<{ ok: boolean; serve: unknown }> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/serve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, result }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function undoServe(sessionId: string): Promise<{ ok: boolean }> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/serve/undo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getTrackerSession(sessionId: string): Promise<TrackerSession> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/session/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listTrackerSessions(): Promise<TrackerSession[]> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/sessions`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function endTrackerSession(sessionId: string): Promise<TrackerSession> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/session/end`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteTrackerSession(sessionId: string): Promise<{ ok: boolean }> {
  const base = getApiBase();
  const res = await fetch(`${base}/api/sessions/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
