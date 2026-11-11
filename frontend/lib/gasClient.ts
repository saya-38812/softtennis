/**
 * 管理用スプレッドシートの GAS（Web アプリ）と通信するモジュール
 *
 * 動画を Drive に上げたあと、「この動画の解析をお願いします」という連絡を GAS に送る。
 * GAS はそれを管理用スプシに1行追加し、Colab がその行を見て解析する。
 *
 * serves（何秒目に IN/FAULT か）は依頼者に見せる必要がないので、
 * Drive にファイルとして置かず、ここで直接スプシへ送る。
 */

import type { ModelAnalysisResult, TrackerSession } from "@/lib/api";

/** GAS の Web アプリの URL（.env.local の NEXT_PUBLIC_GAS_URL） */
const GAS_URL = process.env.NEXT_PUBLIC_GAS_URL ?? "";

/** GAS の URL が設定されているか */
export function isGasConfigured(): boolean {
  return GAS_URL.length > 0;
}

/** GAS からの返事の形 */
interface GasResponse {
  ok: boolean;
  error?: string;
}

/**
 * GAS に POST して、返事（JSON）を受け取る共通処理
 *
 * ★ Content-Type を text/plain にするのがポイント。
 *   application/json にすると、ブラウザが本番の前に「送ってもいい？」という
 *   確認（プリフライト / OPTIONS リクエスト）を送るが、GAS はそれに答えられず失敗する。
 *   text/plain なら確認なしで送れて、GAS 側では e.postData.contents で JSON を読める。
 */
async function postToGas<T extends GasResponse>(body: object, failMessage: string): Promise<T> {
  if (!isGasConfigured()) {
    throw new Error("解析サービス（GAS の URL）が設定されていません");
  }
  let res: Response;
  try {
    res = await fetch(GAS_URL, {
      method: "POST",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(`${failMessage}。通信環境を確認してください。`);
  }
  if (!res.ok) {
    throw new Error(`${failMessage}（${res.status}）`);
  }
  const data: T = await res.json();
  if (!data.ok) {
    // GAS 側のエラーメッセージ（例：ログイン情報を確認できません）をそのまま見せる
    throw new Error(data.error ?? failMessage);
  }
  return data;
}

/** 解析状況1件分（スプシの1行に対応） */
export interface RemoteAnalysisStatus {
  status: string; // 解析待ち / 解析中 / 完了 / エラー
  analyzed_at: string;
  result: ModelAnalysisResult | null;
  error_message: string;
}

/**
 * 解析状況・結果を GAS に問い合わせる
 *
 * GAS 側で「トークンの持ち主 = 依頼者本人」の行だけを返すので、他人の結果は見えない。
 * 戻り値：セッションID → 解析状況（スプシに行がないセッションは含まれない）
 */
export async function fetchAnalysisResults(params: {
  accessToken: string;
  sessionIds: string[];
}): Promise<Record<string, RemoteAnalysisStatus>> {
  const data = await postToGas<GasResponse & { results: Record<string, RemoteAnalysisStatus> }>(
    {
      action: "get_results",
      access_token: params.accessToken,
      session_ids: params.sessionIds,
    },
    "解析結果を取得できませんでした"
  );
  return data.results ?? {};
}

/**
 * 「動画をアップロードしたので解析をお願いします」と GAS に連絡する
 *
 * @param accessToken 依頼者の Google のアクセストークン。
 *                    GAS 側でこれを Google に問い合わせて、本物のユーザーか・誰なのかを確認する
 */
export async function notifyUploadToGas(params: {
  accessToken: string;
  session: TrackerSession;
  folderId: string;
}): Promise<void> {
  const { accessToken, session, folderId } = params;

  if (!session.drive_file) {
    throw new Error("動画がまだアップロードされていません");
  }

  // GAS に送る中身。サーブ一覧は Colab が使う最小限の項目（結果と秒数）だけにする
  const body = {
    action: "notify_upload",
    access_token: accessToken,
    session_id: session.id,
    mode: session.mode ?? "serve_tracking",
    source: session.source ?? "practice", // 練習画面のセッションには source が付いていない
    recorded_at: session.created_at,
    drive_file_id: session.drive_file.id,
    drive_folder_id: folderId,
    serves: (session.serves ?? []).map((s) => ({
      result: s.result,
      timestamp_sec: s.timestamp_sec,
    })),
  };

  await postToGas(body, "解析の依頼に失敗しました");
}
