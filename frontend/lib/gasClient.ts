/**
 * 管理用スプレッドシートの GAS（Web アプリ）と通信するモジュール
 *
 * 動画を Drive に上げたあと、「この動画の解析をお願いします」という連絡を GAS に送る。
 * GAS はそれを管理用スプシに1行追加し、Colab がその行を見て解析する。
 *
 * serves（何秒目に IN/FAULT か）は依頼者に見せる必要がないので、
 * Drive にファイルとして置かず、ここで直接スプシへ送る。
 */

import type { TrackerSession } from "@/lib/api";

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

  if (!isGasConfigured()) {
    throw new Error("解析の依頼先（GAS の URL）が設定されていません");
  }
  if (!session.drive_file) {
    throw new Error("動画がまだアップロードされていません");
  }

  // GAS に送る中身。サーブ一覧は Colab が使う最小限の項目（結果と秒数）だけにする
  const body = {
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

  let res: Response;
  try {
    res = await fetch(GAS_URL, {
      method: "POST",
      // ★ text/plain にするのがポイント。
      //   application/json にすると、ブラウザが本番の前に「送ってもいい？」という
      //   確認（プリフライト / OPTIONS リクエスト）を送るが、GAS はそれに答えられず失敗する。
      //   text/plain なら確認なしで送れて、GAS 側では e.postData.contents で JSON を読める。
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("解析の依頼を送れませんでした。通信環境を確認してください。");
  }

  if (!res.ok) {
    throw new Error(`解析の依頼に失敗しました（${res.status}）`);
  }
  const data: GasResponse = await res.json();
  if (!data.ok) {
    // GAS 側のエラーメッセージ（例：ログイン情報を確認できません）をそのまま見せる
    throw new Error(data.error ?? "解析の依頼に失敗しました");
  }
}
