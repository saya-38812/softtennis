"use client";

/**
 * 解析を依頼したあと、結果が出るまでの「状況」を表示する部品
 *
 * 画面を開いたとき（とボタンを押したとき）に GAS へ問い合わせて、
 *   - 解析待ち / 解析中 → その旨を表示
 *   - 完了               → 結果を端末に保存して、親に知らせる（親が結果表示に切り替える）
 *   - エラー             → 理由を表示
 *
 * ※ 一定時間ごとに自動で問い合わせる（ポーリング）ことはしていない。
 *   解析は運営者が Colab を動かしたときに行われるので、数秒おきに聞いても無駄が多いため。
 */

import { useCallback, useEffect, useState } from "react";
import type { TrackerSession } from "@/lib/api";
import { useGoogleAuth } from "@/contexts/GoogleAuthContext";
import { fetchAnalysisResults, type RemoteAnalysisStatus } from "@/lib/gasClient";
import { saveLocalAnalysis } from "@/lib/localSession";

interface AnalysisStatusPanelProps {
  session: TrackerSession;
  /** 結果が届いたら、保存後の最新セッションを親に渡す */
  onCompleted: (session: TrackerSession) => void;
}

/** 状況ごとの表示文 */
const STATUS_MESSAGES: Record<string, string> = {
  解析待ち: "⏳ 解析待ちです。コーチが解析すると、ここに結果が表示されます。",
  解析中: "🔄 ただいま解析中です。しばらくしてから「最新の状況を確認」を押してください。",
};

export function AnalysisStatusPanel({ session, onCompleted }: AnalysisStatusPanelProps) {
  const { token, busy: authBusy, signIn } = useGoogleAuth();
  const [remote, setRemote] = useState<RemoteAnalysisStatus | null>(null);
  const [checking, setChecking] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  /** GAS に解析状況を問い合わせる */
  const check = useCallback(async () => {
    if (!token) return;
    setChecking(true);
    setError(null);
    try {
      const results = await fetchAnalysisResults({
        accessToken: token.accessToken,
        sessionIds: [session.id],
      });
      const status = results[session.id] ?? null;
      setRemote(status);
      setCheckedAt(new Date());
      // 完了していたら端末に保存して、結果表示に切り替えてもらう
      if (status?.status === "完了" && status.result) {
        onCompleted(saveLocalAnalysis(session.id, status.result));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "解析状況を確認できませんでした");
    } finally {
      setChecking(false);
    }
  }, [token, session.id, onCompleted]);

  // ログイン済みなら、画面を開いたときに1回確認する
  useEffect(() => {
    check();
  }, [check]);

  // ---- 表示 ----
  let message: React.ReactNode;
  if (!token) {
    message = "解析結果を確認するには、Google にログインしてください。";
  } else if (checking && !remote) {
    message = "解析状況を確認しています…";
  } else if (remote?.status === "エラー") {
    message = (
      <span style={{ color: "var(--error)" }}>
        ❌ 解析できませんでした。{remote.error_message}
      </span>
    );
  } else if (remote) {
    message = STATUS_MESSAGES[remote.status] ?? `状況：${remote.status}`;
  } else if (checkedAt) {
    // スプシに行が見つからない（依頼したアカウントと違うアカウントでログインしている、など）
    message = "依頼が見つかりませんでした。解析を依頼したときと同じ Google アカウントでログインしているか確認してください。";
  } else {
    message = "解析状況を確認しています…";
  }

  return (
    <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
      <p style={{ margin: "0 0 0.6rem 0", fontSize: "0.9rem", lineHeight: 1.6 }}>{message}</p>

      {session.drive_file?.webViewLink && (
        <p style={{ margin: "0 0 0.6rem 0" }}>
          <a
            href={session.drive_file.webViewLink}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)", fontSize: "0.85rem", wordBreak: "break-all" }}
          >
            📹 {session.drive_file.name}
          </a>
        </p>
      )}

      {!token ? (
        <button type="button" className="btn btn-outline btn-block" onClick={signIn} disabled={authBusy}>
          {authBusy ? "ログイン中…" : "Googleでログイン"}
        </button>
      ) : (
        <button type="button" className="btn btn-outline btn-block" onClick={check} disabled={checking}>
          {checking ? "確認中…" : "最新の状況を確認"}
        </button>
      )}

      {checkedAt && !checking && (
        <p className="text-muted" style={{ fontSize: "0.75rem", margin: "0.4rem 0 0 0", textAlign: "center" }}>
          最終確認：{checkedAt.toLocaleTimeString("ja-JP")}
        </p>
      )}
      {error && (
        <p style={{ color: "var(--error)", fontSize: "0.8rem", margin: "0.6rem 0 0 0" }}>{error}</p>
      )}
    </div>
  );
}
