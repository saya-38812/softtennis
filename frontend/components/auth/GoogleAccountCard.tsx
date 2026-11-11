"use client";

/**
 * ホーム画面に置く「Google アカウント」カード
 * - 未ログイン：「Googleでログイン」ボタン
 * - ログイン中：メールアドレスと「サーブノート」フォルダへのリンク
 */

import { useGoogleAuth } from "@/contexts/GoogleAuthContext";

export function GoogleAccountCard() {
  const { configured, token, folder, shareStatus, shareError, busy, error, signIn, signOut } =
    useGoogleAuth();

  // クライアントIDが未設定なら、カード自体を出さない（開発中の誤表示を防ぐ）
  if (!configured) return null;

  return (
    <div className="card" style={{ width: "100%", padding: "0.9rem", marginBottom: "1.25rem" }}>
      {token ? (
        // ---- ログイン中 ----
        <>
          <div style={{ fontSize: "0.75rem", color: "var(--muted)" }}>Google アカウント</div>
          <div style={{ fontWeight: 600, wordBreak: "break-all", marginBottom: "0.4rem" }}>
            {token.email || "ログイン中"}
          </div>
          <div style={{ fontSize: "0.85rem", marginBottom: "0.6rem" }}>
            {busy ? (
              <span className="text-muted">Drive のフォルダを準備しています…</span>
            ) : folder ? (
              <>
                保存先：{" "}
                <a href={folder.webViewLink} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)" }}>
                  📁 {folder.name}
                </a>
              </>
            ) : null}
          </div>
          {/* 運営者（コーチ）への共有状態 */}
          {!busy && folder && (
            <div style={{ fontSize: "0.8rem", marginBottom: "0.6rem" }}>
              {shareStatus === "shared" && (
                <span style={{ color: "var(--success)" }}>✓ コーチと共有済み</span>
              )}
              {shareStatus === "self" && (
                <span className="text-muted">運営者アカウントでログイン中（共有は不要です）</span>
              )}
              {shareStatus === "not_configured" && (
                <span style={{ color: "var(--warning)" }}>⚠ 共有先のコーチが設定されていません</span>
              )}
              {shareError && <span style={{ color: "var(--error)" }}>⚠ {shareError}</span>}
            </div>
          )}
          <button
            type="button"
            className="btn btn-outline"
            style={{ fontSize: "0.8rem", padding: "0.35rem 0.8rem" }}
            onClick={signOut}
          >
            ログアウト
          </button>
        </>
      ) : (
        // ---- 未ログイン ----
        <>
          <p style={{ fontSize: "0.85rem", color: "var(--muted)", margin: "0 0 0.6rem 0" }}>
            練習動画を Google ドライブに保存してフォーム解析を受けるには、ログインしてください。
          </p>
          <button
            type="button"
            className="btn btn-outline btn-block"
            onClick={signIn}
            disabled={busy}
          >
            {busy ? "ログイン中…" : "Googleでログイン"}
          </button>
        </>
      )}

      {error && (
        <p style={{ color: "var(--error)", fontSize: "0.8rem", margin: "0.6rem 0 0 0" }}>{error}</p>
      )}
    </div>
  );
}
