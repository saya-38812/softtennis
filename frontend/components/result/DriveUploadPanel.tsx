"use client";

/**
 * 結果画面に置く「Google ドライブにアップロード → 解析を依頼」欄
 *
 * 処理は2段階：
 *   ① 動画を Drive にアップロード（依頼者の「サーブノート」フォルダへ）
 *   ② GAS に「解析をお願いします」と連絡（管理用スプシに1行追加される）
 *
 * ① が成功して ② だけ失敗することもあるので、その場合は
 * 動画を上げ直さずに ② だけやり直せるボタンを出す。
 *
 * 状態によって表示が変わる：
 *   - 依頼済み               → 「解析待ち」と Drive のリンク
 *   - アップロード済み・未依頼 → 「解析を依頼する」ボタン（② のやり直し）
 *   - 動画が手元にない       → 「録画直後にしかアップロードできない」と案内
 *   - 未ログイン             → ログインボタン
 *   - アップロード中         → 進捗バー
 *   - それ以外               → アップロードボタン
 */

import { useState } from "react";
import type { TrackerSession } from "@/lib/api";
import { useGoogleAuth } from "@/contexts/GoogleAuthContext";
import { uploadVideoToDrive } from "@/lib/googleDrive";
import { notifyUploadToGas } from "@/lib/gasClient";
import { markLocalSessionUploaded, markLocalSessionRequested } from "@/lib/localSession";

interface DriveUploadPanelProps {
  session: TrackerSession;
  /** 録画直後（またはファイル選択直後）の動画。ページを再読み込みすると消える */
  videoBlob: Blob | null;
  /** 処理後の最新セッションを親（結果画面）に伝える */
  onUploaded: (session: TrackerSession) => void;
}

/** バイト数を「12.3 MB」のような読みやすい形にする */
function formatSize(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DriveUploadPanel({ session, videoBlob, onUploaded }: DriveUploadPanelProps) {
  const { token, folder, busy: authBusy, signIn } = useGoogleAuth();
  const [uploading, setUploading] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  /**
   * ② GAS に解析を依頼する
   * アップロード直後と、「解析を依頼する」ボタン（やり直し）の両方から呼ぶ。
   */
  const requestAnalysis = async (target: TrackerSession) => {
    if (!token || !folder) return;
    setRequesting(true);
    setError(null);
    try {
      await notifyUploadToGas({ accessToken: token.accessToken, session: target, folderId: folder.id });
      onUploaded(markLocalSessionRequested(target.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "解析の依頼に失敗しました");
    } finally {
      setRequesting(false);
    }
  };

  /** ① アップロードボタンが押されたとき：アップロード → そのまま解析を依頼 */
  const handleUpload = async () => {
    if (!token || !folder || !videoBlob) return;
    setUploading(true);
    setProgress(0);
    setError(null);
    let uploaded: TrackerSession;
    try {
      const file = await uploadVideoToDrive({
        accessToken: token.accessToken,
        folderId: folder.id,
        video: videoBlob,
        sessionId: session.id,
        recordedAt: new Date(session.created_at ?? Date.now()),
        onProgress: setProgress,
      });
      // 送れたことを端末に記録して、画面を「アップロード済み」に切り替える
      uploaded = markLocalSessionUploaded(session.id, file);
      onUploaded(uploaded);
    } catch (e) {
      setError(e instanceof Error ? e.message : "アップロードに失敗しました");
      return;
    } finally {
      setUploading(false);
    }
    // アップロードできたら、続けて解析を依頼する
    await requestAnalysis(uploaded);
  };

  // ---- Drive のリンク（アップロード済みのときに使い回す） ----
  const driveLink = session.drive_file?.webViewLink ? (
    <a
      href={session.drive_file.webViewLink}
      target="_blank"
      rel="noopener noreferrer"
      style={{ color: "var(--accent)", fontSize: "0.875rem", wordBreak: "break-all" }}
    >
      📹 {session.drive_file.name}
    </a>
  ) : null;

  // ---- 表示の切り替え（上から順に判定する） ----
  let content: React.ReactNode;

  if (session.drive_file && session.analysis_requested_at) {
    // 依頼済み
    content = (
      <>
        <p style={{ margin: "0 0 0.4rem 0", color: "var(--success)", fontWeight: 600 }}>
          ✅ 解析を依頼しました（解析待ち）
        </p>
        {driveLink}
        <p className="text-muted" style={{ fontSize: "0.8rem", margin: "0.5rem 0 0 0" }}>
          フォーム解析の結果は、準備ができ次第ここに表示されます。
        </p>
      </>
    );
  } else if (session.drive_file) {
    // アップロード済みだが、依頼がまだ（依頼に失敗した）
    content = (
      <>
        <p style={{ margin: "0 0 0.4rem 0", fontWeight: 600 }}>
          動画はアップロード済みです
        </p>
        {driveLink}
        <div style={{ marginTop: "0.75rem" }}>
          {!token || !folder ? (
            // ログインが切れていたら、先にログインしてもらう
            <button type="button" className="btn btn-outline btn-block" onClick={signIn} disabled={authBusy}>
              {authBusy ? "ログイン中…" : "Googleでログインして解析を依頼"}
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary btn-block"
              onClick={() => requestAnalysis(session)}
              disabled={requesting}
            >
              {requesting ? "依頼しています…" : "解析を依頼する"}
            </button>
          )}
        </div>
      </>
    );
  } else if (!videoBlob) {
    // 動画がメモリにない（再読み込みした・履歴から開いた）
    content = (
      <p className="text-muted" style={{ fontSize: "0.875rem", margin: 0 }}>
        この記録の動画はアップロードされていません。動画のアップロードは、録画・動画選択の直後にこの画面から行えます。
      </p>
    );
  } else if (!token || !folder) {
    // 未ログイン（またはフォルダ準備中）
    content = (
      <>
        <p style={{ fontSize: "0.875rem", margin: "0 0 0.6rem 0" }}>
          フォーム解析を受けるには、Google にログインして動画をアップロードしてください。
        </p>
        <button type="button" className="btn btn-outline btn-block" onClick={signIn} disabled={authBusy}>
          {authBusy ? "ログイン中…" : "Googleでログイン"}
        </button>
      </>
    );
  } else if (uploading) {
    // アップロード中：進捗バー
    content = (
      <>
        <p style={{ fontSize: "0.875rem", margin: "0 0 0.5rem 0" }}>
          アップロード中… {progress}%（画面を閉じないでください）
        </p>
        <div style={{ height: 8, background: "var(--bg)", borderRadius: 4, overflow: "hidden" }}>
          <div
            style={{
              width: `${progress}%`,
              height: "100%",
              background: "var(--accent)",
              transition: "width 0.2s ease",
            }}
          />
        </div>
      </>
    );
  } else {
    // アップロードできる状態
    content = (
      <>
        <p style={{ fontSize: "0.875rem", margin: "0 0 0.6rem 0" }}>
          動画（{formatSize(videoBlob.size)}）を「📁 {folder.name}」フォルダに保存して、フォーム解析を依頼します。
        </p>
        <button type="button" className="btn btn-primary btn-block" onClick={handleUpload} style={{ padding: "0.75rem" }}>
          アップロードして解析を依頼
        </button>
      </>
    );
  }

  return (
    <div className="card" style={{ padding: "1rem", marginBottom: "1rem" }}>
      {content}
      {error && (
        <p style={{ color: "var(--error)", fontSize: "0.8rem", margin: "0.6rem 0 0 0" }}>{error}</p>
      )}
    </div>
  );
}
