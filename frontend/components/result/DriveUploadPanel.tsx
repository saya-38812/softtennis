"use client";

/**
 * 結果画面に置く「Google ドライブにアップロード」欄
 *
 * 状態によって表示が変わる：
 *   - アップロード済み       → Drive のリンクを表示
 *   - 動画が手元にない       → 「録画直後にしかアップロードできない」と案内
 *   - 未ログイン             → ログインボタン
 *   - アップロード中         → 進捗バー
 *   - それ以外               → アップロードボタン
 */

import { useState } from "react";
import type { TrackerSession } from "@/lib/api";
import { useGoogleAuth } from "@/contexts/GoogleAuthContext";
import { uploadVideoToDrive } from "@/lib/googleDrive";
import { markLocalSessionUploaded } from "@/lib/localSession";

interface DriveUploadPanelProps {
  session: TrackerSession;
  /** 録画直後（またはファイル選択直後）の動画。ページを再読み込みすると消える */
  videoBlob: Blob | null;
  /** アップロード後の最新セッションを親（結果画面）に伝える */
  onUploaded: (session: TrackerSession) => void;
}

/** バイト数を「12.3 MB」のような読みやすい形にする */
function formatSize(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DriveUploadPanel({ session, videoBlob, onUploaded }: DriveUploadPanelProps) {
  const { token, folder, busy: authBusy, signIn } = useGoogleAuth();
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  /** アップロードボタンが押されたとき */
  const handleUpload = async () => {
    if (!token || !folder || !videoBlob) return;
    setUploading(true);
    setProgress(0);
    setError(null);
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
      const updated = markLocalSessionUploaded(session.id, file);
      onUploaded(updated);
    } catch (e) {
      setError(e instanceof Error ? e.message : "アップロードに失敗しました");
    } finally {
      setUploading(false);
    }
  };

  // ---- 表示の切り替え（上から順に判定する） ----
  let content: React.ReactNode;

  if (session.drive_file) {
    // アップロード済み
    content = (
      <>
        <p style={{ margin: "0 0 0.4rem 0", color: "var(--success)", fontWeight: 600 }}>
          ✅ Google ドライブにアップロードしました
        </p>
        {session.drive_file.webViewLink && (
          <a
            href={session.drive_file.webViewLink}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "var(--accent)", fontSize: "0.875rem", wordBreak: "break-all" }}
          >
            📹 {session.drive_file.name}
          </a>
        )}
        <p className="text-muted" style={{ fontSize: "0.8rem", margin: "0.5rem 0 0 0" }}>
          フォーム解析の結果は、準備ができ次第ここに表示されます。
        </p>
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
          Google ドライブにアップロード
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
