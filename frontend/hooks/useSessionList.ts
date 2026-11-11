"use client";

import { useEffect, useState, useCallback } from "react";
import type { TrackerSession } from "@/lib/api";
// 履歴はサーバーではなく、ブラウザ内（localStorage）から読み込む
import { listLocalSessions, deleteLocalSession } from "@/lib/localSession";

/**
 * セッション一覧（履歴）を扱うカスタムフック
 * 画面側は「sessions を表示する」「deleteSession を呼ぶ」だけでよいように、
 * 読み込み・削除・エラー処理をここにまとめている。
 */
export function useSessionList() {
  const [sessions, setSessions] = useState<TrackerSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 一覧を読み込み直す
  const refresh = useCallback(() => {
    setError(null);
    setSessions(listLocalSessions());
    setLoading(false);
  }, []);

  // 画面を開いたときに1回読み込む
  // ※ localStorage はブラウザにしかないので、useEffect（＝ブラウザで動く処理）の中で読む
  useEffect(() => {
    refresh();
  }, [refresh]);

  // 1件削除して、画面の一覧からも取り除く
  const deleteSession = useCallback((sessionId: string) => {
    try {
      deleteLocalSession(sessionId);
      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "削除に失敗しました");
    }
  }, []);

  return { sessions, loading, error, refresh, deleteSession };
}
