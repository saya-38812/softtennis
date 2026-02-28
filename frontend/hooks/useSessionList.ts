"use client";

import { useEffect, useState, useCallback } from "react";
import { listTrackerSessions, deleteTrackerSession, type TrackerSession } from "@/lib/api";

export function useSessionList() {
  const [sessions, setSessions] = useState<TrackerSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    listTrackerSessions()
      .then(setSessions)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        await deleteTrackerSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      } catch (e) {
        setError(e instanceof Error ? e.message : "Delete failed");
      }
    },
    []
  );

  return { sessions, loading, error, refresh, deleteSession };
}
