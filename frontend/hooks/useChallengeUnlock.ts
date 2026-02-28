"use client";

import { useState, useEffect } from "react";
import { CHALLENGES, CHALLENGE_UNLOCK_KEY } from "@/lib/constants";

export function useChallengeUnlock() {
  const [unlockedIndex, setUnlockedIndex] = useState(0);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(CHALLENGE_UNLOCK_KEY);
      const n = raw == null ? 0 : Number(raw);
      const safe = Number.isFinite(n)
        ? Math.max(0, Math.min(CHALLENGES.length - 1, Math.floor(n)))
        : 0;
      setUnlockedIndex(safe);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(CHALLENGE_UNLOCK_KEY, String(unlockedIndex));
    } catch {
      // ignore
    }
  }, [unlockedIndex]);

  const unlockNext = () => {
    setUnlockedIndex((i) => Math.min(i + 1, CHALLENGES.length - 1));
  };

  return { unlockedIndex, setUnlockedIndex, unlockNext };
}
