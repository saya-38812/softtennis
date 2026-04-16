"use client";

import React, { createContext, useContext, useState, useCallback } from "react";

interface SessionVideoContextValue {
  videoBlob: Blob | null;
  setVideoBlob: (blob: Blob | null) => void;
  clearVideo: () => void;
}

const SessionVideoContext = createContext<SessionVideoContextValue | null>(null);

export function SessionVideoProvider({ children }: { children: React.ReactNode }) {
  const [videoBlob, setVideoBlobState] = useState<Blob | null>(null);
  const clearVideo = useCallback(() => setVideoBlobState(null), []);
  const setVideoBlob = useCallback((blob: Blob | null) => setVideoBlobState(blob), []);
  return (
    <SessionVideoContext.Provider value={{ videoBlob, setVideoBlob, clearVideo }}>
      {children}
    </SessionVideoContext.Provider>
  );
}

export function useSessionVideo() {
  const ctx = useContext(SessionVideoContext);
  if (!ctx) throw new Error("useSessionVideo must be used within SessionVideoProvider");
  return ctx;
}
