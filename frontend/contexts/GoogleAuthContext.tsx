"use client";

/**
 * Google のログイン状態をアプリ全体で共有するための Context
 *
 * Context は「アプリのどの画面からでも取り出せる共有の置き場所」。
 * ホーム画面でログインして、結果画面でアップロードする…というように、
 * 複数の画面で同じログイン情報を使いたいのでここに置く。
 */

import React, { createContext, useCallback, useContext, useEffect, useState } from "react";
import {
  isGoogleConfigured,
  loadSavedToken,
  signInWithGoogle,
  signOutFromGoogle,
  type GoogleToken,
} from "@/lib/googleAuth";
import {
  ensureAppFolder,
  shareFolderWithOperator,
  type DriveFolder,
  type ShareStatus,
} from "@/lib/googleDrive";

interface GoogleAuthContextValue {
  /** クライアントIDが設定されているか */
  configured: boolean;
  /** ログイン中ならトークン、未ログインなら null */
  token: GoogleToken | null;
  /** 依頼者の Drive にある「サーブノート」フォルダ */
  folder: DriveFolder | null;
  /** 運営者への共有の状態（未確認なら null） */
  shareStatus: ShareStatus | null;
  /** 共有に失敗したときのメッセージ */
  shareError: string | null;
  /** ログインやフォルダ準備の処理中か */
  busy: boolean;
  /** 直近のエラーメッセージ */
  error: string | null;
  signIn: () => Promise<void>;
  signOut: () => void;
}

const GoogleAuthContext = createContext<GoogleAuthContextValue | null>(null);

export function GoogleAuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<GoogleToken | null>(null);
  const [folder, setFolder] = useState<DriveFolder | null>(null);
  const [shareStatus, setShareStatus] = useState<ShareStatus | null>(null);
  const [shareError, setShareError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // サーバー側の描画ではブラウザ用の値が読めないので、画面を開いてから判定する
  const [configured, setConfigured] = useState(false);

  /** トークンを使って「サーブノート」フォルダを用意し、運営者に共有する */
  const prepareFolder = useCallback(async (t: GoogleToken) => {
    const f = await ensureAppFolder(t.accessToken);
    setFolder(f);

    // 共有に失敗しても、動画のアップロード自体はできるので処理は止めない。
    // （エラーは画面に出して、依頼者に知らせる）
    setShareError(null);
    try {
      const status = await shareFolderWithOperator(t.accessToken, f.id, t.email);
      setShareStatus(status);
    } catch (e) {
      setShareStatus(null);
      setShareError(e instanceof Error ? e.message : "コーチへの共有に失敗しました");
    }
  }, []);

  // 画面を開いたとき：前回のログイン（1時間以内）が残っていれば復元する
  useEffect(() => {
    setConfigured(isGoogleConfigured());
    const saved = loadSavedToken();
    if (!saved) return;
    setToken(saved);
    setBusy(true);
    prepareFolder(saved)
      .catch((e) => {
        // 期限切れなどで失敗したら、ログインし直してもらう
        setToken(null);
        setError(e instanceof Error ? e.message : "フォルダの確認に失敗しました");
      })
      .finally(() => setBusy(false));
  }, [prepareFolder]);

  /** ログインボタンが押されたとき：ログイン → フォルダ準備 */
  const signIn = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const t = await signInWithGoogle();
      setToken(t);
      await prepareFolder(t);
    } catch (e) {
      setError(e instanceof Error ? e.message : "ログインに失敗しました");
    } finally {
      setBusy(false);
    }
  }, [prepareFolder]);

  /** ログアウト */
  const signOut = useCallback(() => {
    signOutFromGoogle(token);
    setToken(null);
    setFolder(null);
    setShareStatus(null);
    setShareError(null);
  }, [token]);

  return (
    <GoogleAuthContext.Provider
      value={{ configured, token, folder, shareStatus, shareError, busy, error, signIn, signOut }}
    >
      {children}
    </GoogleAuthContext.Provider>
  );
}

/** 各画面からログイン情報を取り出すためのフック */
export function useGoogleAuth() {
  const ctx = useContext(GoogleAuthContext);
  if (!ctx) throw new Error("useGoogleAuth は GoogleAuthProvider の内側で使ってください");
  return ctx;
}
