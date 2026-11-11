/**
 * Google ログイン（OAuth）を扱うモジュール
 *
 * Google Identity Services（GIS）というGoogle公式のライブラリを使い、
 * 「アクセストークン」を取得する。アクセストークンは、
 * Drive API を呼ぶときに「この人の許可をもらっています」と示す通行証のようなもの。
 *
 * - トークンの有効期限は約1時間。切れたらもう一度ログインボタンを押してもらう。
 * - クライアントシークレットは使わない（ブラウザだけで完結する方式のため）。
 */

// ------------------------------------------------------------
// 設定値
// ------------------------------------------------------------

/** .env.local に書いたクライアントID（NEXT_PUBLIC_ が付いた変数はブラウザから読める） */
const CLIENT_ID = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID ?? "";

/**
 * アプリに許可してもらう権限（スコープ）
 * - drive.file : このアプリが作ったファイル・フォルダだけ読み書きできる（範囲が狭くて安全）
 * - openid email : ログインした人のメールアドレスを知るため（依頼者を区別するのに使う）
 */
const SCOPES = [
  "https://www.googleapis.com/auth/drive.file",
  "openid",
  "email",
].join(" ");

/** GIS ライブラリの読み込み元URL */
const GIS_SCRIPT_URL = "https://accounts.google.com/gsi/client";

/** ページを再読み込みしてもログイン状態を保つための保存キー */
const TOKEN_STORAGE_KEY = "servenote.googleToken";

// ------------------------------------------------------------
// 型定義
// ------------------------------------------------------------

/** ログインに成功したときに手元に持っておく情報 */
export interface GoogleToken {
  accessToken: string;
  /** 有効期限（ミリ秒のタイムスタンプ） */
  expiresAt: number;
  /** ログインした人のメールアドレス */
  email: string;
}

/**
 * GIS ライブラリの型（必要な部分だけ自分で書いている）
 * ※ @types/google.accounts というパッケージもあるが、
 *   使う関数が少ないので依存を増やさずに最小限だけ定義する。
 */
interface TokenResponse {
  access_token?: string;
  expires_in?: number; // 有効期間（秒）
  error?: string;
  error_description?: string;
}
interface TokenClient {
  requestAccessToken: (options?: { prompt?: string }) => void;
}
declare global {
  interface Window {
    google?: {
      accounts: {
        oauth2: {
          initTokenClient: (config: {
            client_id: string;
            scope: string;
            callback: (response: TokenResponse) => void;
            error_callback?: (error: { type: string; message?: string }) => void;
          }) => TokenClient;
          revoke: (accessToken: string, done?: () => void) => void;
        };
      };
    };
  }
}

// ------------------------------------------------------------
// 内部処理
// ------------------------------------------------------------

/**
 * GIS ライブラリ（外部の <script>）を読み込む。
 * すでに読み込み済みなら何もしない。
 * Promise を返すので、await で「読み込み終わるまで待つ」ことができる。
 */
let scriptPromise: Promise<void> | null = null;
function loadGisScript(): Promise<void> {
  if (window.google?.accounts?.oauth2) return Promise.resolve();
  // 2回呼ばれても <script> を二重に追加しないよう、最初の Promise を使い回す
  if (scriptPromise) return scriptPromise;

  scriptPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = GIS_SCRIPT_URL;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => {
      scriptPromise = null; // 次回また読み込みを試せるようにする
      reject(new Error("Googleログインの読み込みに失敗しました。通信環境を確認してください。"));
    };
    document.head.appendChild(script);
  });
  return scriptPromise;
}

/** アクセストークンを使って、ログインした人のメールアドレスを取得する */
async function fetchEmail(accessToken: string): Promise<string> {
  const res = await fetch("https://www.googleapis.com/oauth2/v3/userinfo", {
    headers: { Authorization: `Bearer ${accessToken}` },
  });
  if (!res.ok) {
    throw new Error("アカウント情報の取得に失敗しました");
  }
  const data: { email?: string } = await res.json();
  return data.email ?? "";
}

// ------------------------------------------------------------
// 外部に公開する関数
// ------------------------------------------------------------

/** クライアントIDが設定されているか（未設定ならログインボタンを出さない判断に使う） */
export function isGoogleConfigured(): boolean {
  return CLIENT_ID.length > 0;
}

/**
 * Google ログインを行い、トークンを返す。
 * ※ ポップアップが開くので、必ずボタンのクリックなど「ユーザーの操作」から呼ぶこと。
 *   （ブラウザは、勝手に開くポップアップをブロックするため）
 */
export async function signInWithGoogle(): Promise<GoogleToken> {
  if (!isGoogleConfigured()) {
    throw new Error("GoogleのクライアントIDが設定されていません（.env.local を確認）");
  }
  await loadGisScript();

  // GIS はコールバック方式なので、Promise で包んで await できる形にする
  const response = await new Promise<TokenResponse>((resolve, reject) => {
    const client = window.google!.accounts.oauth2.initTokenClient({
      client_id: CLIENT_ID,
      scope: SCOPES,
      callback: resolve,
      // ポップアップを閉じられた・ブロックされたときに呼ばれる
      error_callback: (err) => {
        if (err.type === "popup_closed") {
          reject(new Error("ログインがキャンセルされました"));
        } else if (err.type === "popup_failed_to_open") {
          reject(new Error("ポップアップがブロックされました。ブラウザの設定で許可してください。"));
        } else {
          reject(new Error(`ログインに失敗しました（${err.type}）`));
        }
      },
    });
    client.requestAccessToken();
  });

  if (response.error || !response.access_token) {
    throw new Error(`ログインに失敗しました（${response.error_description ?? response.error ?? "不明なエラー"}）`);
  }

  const email = await fetchEmail(response.access_token);
  const token: GoogleToken = {
    accessToken: response.access_token,
    // expires_in は秒なので、ミリ秒に直して現在時刻に足す。
    // 期限ギリギリで失敗しないよう、1分早めに「切れた」扱いにする
    expiresAt: Date.now() + ((response.expires_in ?? 3600) - 60) * 1000,
    email,
  };
  saveToken(token);
  return token;
}

/** ログアウトする（Google側の許可も取り消す） */
export function signOutFromGoogle(token: GoogleToken | null): void {
  if (token && window.google?.accounts?.oauth2) {
    window.google.accounts.oauth2.revoke(token.accessToken);
  }
  clearToken();
}

/**
 * 保存しておいたトークンを読み込む。期限切れなら null。
 * sessionStorage は「タブを閉じると消える」保存場所なので、
 * localStorage より長く残らず、トークンの保存先として安全寄り。
 */
export function loadSavedToken(): GoogleToken | null {
  try {
    const raw = window.sessionStorage.getItem(TOKEN_STORAGE_KEY);
    if (!raw) return null;
    const token: GoogleToken = JSON.parse(raw);
    if (Date.now() >= token.expiresAt) {
      clearToken(); // 期限切れのものは捨てる
      return null;
    }
    return token;
  } catch {
    return null;
  }
}

function saveToken(token: GoogleToken): void {
  try {
    window.sessionStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(token));
  } catch {
    // 保存できなくても、このページを開いている間は使えるので無視する
  }
}

function clearToken(): void {
  try {
    window.sessionStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // 何もしない
  }
}
