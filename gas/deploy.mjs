/**
 * GAS を「今の URL のまま」新しいバージョンに更新するスクリプト
 *
 * 使い方（gas フォルダで）：
 *   npm run deploy
 *
 * やっていること：
 *   1. clasp push   … PC の Code.gs を GAS に送る（＝エディタに貼り付けて保存、と同じ）
 *   2. clasp update-deployment … 公開中の Web アプリを新しいバージョンに切り替える
 *      （＝エディタの「デプロイを管理 → 編集 → 新バージョン」と同じ。URL は変わらない）
 *
 * 公開中の Web アプリの ID（デプロイID）は、アプリが使っている GAS の URL
 *   https://script.google.com/macros/s/【デプロイID】/exec
 * の【】の部分。frontend/.env.local の NEXT_PUBLIC_GAS_URL から読み取るので、
 * 同じ値を2か所に書かなくて済む（書き間違いで別の URL を更新する事故も防げる）。
 */

import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

// ---- 1. デプロイIDを .env.local から読み取る ----
let envText;
try {
  envText = readFileSync(new URL("../frontend/.env.local", import.meta.url), "utf-8");
} catch {
  console.error("frontend/.env.local が見つかりません。NEXT_PUBLIC_GAS_URL を設定してください。");
  process.exit(1);
}
const match = envText.match(/^NEXT_PUBLIC_GAS_URL=https:\/\/script\.google\.com\/macros\/s\/([\w-]+)\/exec\s*$/m);
if (!match) {
  console.error("NEXT_PUBLIC_GAS_URL の形が想定と違います（…/macros/s/〇〇/exec の形にしてください）。");
  process.exit(1);
}
const deploymentId = match[1];

/** コマンドを実行して、その出力を画面にそのまま表示する */
function run(command) {
  console.log(`\n$ ${command}`);
  execSync(command, { stdio: "inherit" });
}

// ---- 2. push → デプロイの更新 ----
try {
  run("npx clasp push --force");
  // 説明欄に日時を入れておくと、デプロイの履歴を見たときにいつの更新かわかる
  const description = `clasp で更新 ${new Date().toLocaleString("ja-JP")}`;
  run(`npx clasp update-deployment ${deploymentId} --description "${description}"`);
  console.log("\n✅ GAS を新しいバージョンに更新しました（URL は変わっていません）");
} catch {
  console.error("\n❌ 更新に失敗しました。上のエラーメッセージを確認してください。");
  process.exit(1);
}
