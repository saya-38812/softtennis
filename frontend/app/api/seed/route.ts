import { NextResponse } from "next/server";

const BACKEND_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function POST() {
  try {
    const res = await fetch(`${BACKEND_BASE}/api/seed-sample`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await res.json().catch(() => ({}));
    return NextResponse.json(data, { status: res.status });
  } catch (e) {
    return NextResponse.json(
      { created: [], message: "バックエンドに接続できません。", error: String(e) },
      { status: 502 }
    );
  }
}
