import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  // APIのベースURL（Vercelの環境変数から取得、未設定ならRenderのデフォルトURLを使用）
  const API_BASE = process.env.REACT_APP_API_BASE || "https://softtennis-zzdz.onrender.com";

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [improvement, setImprovement] = useState(null);
  const [improvementMessage, setImprovementMessage] = useState("");

  const [videoQueue, setVideoQueue] = useState([]);
  const REQUIRED_COUNT = 5;

  // ★ 追加：連続練習回数
  const [practiceCount, setPracticeCount] = useState(0);

  const fileInputRef = useRef(null);

  // ============================
  // 動画追加（キューに貯める）
  // ============================
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    if (videoQueue.length >= REQUIRED_COUNT) return;

    setVideoQueue(prev => [...prev, selectedFile]);
    setResult(null);
    setError(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // ============================
  // 解析（5本まとめて送信）
  // ============================
  const handleAnalyze = async () => {
    if (videoQueue.length !== REQUIRED_COUNT) return;

    setLoading(true);
    setError(null);

    let lastResponse = null;

    try {
      for (let i = 0; i < videoQueue.length; i++) {
        const formData = new FormData();
        formData.append("file", videoQueue[i]);

        const res = await axios.post(`${API_BASE}/analyze`, formData);
        lastResponse = res.data;
      }

      if (lastResponse) {
        setResult(lastResponse);
        setImprovement(lastResponse.improvement);
        setImprovementMessage(lastResponse.improvement_message);
        setPracticeCount(prev => prev + 1); // ★ 練習回数アップ
      }

    } catch (err) {
      setError("解析に失敗しました。もう一度お試しください。");
      console.error(err);
    } finally {
      setLoading(false);
      setVideoQueue([]);
    }
  };

  // ============================
  // 継続（リセットではない）
  // ============================
  const handleContinue = () => {
    setResult(null);
    setError(null);
    setVideoQueue([]);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // ============================
  // 行動メッセージ生成
  // ============================
  const getActionMessage = () => {
    if (practiceCount === 0) return "まず感覚を作りましょう";
    if (practiceCount === 1) return "今の感覚でもう一度";

    if (improvement && Math.abs(improvement.impact_height) > 0.05)
      return "今の動き良いです";

    return "そのまま続けてください";
  };

  // ============================
  // 表示
  // ============================
  return (
    <div className="app-container">
      <div className="app-content">

        {/* ヘッダー */}
        <header className="app-header">
          <span className="tennis-ball-icon">🎾</span>
          <h1 className="app-title">サーブAIコーチ</h1>
        </header>

        {/* アップロード */}
        {!result && (
          <div className="upload-section">

            <div className="progress-info">
              撮影 {videoQueue.length} / {REQUIRED_COUNT}
            </div>

            <label className="file-label">
              動画を選択
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4"
                onChange={handleFileChange}
                disabled={loading || videoQueue.length >= REQUIRED_COUNT}
                className="file-input"
              />
            </label>

            <button
              onClick={handleAnalyze}
              disabled={videoQueue.length !== REQUIRED_COUNT || loading}
              className="analyze-button"
            >
              {loading
                ? "解析中…"
                : videoQueue.length < REQUIRED_COUNT
                  ? `あと${REQUIRED_COUNT - videoQueue.length}本`
                  : "チェック"}
            </button>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {/* 結果（行動画面） */}
        {result && (
          <div className="result-container">

            {/* コーチメッセージ */}
            {result.ai_text && (
              <div className="coach-message">
                {result.ai_text}
              </div>
            )}

            {/* フォーム画像 */}
            {result.user_image && (
              <div className="image-container">
                <img
                  src={`${API_BASE}${result.user_image}`}
                  alt="your form"
                  className="main-user-image"
                />
              </div>
            )}

            {/* 行動セクション */}
            <div className="action-section">
              <p className="short-message">{getActionMessage()}</p>
              <button className="huge-retry-button" onClick={handleContinue}>
                {practiceCount === 0 ? "はじめる" : "続ける"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;