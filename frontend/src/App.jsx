import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
    // APIのベースURL（Vercelの環境変数 VITE_API_BASE から取得、未設定ならRenderのデフォルトURLを使用）
    const API_BASE = import.meta.env.VITE_API_BASE || "https://softtennis-zzdz.onrender.com";

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const [improvement, setImprovement] = useState(null);
    const [improvementMessage, setImprovementMessage] = useState("");

    const [videoQueue, setVideoQueue] = useState([]); // { file, previewUrl }
    const REQUIRED_COUNT = 3;

    const [sessionId, setSessionId] = useState(null);
    const [analyzingStep, setAnalyzingStep] = useState(0);

    const fileInputRef = useRef(null);

    // ============================
    // 動画追加（キューに貯める）
    // ============================
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (!selectedFile) return;

        if (videoQueue.length >= REQUIRED_COUNT) return;

        // プレビュー用URL生成
        const previewUrl = URL.createObjectURL(selectedFile);
        setVideoQueue(prev => [...prev, { file: selectedFile, previewUrl }]);

        setResult(null);
        setError(null);

        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleRemoveVideo = (index) => {
        setVideoQueue(prev => {
            const newQueue = [...prev];
            URL.revokeObjectURL(newQueue[index].previewUrl);
            newQueue.splice(index, 1);
            return newQueue;
        });
    };

    // ============================
    // 解析（3本まとめて送信）
    // ============================
    const handleAnalyze = async () => {
        if (videoQueue.length !== REQUIRED_COUNT) return;

        setLoading(true);
        setError(null);
        setAnalyzingStep(1);

        let lastResponse = null;
        let currentSessionId = sessionId;

        try {
            for (let i = 0; i < videoQueue.length; i++) {
                setAnalyzingStep(i + 1);
                const formData = new FormData();
                formData.append("file", videoQueue[i].file);

                const config = {
                    headers: currentSessionId ? { "X-Session-ID": currentSessionId } : {}
                };

                const res = await axios.post(`${API_BASE}/analyze`, formData, config);
                lastResponse = res.data;

                // 最初のレスポンスでセッションIDを取得して保持
                if (!currentSessionId && res.headers["x-session-id"]) {
                    currentSessionId = res.headers["x-session-id"];
                    setSessionId(currentSessionId);
                }
            }

            if (lastResponse) {
                setResult(lastResponse);
                setImprovement(lastResponse.improvement);
                setImprovementMessage(lastResponse.improvement_message);
                setPracticeCount(prev => prev + 1);
            }

        } catch (err) {
            setError("解析に失敗しました。もう一度お試しください。");
            console.error(err);
        } finally {
            setLoading(false);
            setAnalyzingStep(0);
            // 動画キューをクリア（URLは解放）
            videoQueue.forEach(item => URL.revokeObjectURL(item.previewUrl));
            setVideoQueue([]);
        }
    };

    // ============================
    // 継続
    // ============================
    const handleContinue = () => {
        setResult(null);
        setError(null);
        setVideoQueue([]);
        setSessionId(null); // セッションをリセット

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

                        <div className="slot-container">
                            {[0, 1, 2].map((i) => (
                                <div key={i} className={`video-slot ${videoQueue[i] ? "has-video" : ""}`}>
                                    {videoQueue[i] ? (
                                        <div className="video-preview-wrapper">
                                            <video src={videoQueue[i].previewUrl} muted className="video-thumb" />
                                            <button className="remove-btn" onClick={() => handleRemoveVideo(i)}>×</button>
                                        </div>
                                    ) : (
                                        <div className="empty-slot">
                                            <span className="slot-num">{i + 1}</span>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>

                        <div className="progress-info">
                            {videoQueue.length} / {REQUIRED_COUNT} 本 撮影済み
                        </div>

                        {videoQueue.length < REQUIRED_COUNT && (
                            <label className="file-label">
                                <span className="upload-icon">📹</span>
                                <div>動画を選択/撮影</div>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="video/mp4,video/quicktime"
                                    onChange={handleFileChange}
                                    disabled={loading}
                                    className="file-input"
                                />
                            </label>
                        )}

                        <button
                            onClick={handleAnalyze}
                            disabled={videoQueue.length !== REQUIRED_COUNT || loading}
                            className="analyze-button"
                        >
                            {loading ? (
                                <>
                                    <div className="spinner"></div>
                                    <span>{analyzingStep}/{REQUIRED_COUNT} 解析中…</span>
                                </>
                            ) : videoQueue.length < REQUIRED_COUNT
                                ? `あと ${REQUIRED_COUNT - videoQueue.length} 本でチェック`
                                : "チェックを開始"}
                        </button>
                    </div>
                )}

                {error && <div className="error-message">{error}</div>}

                {/* 結果（行動画面） */}
                {result && (
                    <div className="result-container">

                        <div className="session-badge">
                            3本のサーブを分析した結果
                        </div>

                        {/* コーチメッセージ */}
                        {result.ai_text && (
                            <div className="coach-message">
                                {result.ai_text}
                            </div>
                        )}

                        {/* フォーカスポイント */}
                        {result.focus_label && (
                            <div className="focus-indicator">
                                改善ポイント: <span className="focus-name">{result.focus_label}</span>
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
                                <div className="image-caption">最新のサーブ</div>
                            </div>
                        )}

                        {/* 指標サマリー */}
                        <div className="scores-summary">
                            <div className="score-item">
                                <span className="score-label">打点</span>
                                <span className="score-val">{(result.scores.impact_height * 10).toFixed(1)}</span>
                            </div>
                            <div className="score-item">
                                <span className="score-label">肘</span>
                                <span className="score-val">{(result.scores.elbow_angle).toFixed(0)}°</span>
                            </div>
                            <div className="score-item">
                                <span className="score-label">ブレ</span>
                                <span className="score-val">{(result.scores.body_sway * 100).toFixed(0)}</span>
                            </div>
                        </div>

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
