import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

const TODAY_STR = new Date().toLocaleDateString();

function App() {
    const API_BASE = import.meta.env.VITE_API_BASE ||
        (window.location.hostname === "localhost" ? "http://localhost:8000" : "https://softtennis-zzdz.onrender.com");

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [practiceCount, setPracticeCount] = useState(0);
    const [sessionId, setSessionId] = useState(null);
    const [analyzingStep, setAnalyzingStep] = useState(0);
    const [progressPercent, setProgressPercent] = useState(0);

    const [isPlaying, setIsPlaying] = useState(true);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const videoRef = useRef(null);
    const fileInputRef = useRef(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = 0.5;
        }
    }, [result]);


    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (!file) return;

        handleAnalyze(file);

        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleAnalyze = async (file) => {
        setLoading(true);
        setError(null);
        setResult(null);
        setProgressPercent(0);

        const formData = new FormData();
        formData.append("file", file);

        const headers = {};
        if (sessionId) headers["X-Session-ID"] = sessionId;

        try {
            setAnalyzingStep(1);
            const res = await fetch(`${API_BASE}/analyze`, {
                method: "POST",
                headers,
                body: formData,
            });

            if (!res.ok) {
                const errBody = await res.json().catch(() => ({}));
                throw new Error(errBody.message || "解析に失敗しました。");
            }

            const newSessionId = res.headers.get("x-session-id");
            if (newSessionId && !sessionId) {
                setSessionId(newSessionId);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let gotResult = false;

            const processLine = (line) => {
                if (!line.startsWith("data: ")) return;
                try {
                    const payload = JSON.parse(line.slice(6));

                    if (payload.type === "progress") {
                        setProgressPercent(payload.percent);
                    } else if (payload.type === "error") {
                        setError(payload.message);
                    } else if (payload.type === "result" && payload.status === "complete") {
                        setProgressPercent(100);
                        setResult(payload);
                        setPracticeCount(prev => prev + 1);
                        gotResult = true;
                        if (payload.session_id && !sessionId) {
                            setSessionId(payload.session_id);
                        }
                    }
                } catch (parseErr) {
                    console.warn("SSE parse error:", parseErr, line);
                }
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    // flush remaining buffer
                    if (buffer.trim()) {
                        processLine(buffer.trim());
                    }
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    processLine(line);
                }
            }

            if (!gotResult && !error) {
                setError("解析結果を受信できませんでした。もう一度お試しください。");
            }
        } catch (err) {
            setError(err.message || "解析に失敗しました。");
        } finally {
            setLoading(false);
            setAnalyzingStep(0);
        }
    };

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) videoRef.current.pause();
            else videoRef.current.play();
            setIsPlaying(!isPlaying);
        }
    };

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime);
            setDuration(videoRef.current.duration);
        }
    };

    const handleSeek = (e) => {
        if (videoRef.current && duration) {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const clickedPos = x / rect.width;
            videoRef.current.currentTime = clickedPos * duration;
        }
    };

    const formatTime = (time) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    const handleReset = () => {
        setResult(null);
        setSessionId(null);
        window.scrollTo({ top: 0, behavior: "smooth" });
    };

    const handleClearAll = async () => {
        if (!window.confirm("全ての過去データと診断結果を完全に削除しますか？\nこの操作は取り消せません。")) {
            return;
        }

        try {
            await axios.post(`${API_BASE}/reset-all`);
            setResult(null);
            setSessionId(null);
            setPracticeCount(0);
            alert("全てのデータを削除しました。");
        } catch (err) {
            console.error(err);
            alert("データの削除に失敗しました。");
        }
    };

    const clampScore = (v) => Math.min(98, Math.max(5, Math.round(Number(v) || 0)));

    if (!result) {
        return (
            <div className="app-container">
                <header className="app-header">
                    <div className="header-left">
                        <h1 className="app-title">サーブノート</h1>
                        <span className="app-subtitle">Analysis & Coaching</span>
                    </div>
                    <div className="header-right">
                        <button className="reset-history-btn" onClick={handleClearAll} title="データを全てリセット">🗑️</button>
                        <div className="user-profile">👤</div>
                    </div>
                </header>

                <div className="empty-state animate-in">
                    <label className="upload-circle">
                        <span>{loading ? "⌛" : "📹"}</span>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="video/*"
                            onChange={handleFileChange}
                            disabled={loading}
                            className="file-input"
                        />
                    </label>

                    <div className="upload-guide">
                        <p className="guide-main">{loading ? "AIが解析しています..." : "動画をアップロード"}</p>
                        <p className="guide-sub">{loading
                            ? (progressPercent < 65 ? "骨格の動きを検出しています"
                                : progressPercent < 70 ? "指標を計算しています"
                                    : progressPercent < 95 ? "解析動画を生成しています"
                                        : "仕上げ中です")
                            : "サーブの動画を1本選んでください"}</p>
                    </div>

                    {loading && (
                        <div className="analyzing-section">
                            <div className="analyzing-percent">{progressPercent}%</div>
                            <div className="analyzing-progress-bar">
                                <div className="analyzing-progress-fill-real" style={{ width: `${progressPercent}%` }}></div>
                            </div>
                        </div>
                    )}

                    {error && <p className="error-message">{error}</p>}
                </div>
            </div>
        );
    }

    // 解析完了後の表示 — サーバーで正規化済みの 0-100 スコアを使用
    const ns = result.normalized_scores || {};
    const heightScore = clampScore(ns.impact_height);
    const elbowScore = clampScore(ns.elbow_angle);
    const swayScore = clampScore(ns.body_sway);
    const waistScore = clampScore(ns.waist_speed);
    const weightScore = clampScore(ns.weight_transfer);
    const overallScore = Math.round((heightScore + elbowScore + swayScore + waistScore + weightScore) / 5);

    return (
        <div className="app-container">
            <header className="app-header">
                <div className="header-left">
                    <h1 className="app-title">サーブノート</h1>
                    <span className="app-subtitle">{TODAY_STR} ANALYSIS</span>
                </div>
                <div className="header-right">
                    <button className="reset-history-btn" onClick={handleClearAll} title="データを全てリセット">🗑️</button>
                    <div className="user-profile">👤</div>
                </div>
            </header>

            {/* Video Card */}
            <div className="video-section animate-in">
                {result.user_video ? (
                    <video
                        ref={videoRef}
                        src={`${API_BASE}${result.user_video}`}
                        autoPlay
                        loop
                        muted
                        playsInline
                        className="main-video"
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                        onTimeUpdate={handleTimeUpdate}
                        onLoadedMetadata={handleTimeUpdate}
                    />
                ) : (
                    <div className="main-video" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
                        <p>Video analysis not available</p>
                    </div>
                )}
                <div className="video-overlay-tags">
                    <span className="tag tag-tracking">
                        <div className="recording-indicator" /> AI LIVE TRACKING
                    </span>
                    <span className="tag tag-frame">フレーム: インパクト</span>
                </div>
                <div className="skeleton-toggle">
                    <div className="toggle-dot" /> SKELETON ON
                </div>

                <div className="scanning-line"></div>

                <div className="video-controls">
                    <div className="progress-bar-container" onClick={handleSeek}>
                        <div className="progress-bar-fill" style={{ width: `${(currentTime / duration) * 100 || 0}%` }} />
                        <div className="progress-handle" style={{ left: `${(currentTime / duration) * 100 || 0}%` }} />
                    </div>
                    <div className="control-buttons">
                        <span className="playback-time">{formatTime(currentTime)} / {formatTime(duration)}</span>
                        <div className="main-controls">
                            <button className="control-btn" onClick={togglePlay}>
                                {isPlaying ? "⏸" : "▶️"}
                            </button>
                        </div>
                        <span className="playback-speed" style={{ color: 'var(--accent-cyan)', fontWeight: 'bold' }}>0.25x Slow</span>
                    </div>
                </div>
            </div>

            {/* Overall & Improvement */}
            <div className="summary-grid animate-in" style={{ animationDelay: '0.1s' }}>
                <div className="score-card">
                    <div className="card-title">Overall Score</div>
                    <div className="overall-score-box">
                        <div className="score-value-container">
                            <span className="score-main">{overallScore}</span>
                            <span className="score-total">/100</span>
                        </div>
                        <div className="score-improvement">{result.improvement_message || "前回比 --"}</div>
                    </div>
                </div>
                <div className="score-card">
                    <div className="card-title">改善ポイント</div>
                    <div className="point-text">{result.focus_label}</div>
                    <div className="comparison-stats">
                        <div className="stat-item">
                            <span className="stat-label">{result.comparison?.label_ideal || "Ideal"}</span>
                            <span className="stat-value ideal">{result.comparison?.value_ideal || "2.7m"}</span>
                        </div>
                        <div className="stat-item">
                            <span className="stat-label">{result.comparison?.label_user || "You"}</span>
                            <span className="stat-value user">{result.comparison?.value_user || "2.2m"}</span>
                        </div>
                    </div>
                    <div className="action-tip">{result.comparison?.action_tip || "もっと高い打点を意識しましょう"}</div>
                </div>
            </div>

            {/* Technical Breakdown */}
            <div className="breakdown-card animate-in" style={{ animationDelay: '0.2s' }}>
                <div className="card-title">Technical Breakdown</div>
                <div className="breakdown-list">
                    <div className="breakdown-item">
                        <div className="item-header">
                            <span className="item-name">トス高さ</span>
                            <span className={`item-score ${heightScore < 70 ? "warning" : "good"}`}>{heightScore}</span>
                        </div>
                        <div className="bar-container">
                            <div className="bar-fill" style={{ width: `${heightScore}%`, background: 'var(--accent-orange)' }} />
                        </div>
                    </div>
                    <div className="breakdown-item">
                        <div className="item-header">
                            <span className="item-name">肘の余裕</span>
                            <span className={`item-score ${elbowScore < 70 ? "warning" : "good"}`}>{elbowScore}</span>
                        </div>
                        <div className="bar-container">
                            <div className="bar-fill" style={{ width: `${elbowScore}%`, background: 'var(--accent-green)' }} />
                        </div>
                    </div>
                    <div className="breakdown-item">
                        <div className="item-header">
                            <span className="item-name">体の安定</span>
                            <span className={`item-score ${swayScore < 70 ? "warning" : "good"}`}>{swayScore}</span>
                        </div>
                        <div className="bar-container">
                            <div className="bar-fill" style={{ width: `${swayScore}%`, background: 'var(--accent-green)' }} />
                        </div>
                    </div>
                    <div className="breakdown-item">
                        <div className="item-header">
                            <span className="item-name">腰のキレ</span>
                            <span className={`item-score ${waistScore < 70 ? "warning" : "good"}`}>{waistScore}</span>
                        </div>
                        <div className="bar-container">
                            <div className="bar-fill" style={{ width: `${waistScore}%`, background: 'var(--accent-cyan)' }} />
                        </div>
                    </div>
                    <div className="breakdown-item">
                        <div className="item-header">
                            <span className="item-name">体重移動</span>
                            <span className={`item-score ${weightScore < 70 ? "warning" : "good"}`}>{weightScore}</span>
                        </div>
                        <div className="bar-container">
                            <div className="bar-fill" style={{ width: `${weightScore}%`, background: 'var(--accent-green)' }} />
                        </div>
                    </div>
                </div>
            </div>

            {/* AI Coach */}
            <div className="coach-section animate-in" style={{ animationDelay: '0.3s' }}>
                <div className="coach-header">
                    <span className="coach-tag">AI COACH</span>
                    <span className="coach-status">ONLINE ⚡</span>
                </div>
                <p className="coach-message">
                    {result.ai_text} {result.practice}
                </p>
            </div>

            {/* Footer Buttons */}
            <div className="footer-buttons animate-in" style={{ animationDelay: '0.4s' }}>
                <button className="btn btn-primary" onClick={handleReset}>もう一回撮影</button>
            </div>
        </div>
    );
}

export default App;
