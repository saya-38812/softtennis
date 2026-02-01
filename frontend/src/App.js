import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const API_BASE = "https://softtennis-zzdz.onrender.com";

  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [menuDetail, setMenuDetail] = useState("");
  const [loadingMenu, setLoadingMenu] = useState(false);

  // ============================
  // ファイル選択
  // ============================
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
    setMenuDetail("");
  };

  // ============================
  // 動画解析
  // ============================
  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setMenuDetail("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 5 * 60 * 1000,
      });

      setResult(res.data);

    } catch (err) {
      setError("解析に失敗しました。もう一度お試しください。");
      console.error(err);

    } finally {
      setLoading(false);
    }
  };

  // ============================
  // 最初の練習メニュー詳細を自動取得
  // ============================
  useEffect(() => {
    if (!result?.menu?.length) return;

    const firstMenu = result.menu[0];

    setLoadingMenu(true);

    axios
      .post(`${API_BASE}/menu-detail`, {
        menu_name: firstMenu,
        diagnosis: result.diagnosis,
      })
      .then((res) => {
        setMenuDetail(res.data.detail);
      })
      .catch(() => {
        setMenuDetail("詳細を取得できませんでした。");
      })
      .finally(() => {
        setLoadingMenu(false);
      });

  }, [result]);

  // ============================
  // 表示
  // ============================
  return (
    <div className="app-container">
      <div className="app-content">

        {/* ヘッダー */}
        <header className="app-header">
          <span className="tennis-ball-icon">🎾</span>
          <h1 className="app-title">ソフトテニス サーブフォームAIコーチ</h1>
        </header>

        {/* アップロード */}
        <div className="upload-section">
          <label className="file-label">
            {file ? file.name : "動画ファイルを選択"}
            <input
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              disabled={loading}
              className="file-input"
            />
          </label>

          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="analyze-button"
          >
            {loading ? "解析中…" : "解析を開始"}
          </button>
        </div>

        {/* エラー */}
        {error && <div className="error-message">{error}</div>}

        {/* 結果表示 */}
        {result && (
          <>

            {/* AIアドバイスセクション */}
            {result.ideal_image && result.user_image && result.message && (
              <div className="result-card">
                <h2 className="section-title">
                  <span className="section-icon">ℹ️</span>
                  AIアドバイス
                </h2>
                <p className="ai-advice-message">{result.message}</p>
                <div className="comparison-panels">
                  <div className="comparison-panel bad-example">
                    <div className="panel-header">
                      <span className="x-icon">✕</span>
                      <span className="panel-label">悪い例</span>
                    </div>
                    <div className="panel-content">
                      <img
                        src={`${API_BASE}${result.user_image}`}
                        alt="bad example"
                        className="comparison-img"
                      />
                      <p className="panel-description">
                        {result.focus_label && `${result.focus_label}が下がっている...`}
                      </p>
                    </div>
                  </div>
                  <div className="comparison-panel good-example">
                    <div className="panel-header">
                      <span className="check-icon">✓</span>
                      <span className="panel-label">良い例</span>
                    </div>
                    <div className="panel-content">
                      <img
                        src={`${API_BASE}${result.ideal_image}`}
                        alt="good example"
                        className="comparison-img"
                      />
                      <p className="panel-description good-description">
                        {result.focus_label && `${result.focus_label}を高く引き上げよう!`}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* アドバイスセクション */}
            {result.message && (
              <div className="result-card">
                <h2 className="section-title">アドバイス</h2>
                <p className="advice-text">
                  {result.message}
                  {result.focus_label && ` ${result.focus_label}が下がっています。${result.focus_label}をもっと高く引き上げて、打点を高くしましょう!`}
                </p>
              </div>
            )}

            {/* 練習メニューセクション */}
            {result.menu?.length > 0 && (
              <div className="result-card">
                <h2 className="section-title">
                  <span className="section-icon">📋</span>
                  練習メニュー
                </h2>
                <p className="menu-title">{result.menu[0]}</p>
                {loadingMenu ? (
                  <p className="loading-text">読み込み中…</p>
                ) : (
                  <p className="menu-detail" style={{ whiteSpace: "pre-line" }}>
                    {menuDetail}
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
