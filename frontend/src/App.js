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
          <h1 className="app-title">サーブフォームAIコーチ</h1>
          <p className="app-subtitle">
            動画をアップロードして改善点を確認しましょう
          </p>
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
            {/* スコア */}
            <p className="score-text">
              スコア：{result?.diagnosis?.player?.serve_score ?? "-"}点
            </p>

            {/* フォーム比較 */}
            {result.ideal_image && result.user_image && (
              <div className="result-card">
                <h2 className="result-title">
                  フォーム比較（理想 vs あなた）
                </h2>

                <p className="focus-label">
                  改善ポイント：{result.focus_label}
                </p>

                <p className="focus-message">{result.message}</p>

                <div className="compare-grid">
                  <div className="compare-box">
                    <h3>理想フォーム</h3>
                    <img
                      src={`${API_BASE}${result.ideal_image}`}
                      alt="ideal"
                      className="compare-img"
                    />
                  </div>

                  <div className="compare-box">
                    <h3>あなたのフォーム</h3>
                    <img
                      src={`${API_BASE}${result.user_image}`}
                      alt="user"
                      className="compare-img"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* 練習メニュー（1つだけ表示） */}
            {result.menu?.length > 0 && (
              <div className="result-card">
                <h2 className="result-title">今日の練習</h2>

                <p className="menu-title">{result.menu[0]}</p>

                {loadingMenu ? (
                  <p>読み込み中…</p>
                ) : (
                  <p className="menu-detail" style={{ whiteSpace: "pre-line" }}>
                    {menuDetail}
                  </p>
                )}
              </div>
            )}

            {/* AIアドバイス */}
            {result.ai_text && (
              <div className="result-card">
                <h2 className="result-title">AIコーチ</h2>

                <p style={{ whiteSpace: "pre-line" }}>
                  {result.ai_text}
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default App;
