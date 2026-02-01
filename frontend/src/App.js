import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const API_BASE = "https://softtennis-zzdz.onrender.com";

  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [menuDetails, setMenuDetails] = useState({});
  const [loadingDetails, setLoadingDetails] = useState({});

  // ファイル選択
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
    setMenuDetails({});
  };

  // 動画解析
  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setMenuDetails({});

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${API_BASE}/analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 5 * 60 * 1000,
      });

      console.log("API RESULT:", res.data);
      setResult(res.data);

    } catch (err) {
      if (err.response?.data?.detail) {
        setError("解析に失敗しました: " + err.response.data.detail);
      } else {
        setError("解析に失敗しました。もう一度お試しください。");
      }
      setResult(null);

    } finally {
      setLoading(false);
    }
  };

  // 最初のメニュー詳細を自動取得
  useEffect(() => {
    if (!result?.menu?.length) return;

    const firstMenu = result.menu[0];

    if (menuDetails[firstMenu] || loadingDetails[firstMenu]) return;

    setLoadingDetails((prev) => ({ ...prev, [firstMenu]: true }));

    axios
      .post(
        `${API_BASE}/menu-detail`,
        {
          menu_name: firstMenu,
          diagnosis: result.diagnosis,
        },
        { headers: { "Content-Type": "application/json" } }
      )
      .then((res) => {
        setMenuDetails((prev) => ({
          ...prev,
          [firstMenu]: res.data.detail,
        }));
      })
      .catch(() => {
        setError("メニュー詳細の取得に失敗しました。");
      })
      .finally(() => {
        setLoadingDetails((prev) => {
          const copy = { ...prev };
          delete copy[firstMenu];
          return copy;
        });
      });
  }, [result]);

  // メニュー詳細の開閉
  const handleGetMenuDetail = async (menuName) => {
    if (menuDetails[menuName]) {
      setMenuDetails((prev) => {
        const copy = { ...prev };
        delete copy[menuName];
        return copy;
      });
      return;
    }

    setLoadingDetails((prev) => ({ ...prev, [menuName]: true }));

    try {
      const res = await axios.post(
        `${API_BASE}/menu-detail`,
        {
          menu_name: menuName,
          diagnosis: result.diagnosis,
        },
        { headers: { "Content-Type": "application/json" } }
      );

      setMenuDetails((prev) => ({
        ...prev,
        [menuName]: res.data.detail,
      }));
    } catch {
      setError("メニュー詳細の取得に失敗しました。");
    } finally {
      setLoadingDetails((prev) => {
        const copy = { ...prev };
        delete copy[menuName];
        return copy;
      });
    }
  };

  return (
    <div className="app-container">
      <div className="app-content">
        {/* ヘッダー */}
        <header className="app-header">
          <h1 className="app-title">ソフトテニス サーブフォームAIコーチ</h1>
          <p className="app-subtitle">
            動画をアップロードして、あなたのサーブを分析しましょう
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

        {/* スコア */}
        {result && (
          <p className="score-text">
            スコア：{result.diagnosis.player.serve_score}点
          </p>
        )}

        {/* フォーム比較 */}
        {result?.ideal_image && result?.user_image && (
          <div className="result-card">
            <h2 className="result-title">フォーム比較（理想 vs あなた）</h2>

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

        {/* 練習メニュー */}
        {result?.menu?.length > 0 && (
          <div className="result-card">
            <h2 className="result-title">おすすめ練習メニュー</h2>

            {result.menu.map((menuName, i) => (
              <div key={i} className="menu-item">
                <div className="menu-item-header">
                  <span>{menuName}</span>

                  <button
                    onClick={() => handleGetMenuDetail(menuName)}
                    disabled={loadingDetails[menuName]}
                  >
                    {loadingDetails[menuName]
                      ? "読み込み中…"
                      : menuDetails[menuName]
                      ? "閉じる"
                      : "詳細を見る"}
                  </button>
                </div>

                {menuDetails[menuName] && (
                  <div className="menu-detail-content">
                    {menuDetails[menuName]}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* AIアドバイス */}
        {result?.ai_text && (
          <div className="result-card">
            <h2 className="result-title">AIコーチからのアドバイス</h2>
            <p>{result.ai_text}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
