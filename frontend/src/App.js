import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false); // 解析中フラグ
  const [error, setError] = useState(null); // エラー表示用
  const [menuDetails, setMenuDetails] = useState({}); // メニュー詳細を保存
  const [loadingDetails, setLoadingDetails] = useState({}); // 各メニューの詳細読み込み状態
  const API_BASE = "https://softtennis-zzdz.onrender.com";

  // ファイル選択時のハンドラ
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null); // ファイル変更時は前回結果をクリア
    setError(null);
  };

  // 解析リクエスト送信
  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setMenuDetails({}); // メニュー詳細をリセット
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await axios.post(
        '${API_BASE}/analyze',
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 5 * 60 * 1000, // 5分
        }
      );
      setResult(res.data);
    } catch (err) {
      // エラー詳細も表示
      if (err.response && err.response.data && err.response.data.detail) {
        setError("解析に失敗しました: " + err.response.data.detail);
      } else {
        setError("解析に失敗しました。もう一度お試しください。");
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  // 解析結果が取得されたら、最初のメニューの詳細を自動取得
  useEffect(() => {
    if (result && Array.isArray(result.menu) && result.menu.length > 0) {
      const firstMenu = result.menu[0];
      // 既に詳細が取得済みの場合はスキップ
      if (!menuDetails[firstMenu] && !loadingDetails[firstMenu]) {
        // 詳細を取得
        setLoadingDetails(prev => ({ ...prev, [firstMenu]: true }));
        axios.post(
          '${API_BASE}/menu-detail',
          {
            menu_name: firstMenu,
            diagnosis: result.diagnosis
          },
          {
            headers: { "Content-Type": "application/json" },
            timeout: 30000, // 30秒
          }
        )
        .then(res => {
          setMenuDetails(prev => ({
            ...prev,
            [firstMenu]: res.data.detail
          }));
        })
        .catch(err => {
          setError("メニュー詳細の取得に失敗しました。もう一度お試しください。");
        })
        .finally(() => {
          setLoadingDetails(prev => {
            const newDetails = { ...prev };
            delete newDetails[firstMenu];
            return newDetails;
          });
        });
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]); // resultが変更されたときに実行

  // メニュー詳細を取得
  const handleGetMenuDetail = async (menuName, autoLoad = false) => {
    // 自動読み込みの場合は表示/非表示の切り替えをしない
    if (!autoLoad) {
      // 既に詳細が取得済みの場合は表示/非表示を切り替え
      if (menuDetails[menuName]) {
        setMenuDetails(prev => {
          const newDetails = { ...prev };
          delete newDetails[menuName];
          return newDetails;
        });
        return;
      }
    }

    // 詳細を取得
    setLoadingDetails(prev => ({ ...prev, [menuName]: true }));
    try {
      const res = await axios.post(
        '${API_BASE}/menu-detail',
        {
          menu_name: menuName,
          diagnosis: result.diagnosis
        },
        {
          headers: { "Content-Type": "application/json" },
          timeout: 30000, // 30秒
        }
      );
      setMenuDetails(prev => ({
        ...prev,
        [menuName]: res.data.detail
      }));
    } catch (err) {
      setError("メニュー詳細の取得に失敗しました。もう一度お試しください。");
    } finally {
      setLoadingDetails(prev => {
        const newDetails = { ...prev };
        delete newDetails[menuName];
        return newDetails;
      });
    }
  };

  return (
    <div className="app-container">
      <div className="app-content">
        <header className="app-header">
          <h1 className="app-title">ソフトテニス　サーブフォームAIコーチ</h1>
          <p className="app-subtitle">動画をアップロードして、あなたのサーブを分析しましょう</p>
        </header>

        <div className="upload-section">
          <div className="file-input-wrapper">
            <label htmlFor="file-input" className="file-label">
              <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
              <span className="file-label-text">
                {file ? file.name : "動画ファイルを選択"}
              </span>
            </label>
            <input
              id="file-input"
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              disabled={loading}
              className="file-input"
            />
          </div>
          
          <button
            onClick={handleAnalyze}
            className="analyze-button"
            disabled={loading || !file}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>解析中…</span>
              </>
            ) : (
              "解析を開始"
            )}
          </button>
        </div>

        {error && (
          <div className="error-message">
            <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            <span>{error}</span>
          </div>
        )}

<p className="score-text">
  スコア：{result.diagnosis.player.serve_score}点
</p>


              {/* ============================= */}
{/* フォーム比較カード（MVP） */}
{/* ============================= */}
{result.ideal_image && result.user_image && (
  <div className="result-card">
    <h2 className="result-title">
      フォーム比較（理想 vs あなた）
    </h2>

    {/* 改善ポイント */}
    <p className="focus-label">
      改善ポイント：{result.focus_label}
    </p>
    <p className="focus-message">
      {result.message}
    </p>

    {/* 左右比較 */}
    <div className="compare-grid">
      {/* 理想フォーム */}
      <div className="compare-box">
        <h3 className="compare-title">理想フォーム</h3>
        <img
          src={`${API_BASE}${result.ideal_image}`}
          alt="ideal form"
          className="compare-img"
        />
      </div>

      {/* あなたフォーム */}
      <div className="compare-box">
        <h3 className="compare-title">あなたのフォーム</h3>
        <img
          src={`${API_BASE}${result.user_image}`}
          alt="your form"
          className="compare-img"
        />
      </div>
    </div>
  </div>
)}


        {result && (
          <div className="results-section">
            <div className="result-card">
              <h2 className="result-title">
                <svg className="result-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                  <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                診断結果
              </h2>
            </div>

            {Array.isArray(result.menu) && result.menu.length > 0 && (
              <div className="result-card">
                <h2 className="result-title">
                  <svg className="result-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                  </svg>
                  おすすめ練習メニュー
                </h2>
                <div className="menu-list">
                  {(() => {
                    const firstMenu = result.menu[0];
                    return (
                      <div className="menu-item">
                        <div className="menu-item-header">
                          <span className="menu-number">1</span>
                          <span className="menu-text">{firstMenu}</span>
                          <button
                            className="detail-button"
                            onClick={() => handleGetMenuDetail(firstMenu)}
                            disabled={loadingDetails[firstMenu]}
                          >
                            {loadingDetails[firstMenu] ? (
                              <>
                                <span className="spinner-small"></span>
                                <span>読み込み中...</span>
                              </>
                            ) : menuDetails[firstMenu] ? (
                              <>
                                <svg className="detail-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <polyline points="18 15 12 9 6 15"></polyline>
                                </svg>
                                <span>詳細を閉じる</span>
                              </>
                            ) : (
                              <>
                                <svg className="detail-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                                <span>詳細を見る</span>
                              </>
                            )}
                          </button>
                        </div>
                        {menuDetails[firstMenu] && (
                          <div className="menu-detail-content">
                            <div className="menu-detail-text" dangerouslySetInnerHTML={{ __html: menuDetails[firstMenu].replace(/\n/g, '<br />') }} />
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              </div>
            )}

            {result.ai_text && (
              <div className="result-card ai-advice-card">
                <h2 className="result-title">
                  <svg className="result-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                    <path d="M2 17l10 5 10-5"></path>
                    <path d="M2 12l10 5 10-5"></path>
                  </svg>
                  AIコーチからのアドバイス
                </h2>
                <div className="ai-advice-content">
                  <p>{result.ai_text}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
