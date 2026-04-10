# 香港回南天預測系統：基於物理代理與機器學習的混合架構
**(Hong Kong Return-Flow Weather Prediction: A Hybrid Machine Learning Approach)**

這是一個端到端 (End-to-End) 的數據科學專案，旨在利用過往 10 年的歷史氣象數據，精準預測香港的「回南天 (HNT)」極端潮濕天氣，以協助倉儲業與一般民眾提前進行防潮決策。

## 專案背景與痛點 (Business Context)
「回南天」是華南地區特有的天氣現象，常導致室內嚴重結露，對精密電子倉儲、藝術品保存及市民生活造成巨大影響。
本專案面臨的最大挑戰在於：**香港本地的氣象數據庫中，缺失了預測回南天最核心的物理指標——「地表溫度 (Ground Temperature)」**。

為解決此痛點，本專案首創了**「物理代理模型 + 監督式學習」的混合架構 (Hybrid Modeling)**，並採用**商業導向的評估策略**，成功打造出高敏感度的預警系統。

---

## 核心技術亮點 (Key Highlights)

### 1. 穩健的數據工程與 ETL (Data Engineering)
* **自動化流水線:** 使用 Python (`Pandas`, `Glob`) 自動遍歷、提取並合併香港及周邊 6 個城市、跨度 10 年的繁雜 Excel 氣象報表。
* **防禦性清洗:** 針對真實世界數據的髒亂特徵，開發了 R 語言自定義函數 `robust_read_csv` (自動適配 GB18030/UTF-8/Big5 等 4 種編碼) 與正則表達式清洗機制。
* **物理特徵工程:** 將 360 度的風向指標利用三角函數轉化為連續的南北風分量 (`Wind_NorthSouth`)，並引入 3 天滾動平均與時間滯後特徵 (`Lag`) 以捕捉建築物牆體的「熱慣性」。

### 2. 混合建模架構 (Hybrid Modeling Strategy)
本專案不盲目依賴單一算法，而是將物理學領域知識與機器學習完美結合：
* **Phase 1 (補足缺失特徵):** 利用周邊城市（如深圳、陽江）的完整數據，訓練多元線性迴歸模型作為**物理代理模型 (Physics Proxy)**，精準推算香港缺失的「虛擬地表溫度」。
* **Phase 2 (終極預測):** 將推算出的物理特徵，結合其他氣象指標，輸入至 **C5.0 (決策樹)**、**Naive Bayes (樸素貝葉斯結合 KDE)** 與 **GLM (邏輯迴歸)** 中進行並行預測。

### 3. 商業導向的模型評估 (Business-Oriented Evaluation)
面對回南天天數極少（極度不平衡數據）的挑戰，本專案**屏棄了傳統的「準確率 (Accuracy)」迷思**：
* 引入 **約登指數 (Youden's J Statistic)** 動態尋找最佳分類閾值，在「防範風險」與「減少擾民」之間取得最佳平衡。
* 採用 **Sensitivity (敏感度)**、**Specificity (特異度)** 與 **Kappa 係數** 作為核心決策指標。

---

## 模型表現與商業應用 (Model Performance & Results)

基於 30% 獨立測試集的最終表現（詳見 `Scorecard`）：

| 模型名稱 | Sensitivity (抓出風險的能力) | Specificity (不擾民的能力) | Kappa (真實預測力) | 商業應用場景建議 |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Bayes** | **94.74%** | - | - | **高風險厭惡型客戶 (如精密倉儲)**：寧可錯殺不可放過，提供極致的防潮預警防護網。 |
| **C5.0 (決策樹)** | 78.95% | **83.44%** | **0.1135** | **體驗優先型客戶 (如大眾 APP)**：在精準預警與避免「狼來了」之間取得最佳平衡。 |
| **物理代理基準** | 36.84% | 78.88% | 0.0240 | (僅作為 Baseline 參考) |

> **核心結論：** 引入 AI 混合特徵後的 C5.0 模型，其 Kappa 係數（扣除機率瞎猜後的真實預測力）達到了純物理模型的 **5 倍以上**，證明本系統確實從數據中萃取出了有效的氣象規律。

---

## 專案檔案結構 (Repository Structure)

* `process_weather_data_py.ipynb`：前端數據工程與 ETL 清洗腳本 (Python)。
* `HNT prediction - 202512082315 Final Run.R`：特徵工程、物理代理模型訓練與混合預測核心腳本 (R)。
* `Model_Evaluation_Scorecard.txt`：最終模型測試集表現之綜合計分卡 (原始輸出)。
* `sample_data.csv`：清洗後的結構化氣象數據範例 (為保護版權與節省空間，僅提供 100 行 Sample)。

## 技術棧 (Tech Stack)
* **Python:** Pandas, NumPy, Glob (Data Extraction & ETL)
* **R:** caret, C50, klaR, pROC (Feature Engineering, Machine Learning, Evaluation)
