# 台灣食品安全衛生管理法 RAG 知識檢索系統

## 技術棧

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-FF6B35?style=for-the-badge&logo=llama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logo=database&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Rich](https://img.shields.io/badge/Rich-000000?style=for-the-badge&logo=python&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)

</div>

---

## 系統簡介

這是一個基於 LlamaIndex 和 OpenAI 的檢索增強生成（RAG）系統，專門用於查詢台灣食品安全衛生管理法的相關規定。系統會自動從法務部網站擷取最新的法規內容，建立向量索引，並提供智慧問答服務。

## 功能特色

- 🏛️ **自動法規擷取**：從法務部官網取得最新的食品安全衛生管理法
- 🔍 **智慧語義搜尋**：使用向量嵌入技術進行精確的條文檢索
- 🤖 **AI 問答服務**：基於 GPT 模型提供專業的法規解釋
- 📊 **查詢分類**：自動識別查詢類型（罰則、標示、添加物等）
- 💻 **友善介面**：提供互動式命令列介面和批次處理功能
- 📈 **智慧監控**：整合 Weights & Biases 全面監控系統效能和使用情況

## 環境需求

- Python 3.8 或更高版本
- OpenAI API Key
- （可選）Weights & Biases 帳號用於監控功能

## 快速開始

### 1. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數

複製環境設定檔案並設定您的 OpenAI API Key：

```bash
cp .env.template .env
```

編輯 `.env` 檔案，設定您的 API Key：

```env
OPENAI_API_KEY=your_openai_api_key_here

# W&B 監控（可選）
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=food-safety-rag
WANDB_MODE=online  # online, offline, disabled
```

### 3. 選擇執行模式

系統提供三種執行方式，依據您的需求選擇：

#### 方式一：簡化版（推薦新手）

```bash
# 使用簡化版，不含監控功能
python main_no_wandb.py
```

**優點**：
- ✅ 無需額外安裝，立即可用
- ✅ 包含完整 RAG 功能
- ✅ 啟動速度快

#### 方式二：完整版（不含監控）

```bash
# 使用完整版，停用監控功能
python main.py --no-monitoring
```

**優點**：
- ✅ 完整功能，未來可啟用監控
- ✅ 更好的錯誤處理
- ✅ 支援所有進階參數

#### 方式三：完整版（含監控）

```bash
# 首先安裝 W&B
pip install wandb>=0.15.0

# 使用完整版，含監控功能
python main.py
```

**優點**：
- 🚀 完整的效能監控和分析
- 📊 查詢統計和使用趨勢
- 🔍 錯誤追蹤和系統健康監控
- 📈 視覺化儀表板

**系統會自動**：
1. 檢查環境設定
2. 下載法規資料（如果尚未存在）
3. 建立向量索引
4. 初始化監控（如啟用）
5. 啟動互動式問答介面

### 4. 查詢模式

#### 單一查詢模式

```bash
# 簡化版
python main_no_wandb.py -q "食品添加物有什麼限制？"

# 完整版
python main.py -q "食品添加物有什麼限制？"
```

#### 批次查詢模式

```bash
# 簡化版
python main_no_wandb.py --batch config/query_examples.txt

# 完整版
python main.py --batch config/query_examples.txt
```

### 5. 系統管理指令

重新下載法規資料：
```bash
# 任選一種方式
python main_no_wandb.py --fetch-data
python main.py --fetch-data
```

重建向量索引：
```bash
# 任選一種方式
python main_no_wandb.py --rebuild-index
python main.py --rebuild-index
```

顯示索引統計：
```bash
# 任選一種方式
python main_no_wandb.py --stats
python main.py --stats
```

停用監控功能（完整版）：
```bash
python main.py --no-monitoring
```

## 系統架構

```
LamaIndex/
├── src/
│   ├── data_fetcher.py      # 法規資料擷取
│   ├── document_processor.py # 文件處理和分塊
│   ├── index_builder.py     # 向量索引建立
│   ├── rag_system.py        # RAG 查詢系統
│   └── monitoring.py        # W&B 監控整合
├── data/                    # 法規資料儲存
├── config/                  # 設定檔和範例
├── chroma_db/              # 向量資料庫
├── main.py                 # 主程式入口（含監控）
├── main_no_wandb.py        # 簡化版主程式
├── requirements.txt        # Python 相依套件
├── .env.template          # 環境設定範本
└── doc/                    # 技術文檔和指南
```

## 使用範例

### 基本查詢範例

```
問題：食品添加物有什麼使用限制？

回答：根據食品安全衛生管理法第18條規定，食品添加物之使用應符合下列規定：
一、使用範圍及限量標準，應符合中央主管機關之規定。
二、規格應符合中央主管機關所定食品添加物使用範圍及限量暨規格標準之規定。
...

相關法條：
- 第18條 (相似度: 0.892)
- 第19條 (相似度: 0.764)
- 第21條 (相似度: 0.723)
```

### 進階功能

系統支援多種查詢類型的自動識別和優化：

- **罰則查詢**：自動強化對處罰條文的搜尋
- **標示查詢**：專注於標籤和包裝相關規定
- **添加物查詢**：針對食品添加物的特定規範
- **衛生查詢**：關注衛生安全標準和要求

## 開發和自訂

### 自訂相關度評分

在 `src/rag_system.py` 中的 `calculate_relevance_score()` 函數，您可以實作自己的相關度評分邏輯：

```python
def calculate_relevance_score(self, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 實作您的自訂評分邏輯
    # 考慮因素：條文類型、章節相關性、法條層級等
    pass
```

### 擴充其他法規

系統架構支援擴充其他法規文件，只需：
1. 修改 `data_fetcher.py` 以支援新的資料來源
2. 調整 `document_processor.py` 以適應不同的文件結構
3. 更新 `rag_system.py` 的查詢分類邏輯

## 🎯 監控功能詳解

### W&B 監控指標

**查詢效能**：
- 端到端查詢延遲
- 檢索時間和 LLM 回應時間
- 記憶體使用監控
- Token 使用量統計

**系統品質**：
- 檢索精度（相似度分數分布）
- 回答信心度追蹤
- 查詢類型分類準確性
- 錯誤率和異常監控

**使用分析**：
- 查詢頻率和模式
- 使用者行為分析
- 系統健康度儀表板
- 效能趨勢分析

### 監控設置

1. **註冊 W&B 帳號**：前往 [wandb.ai](https://wandb.ai) 註冊
2. **獲取 API Key**：在 [Settings](https://wandb.ai/settings) 中複製 API key
3. **設定環境變數**：在 `.env` 中設定 `WANDB_API_KEY`
4. **啟動監控**：使用 `python main.py`（不加 `--no-monitoring`）

## 🔧 技術細節

### 核心技術棧
- **嵌入模型**：OpenAI text-embedding-3-small
- **語言模型**：GPT-3.5-turbo
- **向量資料庫**：ChromaDB（支援本地/遠程）
- **監控平台**：Weights & Biases
- **文件分塊**：保留法條結構的智慧分塊
- **檢索策略**：向量相似度 + 自訂相關度評分

### 架構特色
- **優雅降級**：監控功能可選，不影響核心功能
- **容錯設計**：ChromaDB 連線失敗自動回退
- **模組化**：監控、索引、查詢各自獨立
- **可擴展**：支援自訂相關度算法和新法規

## ❓ 常見問題

### Q: 程式啟動失敗怎麼辦？
A: 按照以下步驟排除：
1. **使用簡化版**：`python main_no_wandb.py --help`
2. **檢查環境**：確認已設定 `OPENAI_API_KEY`
3. **停用監控**：`python main.py --no-monitoring`
4. **查看錯誤**：詳細錯誤訊息通常會指出問題所在

### Q: W&B 監控相關問題
A: 監控功能疑難排解：
- **未安裝 wandb**：`pip install wandb>=0.15.0`
- **API key 錯誤**：在 `.env` 中設定正確的 `WANDB_API_KEY`
- **網路問題**：設定 `WANDB_MODE=offline` 或 `WANDB_MODE=disabled`
- **不需要監控**：使用 `--no-monitoring` 參數或簡化版

### Q: ChromaDB 連線問題
A: 常見解決方案：
- **遠程連線失敗**：系統會自動回退到本地模式
- **權限問題**：確保有 `./chroma_db/` 目錄的寫入權限
- **版本衝突**：更新到最新版本：`pip install chromadb>=0.4.24`

### Q: 系統回答不準確怎麼辦？
A: 可以嘗試：
1. 調整查詢的描述方式
2. 使用更具體的法律術語
3. 在 `rag_system.py` 中調整 `similarity_cutoff` 參數
4. 使用監控功能分析查詢模式

### Q: 如何更新法規內容？
A: 重新下載並建立索引：
```bash
# 任選一種
python main_no_wandb.py --fetch-data --rebuild-index
python main.py --fetch-data --rebuild-index
```

### Q: 能否支援其他法規？
A: 系統架構支援擴充，需要修改對應的資料擷取和處理模組

## 🚀 快速故障排除

如果遇到啟動問題，按照以下順序測試：

```bash
# 1. 測試簡化版（最可靠）
python main_no_wandb.py --help

# 2. 測試完整版但停用監控
python main.py --no-monitoring --help

# 3. 測試完整版含監控
python main.py --help
```

詳細的故障排除指南請參考：
- `QUICK_FIX.md` - 快速修復指南
- `MONITOR_FIX_FINAL.md` - 監控問題修復
- `WANDB_MONITORING_GUIDE.md` - W&B 監控完整指南

## 📚 文檔資源

- **快速開始**：本 README.md
- **W&B 監控指南**：`WANDB_MONITORING_GUIDE.md`
- **技術架構文檔**：`doc/RAG_技術架構文檔.md`
- **故障排除**：`QUICK_FIX.md`、`MONITOR_FIX_FINAL.md`
- **測試腳本**：`test_startup.py`、`test_monitor_fix.py`

## 🤝 貢獻

歡迎提交問題報告和改進建議！

## 📄 授權

本專案僅供學習和研究使用。法規內容來源於中華民國法務部網站。

---

**版本說明**：本系統已整合 Weights & Biases 監控功能，提供完整的效能分析和使用統計。選擇適合您需求的執行方式開始使用！