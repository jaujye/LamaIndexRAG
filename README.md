# 台灣食品安全衛生管理法 RAG 知識檢索系統

## 技術棧

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-FF6B35?style=for-the-badge&logo=llama&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logo=database&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Rich](https://img.shields.io/badge/Rich-000000?style=for-the-badge&logo=python&logoColor=white)

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

## 環境需求

- Python 3.8 或更高版本
- OpenAI API Key

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

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 執行系統

#### 互動式查詢模式（推薦）

```bash
python main.py
```

系統會自動：
1. 檢查環境設定
2. 下載法規資料（如果尚未存在）
3. 建立向量索引
4. 啟動互動式問答介面

#### 單一查詢模式

```bash
python main.py -q "食品添加物有什麼限制？"
```

#### 批次查詢模式

```bash
python main.py --batch config/query_examples.txt
```

### 4. 系統管理指令

重新下載法規資料：
```bash
python main.py --fetch-data
```

重建向量索引：
```bash
python main.py --rebuild-index
```

顯示索引統計：
```bash
python main.py --stats
```

## 系統架構

```
LamaIndex/
├── src/
│   ├── data_fetcher.py      # 法規資料擷取
│   ├── document_processor.py # 文件處理和分塊
│   ├── index_builder.py     # 向量索引建立
│   └── rag_system.py        # RAG 查詢系統
├── data/                    # 法規資料儲存
├── config/                  # 設定檔和範例
├── chroma_db/              # 向量資料庫
├── main.py                 # 主程式入口
├── requirements.txt        # Python 相依套件
└── .env.template          # 環境設定範本
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

## 技術細節

- **嵌入模型**：OpenAI text-embedding-3-small
- **語言模型**：GPT-3.5-turbo
- **向量資料庫**：ChromaDB
- **文件分塊**：保留法條結構的智慧分塊
- **檢索策略**：向量相似度 + 自訂相關度評分

## 常見問題

### Q: 系統回答不準確怎麼辦？
A: 可以嘗試：
1. 調整查詢的描述方式
2. 使用更具體的法律術語
3. 在 `rag_system.py` 中調整 `similarity_cutoff` 參數

### Q: 如何更新法規內容？
A: 執行 `python main.py --fetch-data --rebuild-index` 重新下載並建立索引

### Q: 能否支援其他法規？
A: 系統架構支援擴充，需要修改對應的資料擷取和處理模組

## 貢獻

歡迎提交問題報告和改進建議！

## 授權

本專案僅供學習和研究使用。法規內容來源於中華民國法務部網站。