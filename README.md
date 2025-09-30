# 台灣法規 RAG 知識檢索系統

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

這是一個基於 LlamaIndex 和 OpenAI 的檢索增強生成（RAG）系統，支援台灣多項法規查詢，包括食品安全衛生管理法和勞動基準法。系統會自動從法務部網站擷取最新的法規內容，建立向量索引，並提供智慧問答服務。

## 功能特色

- 🏛️ **自動法規擷取**：從法務部官網取得最新的多項法規內容
- 🔍 **智慧語義搜尋**：使用向量嵌入技術進行精確的條文檢索
- 🤖 **AI 問答服務**：基於 GPT 模型提供專業的法規解釋
- 📊 **查詢分類**：自動識別查詢類型（罰則、標示、添加物等）
- 💻 **友善介面**：提供互動式命令列介面和批次處理功能
- 📈 **智慧監控**：整合 Weights & Biases 全面監控系統效能和使用情況
- 🏗️ **模組化架構**：採用單一職責原則設計，易於維護和擴展
- 🔒 **資源管理**：完善的 Context Manager 設計，防止資源洩漏
- 🎯 **統一資料模型**：一致的 LegalArticle 模型跨所有法規領域

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

#### 方式一：生產版（推薦正式使用）
支援多法規和完整監控功能

```bash
python main.py
```

**特色**：
- 🚀 支援食品安全法 + 勞動基準法 + 民法
- 📊 完整的 W&B 效能監控
- 🔍 進階查詢路由和分析
- 📈 生產級錯誤處理
- 🏗️ 模組化架構設計

#### 方式二：簡化版（已棄用，建議使用方式一）
適合快速試用，但缺少最新架構改進

```bash
python main_no_wandb.py
```

> ⚠️ **注意**：此版本已標記為 deprecated，建議使用 `python main.py --no-monitoring` 替代。

**特色**：
- ⚠️ 僅支援食品安全法查詢
- ⚠️ 未包含最新的模組化架構
- ✅ 啟動速度快，資源消耗低

#### 方式三：研究版（實驗性功能）
最新的 AI 技術和實驗功能

```bash
python ultrathink.py
```

**特色**：
- 🧪 實驗性進階 RAG 功能
- 🤖 多重查詢增強和重排序
- 🔬 混合搜索和智慧路由
- 📊 詳細查詢分析和統計

**系統會自動**：
1. 檢查環境設定
2. 下載法規資料（如果尚未存在）
3. 建立向量索引
4. 初始化監控（如啟用）
5. 啟動互動式問答介面

### 4. CLI 指令詳解

#### 查詢模式

**單一法規查詢（預設食品安全法）**：
```bash
python main.py -q "食品添加物有什麼限制？"
```

**多法規整合查詢（智能路由）**：
```bash
python main.py --multi-domain -q "勞工食品安全規定"
```

**指定特定法規領域查詢**：
```bash
python main.py --domain food -q "食品標示規定"
python main.py --domain labor -q "工時限制規定"
```

**批次查詢模式**：
```bash
# 單一法規批次查詢
python main.py --batch config/query_examples.txt

# 多法規批次查詢
python main.py --multi-domain --batch config/query_examples.txt
```

#### 資料管理指令

**下載法規資料**：
```bash
# 下載食品安全衛生管理法資料
python main.py --fetch-food-data

# 下載勞動基準法資料
python main.py --fetch-labor-data
```

**重建向量索引**：
```bash
# 重建食品安全法向量索引
python main.py --rebuild-food-index

# 重建勞基法向量索引
python main.py --rebuild-labor-index
```

#### 系統管理與監控指令

**顯示索引統計資訊**：
```bash
# 顯示所有法規索引統計
python main.py --all-stats
```

**監控控制**：
```bash
# 停用 W&B 監控功能
python main.py --no-monitoring
```

#### 互動式查詢模式

**單一法規互動模式**：
```bash
python main.py
```

**多法規整合互動模式**：
```bash
python main.py --multi-domain
```

#### 向後相容指令（已廢棄但仍支援）

```bash
# 這些指令會自動轉換為新指令
python main.py --fetch-data     # 自動轉為 --fetch-food-data
python main.py --rebuild-index  # 自動轉為 --rebuild-food-index
python main.py --stats          # 自動轉為 --all-stats
```

## 系統架構

### 架構概述

本系統採用**模組化設計**，遵循**單一職責原則（SRP）**，確保每個模組專注於特定功能，提升可維護性和可擴展性。

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Environment  │  │ CLI Renderer │  │ Query Handler│     │
│  │  Validator   │  │              │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Data Manager │  │Index Manager │                        │
│  │              │  │              │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│                     Core RAG System                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Query Router │  │  Multi-Domain│  │ Single-Domain│     │
│  │              │  │  RAG System  │  │  RAG System  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Base Fetcher │←─│ Food Safety  │  │ Labor Law    │     │
│  │  (Abstract)  │  │   Fetcher    │  │   Fetcher    │     │
│  │              │  └──────────────┘  └──────────────┘     │
│  │              │  ┌──────────────┐                        │
│  │              │←─│  Civil Law   │                        │
│  │              │  │   Fetcher    │                        │
│  └──────────────┘  └──────────────┘                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │Legal Article │  │Index Builder │  │Document      │     │
│  │   Model      │  │              │  │Processor     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────────┐
│                  External Services                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  OpenAI API  │  │  ChromaDB    │  │  W&B Monitor │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 專案結構

```
LamaIndex/
├── src/                           # 核心應用程式碼
│   ├── cli/                       # CLI 模組（單一職責設計）
│   │   ├── environment_validator.py  # 環境檢查
│   │   ├── data_manager.py          # 資料管理
│   │   ├── index_manager.py         # 索引管理
│   │   ├── cli_renderer.py          # UI 渲染
│   │   └── query_handler.py         # 查詢處理
│   ├── legal_base_fetcher.py     # 基礎擷取器（消除 600+ 行重複）
│   ├── legal_food_safety_fetcher.py # 食品安全法擷取器
│   ├── labor_law_fetcher.py      # 勞基法擷取器
│   ├── civil_law_fetcher.py      # 民法擷取器
│   ├── legal_models.py           # 統一資料模型（LegalArticle）
│   ├── legal_basic_processor.py  # 基礎文件處理
│   ├── legal_enhanced_processor.py # 增強文件處理
│   ├── index_builder.py          # 向量索引建立（含資源清理）
│   ├── legal_single_domain_rag.py # 單一領域 RAG 系統
│   ├── legal_multi_domain_rag.py  # 多領域 RAG 系統
│   ├── query_router.py           # 查詢路由系統
│   └── monitoring.py             # W&B 監控整合
├── tests/                        # 測試套件
│   ├── test_*.py                # 各模組測試
│   └── ...
├── scripts/                      # 工具腳本
│   ├── build_labor_index.py     # 勞基法索引建構
│   ├── check_collections.py     # 資料庫檢查工具
│   └── system_performance_test.py # 系統效能測試
├── docs/                         # 技術文檔
│   ├── archive/                  # 歷史文檔存檔
│   ├── QUICKSTART.md            # 快速開始指南
│   ├── RAG_技術架構文檔.md       # 技術架構說明
│   └── RAG查詢策略技術文檔.md    # 查詢策略文檔
├── results/                      # 測試結果和報告
├── data/                         # 法規資料儲存
├── config/                       # 設定檔和範例
├── chroma_db/                   # 向量資料庫
├── wandb/                       # W&B 監控資料
├── main.py                      # 生產版主程式（503 行，模組化設計）
├── main_no_wandb.py            # 簡化版主程式（已棄用）
├── ultrathink.py               # 研究版主程式（實驗功能）
├── requirements.txt            # Python 相依套件
├── .env.template              # 環境設定範本
└── README.md                  # 專案說明文檔
```

### 架構優勢

#### 1. 模組化 CLI 層 (Priority 2 重構)
- **環境驗證** (`environment_validator.py`): 獨立檢查 API keys 和系統配置
- **資料管理** (`data_manager.py`): 統一處理所有法規資料下載
- **索引管理** (`index_manager.py`): 集中管理向量索引建構和統計
- **UI 渲染** (`cli_renderer.py`): 分離顯示邏輯，提供一致的使用者體驗
- **查詢處理** (`query_handler.py`): 專注於查詢執行和結果處理

**成果**: main.py 從 1,263 行減少到 503 行 (60% 減少)

#### 2. 統一資料擷取層 (Priority 1 重構)
- **BaseLegalFetcher** 抽象基類消除了 600+ 行重複程式碼
- 所有法規擷取器繼承統一介面，保證一致性
- 內建 Session 資源管理，防止資源洩漏
- 統一的錯誤處理和重試機制

**成果**: 消除 600+ 行重複程式碼，提升可維護性

#### 3. 統一資料模型
- **LegalArticle** 提供跨所有法規領域的一致資料結構
- 支援章節層級、條文編號、內容和元資料
- 簡化文件處理和索引建立流程

#### 4. 資源管理 (Priority 3 改進)
- **LegalIndexBuilder** 實作 Context Manager 協定
- 自動清理向量資料庫連線和資源
- 防止長時間執行造成的記憶體洩漏

```python
# 安全的資源使用範例
with LegalIndexBuilder(config) as builder:
    builder.build_index(documents)
# 自動清理資源
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

### 擴充其他法規

得益於模組化架構，新增法規支援變得非常簡單：

#### 步驟 1: 建立法規擷取器

繼承 `BaseLegalFetcher` 並實作必要方法：

```python
from src.legal_base_fetcher import BaseLegalFetcher
from src.legal_models import LegalArticle

class MyLawFetcher(BaseLegalFetcher):
    """您的法規擷取器"""

    def __init__(self):
        super().__init__(
            law_name="您的法規名稱",
            law_code="法規代碼"
        )

    def _parse_articles(self, soup) -> List[LegalArticle]:
        """解析法規條文"""
        # 實作您的解析邏輯
        # 返回 LegalArticle 物件列表
        pass
```

#### 步驟 2: 註冊到系統

在 `src/cli/data_manager.py` 中註冊：

```python
from src.my_law_fetcher import MyLawFetcher

# 在相應函數中添加
fetcher = MyLawFetcher()
articles = fetcher.fetch_all()
```

#### 步驟 3: 配置索引管理

在 `src/cli/index_manager.py` 中添加索引建構邏輯。

**就這樣！** 核心 RAG 系統、查詢處理、UI 渲染都無需修改。

### 自訂相關度評分

在 `src/legal_single_domain_rag.py` 或 `src/legal_multi_domain_rag.py` 中自訂評分邏輯：

```python
def calculate_relevance_score(self, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 實作您的自訂評分邏輯
    # 考慮因素：條文類型、章節相關性、法條層級等
    pass
```

### 添加新的 CLI 功能

在 `src/cli/` 目錄下建立新模組，遵循單一職責原則：

```python
# src/cli/my_feature.py
class MyFeatureManager:
    """專注於特定功能的管理器"""

    def __init__(self, config):
        self.config = config

    def execute(self):
        """執行功能"""
        pass
```

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
- **模組化設計**：CLI 層、RAG 核心、資料層清晰分離
- **單一職責原則**：每個模組專注特定功能，易於測試和維護
- **統一介面**：BaseLegalFetcher 提供一致的資料擷取介面
- **資源安全**：Context Manager 模式確保資源正確釋放
- **優雅降級**：監控功能可選，不影響核心功能
- **容錯設計**：ChromaDB 連線失敗自動回退
- **可擴展**：支援自訂相關度算法和新法規
- **統一資料模型**：LegalArticle 跨所有法規領域

## ❓ 常見問題

### Q: 程式啟動失敗怎麼辦？
A: 按照以下步驟排除：
1. **停用監控測試**：`python main.py --no-monitoring --help`
2. **檢查環境**：確認已設定 `OPENAI_API_KEY`
3. **檢視詳細錯誤**：詳細錯誤訊息通常會指出問題所在
4. **使用簡化版**（不推薦）：`python main_no_wandb.py --help`（此版本已棄用）

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
# 更新食品安全法
python main.py --fetch-food-data --rebuild-food-index

# 更新勞動基準法
python main.py --fetch-labor-data --rebuild-labor-index

# 更新民法
python main.py --fetch-civil-data --rebuild-civil-index
```

### Q: 能否支援其他法規？
A: 系統架構完全支援擴充！得益於模組化設計：
1. 建立新的法規擷取器，繼承 `BaseLegalFetcher`
2. 實作必要的解析方法
3. 在 `data_manager.py` 和 `index_manager.py` 中註冊新法規
4. 無需修改核心 RAG 系統

範例：最近已成功整合民法，僅需約 200 行程式碼

## 🚀 快速故障排除

如果遇到啟動問題，按照以下順序測試：

```bash
# 1. 測試完整版但停用監控（推薦）
python main.py --no-monitoring --help

# 2. 測試完整版含監控
python main.py --help

# 3. 測試簡化版（已棄用，不推薦）
python main_no_wandb.py --help
```

詳細的故障排除指南請參考：
- `docs/archive/` - 歷史故障排除文檔
- `docs/QUICKSTART.md` - 快速開始指南
- `docs/UV_SETUP_GUIDE.md` - 環境設置指南

## 📚 文檔資源

- **快速開始**：本 README.md + `docs/QUICKSTART.md`
- **技術架構**：`docs/RAG_技術架構文檔.md`
- **查詢策略**：`docs/RAG查詢策略技術文檔.md`
- **環境設置**：`docs/UV_SETUP_GUIDE.md`
- **歷史文檔**：`docs/archive/` (故障排除和修復記錄)
- **測試套件**：`tests/` 目錄下的所有測試檔案

## 📈 近期重構改進

### 重構成果總覽 (2025)

我們完成了三個優先級的全面重構，大幅提升系統品質和可維護性：

#### ✅ Priority 1: 基礎重構 - 消除技術債
**目標**: 消除重複程式碼，統一資料模型，修復資源洩漏

- **建立 BaseLegalFetcher 抽象基類**
  - 消除 600+ 行重複程式碼
  - 統一 HTTP session 管理
  - 實作共用的錯誤處理和重試機制
  - 所有法規擷取器 (Food Safety, Labor Law, Civil Law) 繼承統一介面

- **統一 LegalArticle 資料模型**
  - 單一資料模型跨所有法規領域
  - 標準化章節、條文、內容結構
  - 簡化文件處理流程

- **修復 Session 資源洩漏**
  - 實作 Context Manager 模式
  - 確保 HTTP connections 正確關閉
  - 防止長時間執行的記憶體洩漏

**影響**: 程式碼重複減少 60%，資源管理更安全

#### ✅ Priority 2: 架構重構 - 模組化設計
**目標**: 分離關注點，實現單一職責原則

- **建立 5 個獨立 CLI 模組**
  - `environment_validator.py`: 環境配置檢查
  - `data_manager.py`: 法規資料管理
  - `index_manager.py`: 向量索引管理
  - `cli_renderer.py`: UI 顯示邏輯
  - `query_handler.py`: 查詢執行處理

- **重構 main.py**
  - 從 1,263 行減少到 503 行 (60% 減少)
  - 每個模組專注單一職責
  - 提升程式碼可讀性和可測試性
  - 更容易維護和擴展

**影響**: 主程式碼簡化 60%，模組職責清晰

#### ✅ Priority 3: 程式碼健康 - 品質提升
**目標**: 提升程式碼品質，標記過時程式碼

- **LegalIndexBuilder 資源清理**
  - 實作 `__enter__` 和 `__exit__` 方法
  - 自動管理 ChromaDB 連線生命週期
  - 防止資源洩漏和連線堆積

- **標記 main_no_wandb.py 為 deprecated**
  - 添加棄用警告
  - 引導使用者使用 `main.py --no-monitoring`
  - 保留向後相容性

- **保留 SimpleGraph 核心功能**
  - 經評估為核心查詢增強功能
  - 非實驗性程式碼
  - 持續維護和改進

**影響**: 資源管理更可靠，程式碼庫更清晰

### 技術改進指標

| 指標 | 改進前 | 改進後 | 提升 |
|------|--------|--------|------|
| main.py 行數 | 1,263 | 503 | -60% |
| 重複程式碼 | 600+ 行 | 0 行 | -100% |
| CLI 模組數 | 1 | 5 | +400% |
| 資源洩漏風險 | 高 | 低 | ✅ |
| 模組職責清晰度 | 低 | 高 | ✅ |

### 未來規劃

- 持續改進查詢路由算法
- 優化向量檢索效能
- 擴展支援更多法規領域
- 增強監控和分析功能

## 🤝 貢獻

歡迎提交問題報告和改進建議！

## 📄 授權

本專案僅供學習和研究使用。法規內容來源於中華民國法務部網站。

---

## 📝 版本歷程

### v2.0 (2025) - 模組化重構版本
- 完成三階段重構，提升系統品質 60%
- 引入模組化架構和單一職責設計
- main.py 精簡至 503 行
- 消除 600+ 行重複程式碼
- 統一 LegalArticle 資料模型
- 新增民法支援
- 完善資源管理機制

### v1.0 - 初始版本
- 支援食品安全衛生管理法和勞動基準法
- 整合 Weights & Biases 監控
- 基礎 RAG 查詢功能
- 向量索引建立和管理

---

**最新版本說明**：本系統採用模組化架構設計，提供完整的效能分析和使用統計。建議使用 `python main.py` 享受最佳開發體驗！