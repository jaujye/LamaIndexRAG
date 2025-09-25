# RAG 技術架構文檔
# 台灣食品安全衛生管理法 RAG 知識檢索系統

## 目錄
- [系統架構概覽](#系統架構概覽)
- [核心技術棧](#核心技術棧)
- [RAG 查詢流程](#rag-查詢流程)
- [關鍵類別與方法](#關鍵類別與方法)
- [技術特色與優化](#技術特色與優化)
- [配置與擴展](#配置與擴展)
- [性能考量](#性能考量)

---

## 系統架構概覽

本專案採用 **檢索增強生成 (RAG)** 架構，結合向量檢索與大型語言模型，為台灣食品安全衛生管理法提供智慧問答服務。

### 核心設計理念
- **法律專業性**: 專門針對台灣法規文件優化的 RAG 系統
- **精確性**: 基於原始法條內容生成回答，避免幻覺
- **可擴展性**: 模組化設計，支援多種法規文件
- **多模式查詢**: 支援互動式、批次處理等多種查詢模式

### 系統架構圖
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   使用者查詢     │───▶│   RAG 查詢引擎    │───▶│   生成式回答      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   向量檢索系統     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   ChromaDB       │
                    │   向量資料庫      │
                    └──────────────────┘
```

---

## 核心技術棧

### 1. 主要框架與函式庫
- **LlamaIndex**: RAG 框架核心，處理文件索引與查詢
- **ChromaDB**: 向量資料庫，支援本地與遠端部署
- **OpenAI**: 嵌入模型 (text-embedding-3-small) 與語言模型 (GPT-3.5-turbo)
- **BeautifulSoup**: 法規網頁解析
- **Rich**: 命令列介面美化

### 2. 關鍵 LlamaIndex 組件
```python
# 核心檢索組件
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts.base import PromptTemplate
```

### 3. 資料流架構
```
法規網頁 → 文件解析 → 文件分塊 → 向量嵌入 → 向量資料庫
                                                    ↓
用戶查詢 → 查詢分類 → 查詢增強 → 向量檢索 → 後處理 → LLM 生成
```

---

## RAG 查詢流程

### 1. 系統初始化流程
```python
class LegalRAGSystem:
    def __init__(self):
        # 1. 載入環境變數和 API 金鑰
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        # 2. 配置 ChromaDB 連接
        self.chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")

        # 3. 初始化索引建構器
        self.index_builder = LegalIndexBuilder(...)

        # 4. 載入現有向量索引
        self._load_index()
```

### 2. 查詢處理流程

#### 步驟 1: 查詢分類
```python
def classify_query_type(self, query: str) -> str:
    """
    基於關鍵字匹配對查詢進行分類：
    - penalty: 罰則相關
    - labeling: 標示相關
    - additives: 添加物相關
    - hygiene: 衛生相關
    - inspection: 檢驗相關
    - import: 進口相關
    - manufacturing: 製造相關
    - general: 一般查詢
    """
```

#### 步驟 2: 查詢增強
```python
def enhance_query(self, query: str, query_type: str) -> str:
    """
    根據查詢類型添加相關關鍵字，提升檢索精確度
    例如：'penalty' 類型會加上 "相關的罰則和處罰規定"
    """
```

#### 步驟 3: 向量檢索
```python
def setup_query_engine(self,
                      similarity_top_k: int = 5,
                      similarity_cutoff: float = 0.7):
    """
    配置查詢引擎參數：
    - similarity_top_k: 檢索文件數量
    - similarity_cutoff: 相似度閾值
    """
```

#### 步驟 4: 相關度評分
```python
def calculate_relevance_score(self, query: str, sources: List[Dict]) -> List[Dict]:
    """
    自定義相關度評分邏輯 (待實現):
    - 向量相似度權重
    - 條文類型權重 (主條文 vs 罰則)
    - 章節相關性
    - 法條層級重要性
    """
```

#### 步驟 5: 回應生成
使用自定義提示模板生成專業的法律回答。

---

## 關鍵類別與方法

### 1. LegalRAGSystem 類別

#### 核心屬性
```python
class LegalRAGSystem:
    def __init__(self):
        self.api_key: str                          # OpenAI API 金鑰
        self.chroma_path: str                      # ChromaDB 路徑
        self.collection_name: str                  # 集合名稱
        self.temperature: float                    # LLM 溫度參數
        self.index: VectorStoreIndex              # 向量索引
        self.query_engine: RetrieverQueryEngine   # 查詢引擎
        self.index_builder: LegalIndexBuilder     # 索引建構器
```

#### 關鍵方法

**1. 索引管理**
```python
def _load_index(self) -> None:
    """載入現有向量索引，若不存在則拋出異常"""

def setup_query_engine(self, **params) -> RetrieverQueryEngine:
    """設置查詢引擎，配置檢索參數和後處理器"""
```

**2. 查詢處理**
```python
def query(self, question: str, **params) -> QueryResult:
    """核心查詢方法，整合完整的 RAG 流程"""

def batch_query(self, questions: List[str]) -> List[QueryResult]:
    """批次處理多個查詢"""
```

**3. 工具方法**
```python
def get_related_articles(self, article_number: str) -> List[Dict]:
    """取得與特定條文相關的其他條文"""

def explain_article(self, article_number: str) -> QueryResult:
    """解釋特定條文的內容"""

def search_by_keyword(self, keyword: str) -> List[Dict]:
    """基於關鍵字的文件搜尋"""
```

### 2. QueryResult 資料結構
```python
@dataclass
class QueryResult:
    question: str           # 原始問題
    answer: str            # 生成的回答
    sources: List[Dict]    # 來源文件資訊
    confidence_score: float # 信心度分數
    query_type: str        # 查詢類型
```

---

## 技術特色與優化

### 1. 自定義提示模板
```python
def _create_custom_prompt_template(self) -> PromptTemplate:
    """
    針對台灣法律文件優化的提示模板:
    - 強調使用繁體中文
    - 要求引用具體法條編號
    - 保持專業法律語調
    - 明確標示資訊不足情況
    """
```

### 2. 智慧查詢增強
- **語義增強**: 根據查詢類型添加相關法律術語
- **上下文擴展**: 自動補充相關法規領域關鍵字
- **專業術語對應**: 將口語化問題轉換為法律專業用語

### 3. 多層級相關度評分
- **基礎向量相似度**: OpenAI 嵌入模型計算的語義相似性
- **條文類型權重**: 主條文、罰則條文、例外規定的不同權重
- **章節相關性**: 基於法規章節結構的相關性加權
- **法條重要性**: 根據引用頻率和法律層級調整權重

### 4. 多模式查詢支援
- **互動式查詢**: 即時問答，支援追問
- **批次處理**: 大量問題的批次處理
- **條文解釋**: 針對特定條文的詳細解釋
- **關聯分析**: 條文間的關聯性分析

---

## 配置與擴展

### 1. 環境配置
```bash
# .env 檔案配置
OPENAI_API_KEY=your_api_key_here
CHROMA_DB_PATH=./chroma_db                    # 本地模式
CHROMA_DB_PATH=http://192.168.0.114:7000     # 遠端模式
```

### 2. 查詢參數調整
```python
# main.py 中的預設參數
self.rag_system.setup_query_engine(
    similarity_top_k=10,        # 檢索文件數量
    similarity_cutoff=0.3,      # 相似度閾值
    response_mode="compact"     # 回應合成模式
)
```

### 3. 系統擴展點

#### 增加新的法規文件
```python
# 1. 修改 data_fetcher.py 支援新的資料來源
# 2. 調整 document_processor.py 適應不同文件結構
# 3. 更新 rag_system.py 的查詢分類邏輯
```

#### 客製化相關度評分
```python
def calculate_relevance_score(self, query: str, sources: List[Dict]) -> List[Dict]:
    """
    實作自定義相關度評分:
    - 分析查詢意圖
    - 計算條文類型相關性
    - 考量章節結構權重
    - 整合多個評分因子
    """
```

#### 添加新的查詢類型
```python
def classify_query_type(self, query: str) -> str:
    """
    擴展查詢分類邏輯:
    - 新增查詢類型定義
    - 添加對應的關鍵字模式
    - 實現對應的查詢增強策略
    """
```

---

## 性能考量

### 1. 檢索性能優化
- **向量索引**: 使用 ChromaDB 的高效向量索引
- **相似度閾值**: 平衡檢索品質與回應速度 (預設 0.3)
- **檢索數量**: 控制 similarity_top_k 參數避免過度檢索

### 2. 記憶體管理
- **索引快取**: 向量索引載入後常駐記憶體
- **批次處理**: 大量查詢的記憶體效率優化
- **連接池**: ChromaDB 連接的高效管理

### 3. 錯誤處理與容錯
```python
# 遠端 ChromaDB 連接失敗時自動切換到本地模式
try:
    client = chromadb.HttpClient(host="192.168.0.114", port=7000)
except Exception:
    client = chromadb.PersistentClient(path="./chroma_db")
```

### 4. 系統監控
```python
def get_system_stats(self) -> Dict[str, Any]:
    """
    提供系統運行狀態監控:
    - RAG 系統狀態
    - 查詢引擎配置
    - 索引統計資訊
    - 性能指標
    """
```

---

## 未來發展方向

### 1. 技術增強
- **混合檢索**: 結合關鍵字檢索與向量檢索
- **重新排序**: 實作更精細的結果重新排序算法
- **快取系統**: 常見查詢的結果快取機制
- **多模態支援**: 支援圖表、表格等多模態法規內容

### 2. 功能擴展
- **法規比較**: 不同版本法規的對比分析
- **案例推理**: 整合法律案例的推理能力
- **多語言支援**: 支援英文等其他語言查詢
- **API 介面**: 提供 RESTful API 供外部系統整合

### 3. 效能優化
- **分散式部署**: 支援多節點分散式向量檢索
- **增量更新**: 法規更新的增量索引更新機制
- **GPU 加速**: 利用 GPU 加速向量計算
- **模型優化**: 使用更小更快的專用模型

---

*本文檔最後更新：2024年9月*