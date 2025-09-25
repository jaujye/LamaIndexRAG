# Monitor 屬性問題最終修復

## 問題根本原因

錯誤 `'LegalIndexBuilder' object has no attribute 'monitor'` 的真正原因是：

1. **初始化順序問題**: `monitor` 屬性在被 `@monitor_execution_time` 裝飾器嘗試存取**之前**還沒有初始化
2. **裝飾器執行時機**: `@monitor_execution_time` 裝飾的方法在物件完全初始化前就被調用
3. **ChromaDB 連線失敗**: 遠程 ChromaDB 連線問題導致初始化過程中斷

## 最終修復方案

### 1. 調整初始化順序

**在 `src/index_builder.py` 中**:
```python
# 修復前: monitor 屬性在最後初始化
def __init__(self, ...):
    # ... 其他初始化
    self._setup_chroma_client()  # 這裡有 @monitor_execution_time 裝飾器
    # ... 其他初始化
    self.monitor = monitor  # 太晚了！

# 修復後: monitor 屬性最先初始化
def __init__(self, ...):
    # Initialize monitoring FIRST
    self.enable_monitoring = enable_monitoring
    self.monitor = monitor
    if self.enable_monitoring and not self.monitor:
        self.monitor = WandbMonitor(mode="disabled")

    # 然後才初始化其他組件
    self._setup_chroma_client()  # 現在可以安全使用裝飾器
```

### 2. 修復 RAG 系統的類似問題

**在 `src/rag_system.py` 中**:
```python
# 修復: 確保監控初始化在前，並正確傳遞給 LegalIndexBuilder
def __init__(self, ...):
    # Initialize monitoring FIRST
    self.enable_monitoring = enable_monitoring
    self.monitor = monitor
    if self.enable_monitoring and not self.monitor:
        self.monitor = WandbMonitor(mode="disabled")

    # 然後初始化 LegalIndexBuilder 並傳遞監控參數
    self.index_builder = LegalIndexBuilder(
        api_key=self.api_key,
        chroma_path=chroma_path,
        collection_name=collection_name,
        enable_monitoring=self.enable_monitoring,  # 新增
        monitor=self.monitor  # 新增
    )
```

### 3. 改進 ChromaDB 連線容錯性

**在 `src/index_builder.py` 的 `_setup_chroma_client` 方法中**:
```python
# 新增第三種連線方法
# Approach 3: Simple HttpClient (most basic)
if not client_created:
    try:
        self.chroma_client = chromadb.HttpClient(
            host=host,
            port=port
        )
        client_created = True
        print("[OK] Connected with basic HttpClient")
    except Exception as basic_error:
        print(f"[ERROR] Basic HttpClient failed: {basic_error}")
        raise basic_error
```

## 修復的檔案清單

1. ✅ `src/index_builder.py` - 調整監控初始化順序，改進 ChromaDB 連線
2. ✅ `src/rag_system.py` - 調整監控初始化順序，修復 LegalIndexBuilder 調用
3. ✅ 所有其他檔案中的 `LegalIndexBuilder()` 調用已在之前修復

## 測試步驟

### 1. 執行修復測試
```bash
python test_monitor_fix.py
```

### 2. 測試實際程式啟動
```bash
# 測試簡化版（推薦用於測試）
uv run python main_no_wandb.py --help

# 測試完整版（停用監控）
uv run python main.py --no-monitoring --help

# 測試完整版（啟用監控，需要 wandb）
uv run python main.py --help
```

## 解決的問題

1. ✅ **Monitor 屬性未初始化**: 現在在任何使用裝飾器的方法調用前就初始化
2. ✅ **裝飾器相容性**: `@monitor_execution_time` 裝飾器現在可以安全使用
3. ✅ **ChromaDB 連線容錯**: 多種連線方法確保更好的相容性
4. ✅ **參數傳遞一致性**: 所有 LegalIndexBuilder 和 LegalRAGSystem 調用都正確傳遞監控參數

## 期望結果

修復後，程式應該能夠：
- 🟢 正常初始化所有組件
- 🟢 處理 ChromaDB 連線問題（如果有的話，會回退到本地模式）
- 🟢 正確處理監控功能（啟用或停用都可以）
- 🟢 顯示幫助資訊和開始互動式介面

現在應該不再出現 `'LegalIndexBuilder' object has no attribute 'monitor'` 錯誤了！