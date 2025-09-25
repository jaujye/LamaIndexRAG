# 監控參數問題修復總結

## 問題描述

執行 `uv run python main.py` 時出現錯誤：
```
載入索引失敗: 'LegalIndexBuilder' object has no attribute 'monitor'
```

## 根本原因

在整合 W&B 監控功能時，我修改了 `LegalIndexBuilder` 和 `LegalRAGSystem` 的建構子來接受監控相關參數，但沒有同時更新所有調用這些類別的地方。

## 修復內容

### 1. 修復 main.py 中遺漏的調用點

**位置**: main.py:513
```python
# 修復前
self.index_builder = LegalIndexBuilder()

# 修復後
self.index_builder = LegalIndexBuilder(
    enable_monitoring=self.enable_monitoring,
    monitor=self.monitor
)
```

### 2. 修復 test_system.py

**位置**: test_system.py:156
```python
# 修復前
builder = LegalIndexBuilder()

# 修復後
builder = LegalIndexBuilder(enable_monitoring=False)
```

### 3. 修復 src/index_builder.py 的測試代碼

**位置**: src/index_builder.py:553
```python
# 修復前
builder = LegalIndexBuilder()

# 修復後
builder = LegalIndexBuilder(enable_monitoring=False)
```

### 4. 修復 main_no_wandb.py

**位置**: main_no_wandb.py:105, 320
```python
# 修復前
self.index_builder = LegalIndexBuilder()

# 修復後
self.index_builder = LegalIndexBuilder(enable_monitoring=False)
```

## 修復後的類別參數

### LegalIndexBuilder
```python
def __init__(self,
             api_key: Optional[str] = None,
             chroma_path: str = None,
             collection_name: str = "food_safety_act",
             embedding_model: str = "text-embedding-3-small",
             enable_monitoring: bool = True,  # 新增
             monitor: Optional[WandbMonitor] = None):  # 新增
```

### LegalRAGSystem
```python
def __init__(self,
             api_key: Optional[str] = None,
             chroma_path: str = None,
             collection_name: str = "food_safety_act",
             temperature: float = 0.1,
             enable_monitoring: bool = True,  # 新增
             monitor: Optional[WandbMonitor] = None):  # 新增
```

## 驗證修復

執行測試腳本來驗證修復：
```bash
python test_fix.py
```

或直接測試主程式：
```bash
# 測試完整版
uv run python main.py --help

# 測試簡化版
uv run python main_no_wandb.py --help
```

## 修復的檔案清單

1. ✅ `main.py` - 主程式中的索引建構器調用
2. ✅ `test_system.py` - 系統測試中的調用
3. ✅ `src/index_builder.py` - 測試代碼中的調用
4. ✅ `main_no_wandb.py` - 簡化版中的兩個調用點

## 後向兼容性

修復後：
- 所有新的調用都明確指定監控參數
- 預設情況下監控是啟用的（`enable_monitoring=True`）
- 可以通過 `enable_monitoring=False` 停用監控
- 不會影響現有的核心功能

現在程式應該可以正常啟動了！