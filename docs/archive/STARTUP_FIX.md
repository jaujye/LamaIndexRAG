# 啟動問題修復

## 問題描述

程式啟動時出現錯誤：
```
[FAIL] 程式錯誤: FoodSafetyRAGCLI.__init__() got an unexpected keyword argument 'enable_monitoring'
```

## 問題原因

在整合 W&B 監控功能時，沒有正確更新 `FoodSafetyRAGCLI` 類別的 `__init__` 方法參數定義。

## 修復內容

### 1. 更新 `__init__` 方法參數

**修復前:**
```python
def __init__(self):
    self.console = Console()
    # ...
```

**修復後:**
```python
def __init__(self, enable_monitoring: bool = True):
    self.console = Console()
    # 監控設置
    self.enable_monitoring = enable_monitoring
    self.monitor: Optional[WandbMonitor] = None
    self.session_start_time = time.time()
    # ...
```

### 2. 添加缺少的匯入語句

```python
from src.monitoring import WandbMonitor, initialize_global_monitor, create_config_from_env
```

## 測試修復

### 方法 1: 執行測試腳本

```bash
python test_startup.py
```

這會測試：
- 所有模組是否可以正常匯入
- CLI 類別是否可以正常初始化
- 監控設置功能是否正常
- 命令行參數解析是否正常

### 方法 2: 直接測試主程式

```bash
# 測試預設啟動（含監控）
python main.py --help

# 測試停用監控模式
python main.py --no-monitoring --help

# 測試互動模式（需要完整環境設定）
python main.py --no-monitoring
```

## 預期結果

修復後，程式應該可以：

1. **正常顯示幫助資訊**：
   ```bash
   python main.py --help
   ```

2. **支援監控開關**：
   - `python main.py`（預設啟用監控）
   - `python main.py --no-monitoring`（停用監控）

3. **正確初始化所有組件**，包括：
   - 監控功能（可選）
   - RAG 系統
   - 索引建構器
   - CLI 介面

## 後續使用

修復完成後，你可以：

1. **設定環境變數**（如果要使用 W&B 監控）：
   ```bash
   # 在 .env 檔案中設定
   WANDB_API_KEY=your_api_key_here
   WANDB_MODE=online  # 或 offline, disabled
   ```

2. **啟動系統**：
   ```bash
   python main.py
   ```

3. **如果不需要監控**：
   ```bash
   python main.py --no-monitoring
   ```

程式現在應該可以正常啟動了！