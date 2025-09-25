# 快速修復指南

## 問題說明

遇到兩個主要問題：
1. `W&B not available. Install with: pip install wandb`
2. `name 'time' is not defined`

## 解決方案

### 選項 1: 使用簡化版（推薦用於快速測試）

使用 `main_no_wandb.py`，這是不含 W&B 監控的簡化版本：

```bash
python main_no_wandb.py --help
python main_no_wandb.py  # 啟動互動式介面
```

### 選項 2: 安裝 W&B 並使用完整版

```bash
# 安裝 wandb
pip install wandb>=0.15.0

# 或者如果你想使用虛擬環境
# 先啟動虛擬環境
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate.bat  # Windows

# 然後安裝
pip install wandb>=0.15.0

# 使用完整版
python main.py --help
python main.py  # 含監控功能
python main.py --no-monitoring  # 不含監控功能
```

### 選項 3: 臨時停用監控功能

設定環境變數來停用監控：

```bash
# 在 .env 檔案中設定
echo "WANDB_MODE=disabled" >> .env

# 或者使用命令行參數
python main.py --no-monitoring
```

## 已修復的問題

1. ✅ **time 模組匯入問題**: 已在 main.py 中添加 `import time`
2. ✅ **W&B 錯誤處理**: 監控器初始化現在有異常處理，不會導致程式崩潰
3. ✅ **提供簡化版本**: `main_no_wandb.py` 可直接使用，不需要安裝 wandb

## 測試步驟

### 測試簡化版（最簡單）
```bash
python main_no_wandb.py --help
```

### 測試完整版
```bash
python main.py --help
```

如果出現 W&B 相關錯誤，程式會顯示警告但繼續運行。

## 推薦使用流程

1. **立即可用**: 使用 `python main_no_wandb.py` 開始使用系統
2. **稍後升級**: 當你想要使用監控功能時，安裝 wandb 並使用 `python main.py`

## 功能對比

| 功能 | main_no_wandb.py | main.py (無 wandb) | main.py (有 wandb) |
|------|------------------|-------------------|-------------------|
| 基本查詢 | ✅ | ✅ | ✅ |
| 索引建立 | ✅ | ✅ | ✅ |
| 批次查詢 | ✅ | ✅ | ✅ |
| W&B 監控 | ❌ | ❌ | ✅ |
| 查詢統計 | ❌ | ❌ | ✅ |
| 效能分析 | ❌ | ❌ | ✅ |
| 錯誤追蹤 | ❌ | ❌ | ✅ |

兩個版本都能提供完整的 RAG 查詢功能！