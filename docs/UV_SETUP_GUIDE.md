# 使用 UV 虛擬環境啟動 LamaIndex RAG 專案

## 關於 UV 🚀

UV 是一個極速的 Python 套件安裝器和解析器，由 Rust 編寫。相比傳統的 pip + venv，UV 提供：
- ⚡ **超快速度**: 安裝套件比 pip 快 10-100 倍
- 🔒 **可靠依賴解析**: 避免依賴衝突
- 💾 **全域快取**: 節省磁碟空間
- 🛡️ **安全性**: 內建安全檢查

## 環境需求 📋

- Python 3.8 或更高版本
- UV (如尚未安裝，下方有安裝指南)

## 第一步：安裝 UV

### Windows 系統

使用 PowerShell：
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

或使用 pip：
```bash
pip install uv
```

### macOS/Linux 系統

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或使用 brew (macOS)：
```bash
brew install uv
```

### 驗證安裝

```bash
uv --version
```

## 第二步：建立專案虛擬環境 🏗️

### 方法一：自動建立虛擬環境

導航到專案根目錄：
```bash
cd LamaIndex
```

使用 UV 自動建立虛擬環境並安裝依賴：
```bash
# 建立虛擬環境
uv venv

# 啟動虛擬環境
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 使用 UV 安裝依賴套件
uv pip install -r requirements.txt
```

### 方法二：一步到位安裝

UV 可以直接在虛擬環境中執行指令，無需手動啟動：
```bash
# 建立虛擬環境並安裝依賴
uv venv
uv pip install -r requirements.txt
```

## 第三步：設定環境變數 ⚙️

建立 `.env` 檔案：
```bash
# 複製環境範本
cp .env.template .env

# 編輯 .env 檔案，設定您的 OpenAI API Key
# Windows
notepad .env

# macOS/Linux
nano .env
```

在 `.env` 檔案中設定：
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 第四步：使用 UV 執行專案 🚀

### 方法一：在啟動的虛擬環境中執行

```bash
# 啟動虛擬環境
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 執行專案
python main.py
```

### 方法二：使用 UV 直接執行（推薦）

UV 可以自動在虛擬環境中執行 Python 指令：

```bash
# 執行主程式
uv run python main.py

# 執行系統測試
uv run python test_system.py

# 執行安裝程式
uv run python setup.py

# 單次查詢
uv run python main.py -q "食品添加物有什麼限制？"

# 批次查詢
uv run python main.py --batch config/query_examples.txt
```

## UV 的優勢展示 ⚡

### 速度比較

傳統方式：
```bash
# 建立虛擬環境: ~3-5 秒
python -m venv .venv

# 啟動虛擬環境
source .venv/bin/activate

# 安裝套件: ~30-60 秒
pip install -r requirements.txt
```

使用 UV：
```bash
# 建立虛擬環境: ~0.1 秒
uv venv

# 安裝套件: ~5-10 秒
uv pip install -r requirements.txt
```

### 依賴解析

UV 提供更好的依賴解析，避免套件衝突：
```bash
# 檢查依賴衝突
uv pip check

# 顯示依賴樹
uv pip show --tree llama-index
```

## 常用 UV 指令 📝

### 套件管理

```bash
# 安裝特定套件
uv pip install pandas

# 安裝開發依賴
uv pip install -r requirements.txt pytest black

# 升級套件
uv pip install --upgrade llama-index

# 移除套件
uv pip uninstall pandas

# 凍結當前環境
uv pip freeze > requirements-lock.txt
```

### 虛擬環境管理

```bash
# 建立指定 Python 版本的虛擬環境
uv venv --python 3.9

# 建立指定名稱的虛擬環境
uv venv my-env

# 移除虛擬環境
rm -rf .venv
```

### 專案執行

```bash
# 在虛擬環境中執行任意指令
uv run python -c "import llama_index; print('LlamaIndex 已安裝')"

# 執行測試
uv run pytest tests/

# 執行格式化
uv run black src/

# 執行型別檢查
uv run mypy src/
```

## 完整工作流程範例 🎯

以下是使用 UV 從零開始建立並執行本專案的完整流程：

```bash
# 1. 克隆或下載專案到本地
cd LamaIndex

# 2. 建立虛擬環境
uv venv

# 3. 安裝依賴套件
uv pip install -r requirements.txt

# 4. 設定環境變數
cp .env.template .env
# 編輯 .env 設定 OPENAI_API_KEY

# 5. 執行系統測試
uv run python test_system.py

# 6. 啟動專案
uv run python main.py
```

## 疑難排解 🔧

### 常見問題

**Q1: 依賴版本衝突錯誤**
```
錯誤訊息: No solution found when resolving dependencies
```

這是因為套件版本不相容。UV 的嚴格依賴解析會檢測出版本衝突：

```bash
# 解決方案 1: 使用更新的 requirements.txt (已修復)
uv pip install -r requirements.txt

# 解決方案 2: 如果仍有問題，逐步安裝核心套件
uv pip install llama-index
uv pip install beautifulsoup4 requests pandas numpy
uv pip install chromadb rich tqdm

# 解決方案 3: 使用寬鬆版本約束
uv pip install --resolution=lowest-direct -r requirements.txt
```

**Q2: UV 指令找不到**
```bash
# 檢查 UV 是否正確安裝
uv --version

# 如果未安裝，重新安裝
pip install uv
```

**Q3: 虛擬環境建立失敗**
```bash
# 檢查 Python 版本
python --version

# 指定 Python 版本建立環境
uv venv --python python3.9
```

**Q4: 套件安裝緩慢**
```bash
# 使用鏡像源加速
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**Q5: 權限問題 (Windows)**
```powershell
# 以管理員身份執行 PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 效能優化

```bash
# 啟用並行安裝（預設啟用）
export UV_CONCURRENT_DOWNLOADS=10

# 使用全域快取
export UV_CACHE_DIR=$HOME/.cache/uv

# 檢視快取使用情況
uv cache info
```

## 與其他工具整合 🔄

### 與 Git 整合

建議將以下內容加入 `.gitignore`：
```
.venv/
uv.lock
__pycache__/
.env
```

### 與 IDE 整合

**VS Code:**
1. 安裝 Python 擴充功能
2. 選擇解釋器：`Ctrl+Shift+P` → "Python: Select Interpreter"
3. 選擇 `.venv/Scripts/python.exe` (Windows) 或 `.venv/bin/python` (macOS/Linux)

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. 選擇 Existing Environment
3. 指向 `.venv` 中的 Python 執行檔

## 專案部署建議 🚢

### 開發環境

```bash
# 使用 UV 管理開發依賴
uv pip install -r requirements.txt
uv pip install pytest black mypy  # 開發工具
```

### 生產環境

```bash
# 建立 requirements-lock.txt 確保版本一致
uv pip freeze > requirements-lock.txt

# 在生產環境使用鎖定版本
uv pip install -r requirements-lock.txt
```

## 結論 🎊

使用 UV 管理本 RAG 專案的優勢：

✅ **快速**: 套件安裝和虛擬環境建立都極其快速
✅ **簡潔**: `uv run` 指令讓執行更簡單
✅ **可靠**: 更好的依賴解析避免衝突
✅ **現代**: 採用最新的 Python 生態系最佳實踐

現在您可以享受更高效的 Python 開發體驗！

---

**🚀 現在開始使用 UV 運行您的 RAG 系統吧！**

```bash
uv venv && uv pip install -r requirements.txt && uv run python main.py
```