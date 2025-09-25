# W&B 監控整合指南

本專案已完整整合 Weights & Biases (W&B) 監控功能，可追蹤 RAG 系統的完整運作流程。

## 🚀 快速開始

### 1. 安裝依賴項

首先安裝新增的 wandb 依賴：

```bash
pip install wandb>=0.15.0
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

### 2. 設定 W&B

#### 2.1 獲取 W&B API Key

1. 註冊或登入 [Weights & Biases](https://wandb.ai)
2. 前往 [Settings](https://wandb.ai/settings) 頁面
3. 複製你的 API key

#### 2.2 設定環境變數

複製 `.env.template` 為 `.env` 並設定：

```bash
# W&B 設定
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=food-safety-rag
WANDB_ENTITY=your_username_or_team
WANDB_MODE=online  # online, offline, disabled
```

### 3. 使用方式

#### 3.1 啟用監控（預設）

```bash
python main.py  # W&B 監控預設啟用
```

#### 3.2 停用監控

```bash
python main.py --no-monitoring
```

#### 3.3 設定為離線模式

在 `.env` 檔案中設定：
```
WANDB_MODE=offline
```

## 📊 監控指標

### 查詢相關指標

- **query_text**: 查詢內容（截取前100字元）
- **query_type**: 查詢類型分類（penalty, labeling, additives 等）
- **query_enhanced**: 是否使用查詢增強
- **total_time**: 端到端查詢時間
- **retrieval_time**: 檢索時間
- **llm_time**: LLM 回應時間
- **documents_retrieved**: 檢索到的文檔數量
- **similarity_scores**: 相似度分數清單
- **avg_similarity**: 平均相似度
- **response_length**: 回應長度
- **confidence_score**: 信心分數
- **tokens_used**: 估計使用的 token 數量

### 系統指標

- **memory_usage_mb**: 記憶體使用量
- **environment_check_passed**: 環境檢查是否通過
- **index_load_time**: 索引載入時間
- **index_build_time**: 索引建立時間
- **rag_system_initialized**: RAG 系統初始化狀態
- **chroma_setup_time**: ChromaDB 設定時間
- **session_queries**: 會話中的查詢數量
- **session_duration**: 會話持續時間

### 錯誤追蹤

- **error_type**: 錯誤類型
- **error_message**: 錯誤訊息
- **error_context**: 錯誤上下文資訊

## 🎛️ 監控功能

### 1. 即時查詢監控

每次查詢都會記錄：
- 查詢內容和類型
- 執行時間分解
- 檢索品質指標
- 系統資源使用

### 2. 會話統計

追蹤整個使用會話：
- 總查詢數量
- 平均查詢時間
- 會話持續時間
- 成功/失敗率

### 3. 索引建立監控

記錄索引建立過程：
- 文檔處理時間
- 嵌入生成效能
- ChromaDB 連線狀態
- 記憶體使用變化

### 4. 錯誤分析

完整的錯誤追蹤：
- 錯誤類型分類
- 發生頻率統計
- 上下文資訊記錄

## 📈 儀表板功能

W&B 自動提供：

### 系統效能儀表板
- 查詢延遲趨勢圖
- 記憶體使用監控
- 錯誤率統計
- 系統健康度指標

### 查詢分析儀表板
- 查詢類型分布
- 信心分數分布
- 相似度分數統計
- Token 使用量追蹤

### 使用模式分析
- 查詢頻率分析
- 使用時間分布
- 互動模式識別

## 🔧 進階設定

### 自訂專案名稱

```bash
export WANDB_PROJECT=my-custom-rag-project
```

### 設定團隊/組織

```bash
export WANDB_ENTITY=my-team-name
```

### 離線模式

適合網路受限環境：

```bash
export WANDB_MODE=offline
```

同步離線資料：

```bash
wandb sync wandb/offline-run-*
```

## 🧪 測試監控功能

執行測試腳本確認整合正常：

```bash
python test_wandb_integration.py
```

這會測試：
- 監控模組匯入
- 停用模式操作
- 指標記錄功能
- 配置建立

## 📊 監控最佳實踐

### 1. 適當的專案組織

- 為不同環境使用不同專案名稱
- 使用標籤區分實驗類型
- 定期清理舊的執行記錄

### 2. 效能考量

- 監控功能設計為輕量級
- 在生產環境中考慮使用離線模式
- 定期檢查磁碟空間使用

### 3. 隱私保護

- 查詢內容會被截取以保護隱私
- 敏感資訊不會被記錄
- 可使用 `--no-monitoring` 完全停用

## 🔍 疑難排解

### 常見問題

1. **W&B 登入失敗**
   ```bash
   wandb login your_api_key_here
   ```

2. **網路連線問題**
   ```bash
   export WANDB_MODE=offline
   ```

3. **記憶體不足**
   - 調整 `CHUNK_SIZE` 和 `TOP_K_RETRIEVAL`
   - 使用較小的索引

4. **權限問題**
   ```bash
   export WANDB_DIR=/path/to/writable/directory
   ```

### 除錯模式

啟用詳細日誌：

```bash
export WANDB_CONSOLE=wrap
python main.py
```

## 🎯 效益總結

整合 W&B 監控後，你可以：

1. **效能最佳化**: 識別查詢瓶頸，最佳化系統參數
2. **品質監控**: 追蹤回答品質和使用者滿意度
3. **使用分析**: 了解使用模式，改善使用體驗
4. **錯誤管理**: 快速定位和解決問題
5. **容量規劃**: 基於使用趨勢進行系統擴展

監控功能已完全整合到現有工作流程中，不需要改變你的使用方式！