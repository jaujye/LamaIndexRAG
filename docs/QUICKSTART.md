# 快速開始指南

## 5分鐘快速上手 🚀

### 第一步：自動安裝

執行安裝腳本，它會自動幫您設置所有環境：

```bash
python setup.py
```

安裝過程中會：
- ✅ 檢查 Python 版本
- ✅ 安裝所需套件
- ✅ 設置環境檔案
- ✅ 建立目錄結構
- ✅ 執行系統測試

### 第二步：開始使用

安裝完成後，立即開始使用：

```bash
python main.py
```

系統會自動：
1. 從法務部網站下載最新法規 📥
2. 建立智慧索引 🧠
3. 啟動問答介面 💬

### 第三步：開始提問

系統啟動後，您就可以開始問問題了！

#### 常見問題範例：

**關於食品添加物：**
```
問：食品添加物有什麼使用限制？
答：根據食品安全衛生管理法第18條規定，食品添加物之使用應符合以下規定：
    一、使用範圍及限量標準，應符合中央主管機關之規定...
```

**關於處罰規定：**
```
問：販售過期食品會受到什麼處罰？
答：依據第44條規定，販賣逾有效日期之食品者，處新臺幣六萬元以上二億元以下罰鍰...
```

**關於標示要求：**
```
問：食品標示需要包含什麼內容？
答：依第22條規定，食品及食品原料之容器或外包裝，應明顯標示下列事項：
    一、品名... 二、內容物名稱... 三、淨重、容量或數量...
```

## 進階使用 🎯

### 單次查詢模式

不需要進入互動模式，直接查詢：

```bash
python main.py -q "食品製造業者的衛生條件是什麼？"
```

### 批次查詢模式

一次處理多個問題：

```bash
python main.py --batch config/query_examples.txt
```

### 系統維護

更新法規資料：
```bash
python main.py --fetch-data
```

重建索引：
```bash
python main.py --rebuild-index
```

檢視系統狀態：
```bash
python main.py --stats
```

## 系統測試 🧪

隨時檢查系統狀態：

```bash
python test_system.py
```

測試會檢查：
- 🔧 環境設置
- 📁 檔案結構
- 📥 資料擷取功能
- 📄 文件處理功能
- 🗂️ 向量索引功能
- 🤖 RAG 查詢功能

## 常見問題 ❓

### Q: 第一次使用需要多久？
**A:** 約5-10分鐘，包含：
- 套件安裝：2-3分鐘
- 法規下載：3-5分鐘
- 索引建立：2-3分鐘

### Q: 如何獲得更準確的答案？
**A:** 提問技巧：
- ✅ 使用具體的法律術語
- ✅ 明確描述情境：「餐廳業者...」
- ✅ 指定查詢類型：「處罰規定」、「標示要求」
- ❌ 避免過於籠統的問題

### Q: 系統支援哪些查詢類型？
**A:** 自動識別的查詢類型：
- 🏛️ **penalty**: 處罰、罰則相關
- 🏷️ **labeling**: 標示、標籤相關
- 🧪 **additives**: 食品添加物相關
- 🧽 **hygiene**: 衛生、清潔相關
- 🔍 **inspection**: 檢驗、稽查相關
- 📦 **import**: 進口、邊境管制相關
- 🏭 **manufacturing**: 製造、加工相關

### Q: 如何提升系統效能？
**A:** 效能優化建議：
- 調整 `similarity_top_k` 參數（預設5）
- 修改 `similarity_cutoff` 閾值（預設0.6）
- 在 `src/rag_system.py` 中實作自訂相關度評分

## 自訂和擴展 🛠️

### 實作自訂相關度評分

編輯 `src/rag_system.py` 中的 `calculate_relevance_score()` 函數：

```python
def calculate_relevance_score(self, query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 您的自訂邏輯
    for source in sources:
        base_score = source['similarity_score']

        # 根據查詢類型調整權重
        if 'penalty' in query and source['chunk_type'] == 'article_penalties':
            source['relevance_score'] = base_score * 1.2
        else:
            source['relevance_score'] = base_score

    return sorted(sources, key=lambda x: x['relevance_score'], reverse=True)
```

### 新增其他法規

1. 修改 `src/data_fetcher.py` 支援新的網站
2. 調整 `src/document_processor.py` 處理不同格式
3. 更新 `src/rag_system.py` 的查詢分類

## 技術支援 💡

如果遇到問題：

1. **首先執行測試**：`python test_system.py`
2. **檢查環境設置**：確認 `.env` 檔案中的 API key
3. **查看詳細文件**：閱讀 `README.md`
4. **重新安裝**：刪除 `chroma_db` 目錄並重新執行

---

**🎉 現在開始探索台灣食品安全法規吧！**

輸入 `python main.py` 開始您的第一次查詢！