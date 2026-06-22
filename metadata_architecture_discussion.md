# 結構化儲存與查詢架構討論

> 本文記錄從 keywords 設計出發，逐步推演出「什麼要在 ingest 時存、什麼留到 query 時才萃取」的完整討論脈絡。每個解法都帶出新問題，呈現真實的設計取捨。

---

## 一、Keywords 設計

### 現行做法

`extract_keywords` 採兩段式 LLM 萃取：

```
markdown
  → _split_sections()（依 #/##/### 切割）
  → Pass 1：每個 section 個別送 LLM 萃取候選詞（截斷至 800 字元）
  → Pass 2：dedup + 排名 → 最多 10 個 keywords
```

### 發現的問題

**問題 1：split 方式的缺陷**

`_split_sections()` 依 heading 切割，對以下文件類型失效：
- 評核表、護理評估表（無 heading 的表格式文件）
- 照片路徑的 `vision_description`（連續白話文，無 `#` 標記）
- DI 組合的 markdown（段落前綴不一定有 heading 格式）

失效結果：整份文件被視為單一 section，所有內容進入一次 800 字元截斷的 Pass 1 呼叫，尾段內容永遠丟失。

**問題 2：固定 10 個 keywords 不分文件長度**

6 頁 SOP 與 100 頁癌症指引都只產出 10 個 keywords。
- 6 頁 SOP：覆蓋率約 40–60%
- 68 頁癌症指引：覆蓋率約 3–7%
- 100 頁住院紀錄：覆蓋率約 2–3%

**問題 3：Bug — `build_metadata()` 從未實際呼叫 `extract_keywords()`**

重構後 `build_metadata(markdown=..., llm=...)` 雖接收這兩個參數，但內部只調用 `_extract_pdf_keywords()`（讀 PDF embedded metadata，幾乎永遠為空）。三條路徑的 keywords 實際上都是 `[]`。

---

## 二、Keywords 的用途問題

### 原始設計意圖 vs 實際需求

討論中釐清 keywords 的設計意圖是 **Inverted-index**，用於結構化查詢（不一定走 RAG）：

- Example 1：查詢某病人做了哪些檢驗項目
- Example 2：撈取某病人在特定時間段內的就醫紀錄原文

這與「top 10 主題詞摘要」的設計完全不同：

| | 現行設計 | Inverted-index 需求 |
|---|---|---|
| 目標 | 「這份文件在講什麼」 | 「這份文件包含哪些可查詢概念」 |
| 策略 | 摘要式，取最重要的 | 窮舉式，不能遺漏 |
| 數量 | 固定 10 個 | 應隨文件長度正比增加 |

### 衍生問題：語意消歧

就算 keywords 改為窮舉式，仍無法解決語意消歧。

例：查詢「某病人的跌倒次數」

- 「跌倒防護衛教」→ 含「跌倒」，但不是跌倒事件
- 「Morse Fall Scale 評估」→ 含跌倒概念，但不是跌倒事件
- 「病人由床上跌落」→ 才是跌倒事件

**Keywords/Inverted-index 只能回答「文件裡有沒有提到跌倒」，無法回答「病人是否真的發生跌倒」。**

---

## 三、Metadata 欄位的核心矛盾

### 三角矛盾

任何 schema-first 設計都面臨以下三個目標，無法同時滿足：

```
[預測所有查詢需求]
        ↑
        ×
[不重跑歷史資料] ←—× —→ [能回答新的查詢需求]
```

- 欄位設計越多 → ingest 時更昂貴，加新欄位需重跑
- 欄位設計越少 → query 時需要 LLM 掃描更多文件，成本不可行

### 真正的結構型欄位只有幾個

討論後釐清：大多數看起來像 metadata 的欄位，其實是**語意型**（需要 LLM 或規則推斷），而非**結構型**（從檔案本身確定性推導）。

| 欄位 | 類型 | 理由 |
|------|------|------|
| `file_name` | 結構型 | 檔案系統直接讀取 |
| `file_type` | 結構型 | 副檔名 |
| `page_count` | 結構型 | API 回傳或 fitz 讀取 |
| `patient_id` | 結構型 | 父目錄名若為純數字 |
| `document_type` | **語意型** | 需要規則或 LLM 推斷，會無限擴充 |
| `keywords` | **語意型** | LLM 萃取 |
| `record_date` | **語意型** | LLM 從文件內容萃取 |
| `events[]` | **語意型** | LLM 萃取，開放式清單 |

**Layer A 最重要的工作不是填滿 metadata，而是確保 `data.markdown` 和 `data.page_images` 完整保存。** 只要原始轉換結果完整，語意型欄位可以在之後任何時間點從 markdown 重新萃取，而不需要重跑昂貴的 Azure OCR API。

---

## 四、Document Date 問題

### 缺口

現有欄位 `processed_at` 是 Layer A 的**處理時間**，不是文件本身的日期。兩者可能相差數年（補登歷史資料的情境）。

沒有 `record_date`，就無法用時間做 filter，10 萬筆資料無法縮小到可 LLM 處理的範圍。

### 就醫記錄中的多種日期

| 欄位 | 語意 |
|------|------|
| 記錄日期 | 護理師/醫師填寫文件的時間 |
| 事件日期 | 事件實際發生時間（如跌倒發生於 02:30） |
| 就診日期 | 門診/住院開始日 |
| 出院日期 | — |
| 文件修訂日期 | SOP 專屬 |

### 決策：只存 `record_date`

**理由：**
1. 幾乎每份就醫文件都有，格式一致（通常在表頭第一行）
2. 護理師記錄跌倒事件的時間 ≈ 事件時間（當班即記），誤差在數小時，對季度級查詢不影響
3. 其他日期語意各異，不同文件類型不一定都有

**精確事件時間的處理：** 需要精確到分鐘時，由 query-time 的 LLM prompt 同時萃取，不預先存入 metadata。

---

## 五、查詢設計：以「跌倒次數」為例

### 基本流程

```
Step 1：metadata filter（毫秒級）
  WHERE patient_id = X
    AND record_date BETWEEN Y AND Z
    AND document_type = '護理紀錄'
  → 10 萬筆縮減至 ~80 筆

Step 2：判斷資料品質，選擇輸入來源
  qc_level good/warning → Layer A markdown（快）
  qc_level danger       → 原始檔案送 Vision LLM（慢，但不需重跑 OCR）

Step 2.5：關鍵字預篩（純字串，不需 LLM）
  if "跌倒" not in source → 跳過

Step 3：LLM 語意判斷（並行）
  「這筆護理紀錄中記錄了幾次病人實際發生跌倒的事件？
   （跌倒衛教、跌倒風險評估不算）
   只回答數字，0 表示沒有。」

Step 4：count
  total_falls = sum(r.count for r in results)
```

---

## 六、查詢設計中逐步出現的問題

### 問題 A：Context Window

**提問：** 把整份 markdown 送進 LLM，context window 夠嗎？

**分析：** 對護理紀錄不是問題。`document_type = '護理紀錄'` filter 後，剩下的幾乎都是 1–3 頁、1,000–3,000 字的短文件。長文件（癌症指引、SOP）不會出現在這個查詢結果集裡。

**衍生問題：** 出院摘要這類長文件怎麼辦？

---

### 問題 B：長文件分類（top-K 的任意性）

**提問：** 出院摘要可能 5–20 頁，跌倒事件記錄位置不固定，截斷前 N tokens 不可靠。建議改用 per-document chunk 檢索，取前 5 個最相關 chunk。

**問題：** 為什麼是 5？K 值怎麼決定？

**根本矛盾：top-K 是為「找答案」設計的，但這個任務是「文件分類」。**

| | 檢索（RAG） | 文件分類 |
|---|---|---|
| 問題 | 跌倒記錄在哪？ | 這份文件有沒有跌倒事件？ |
| 適合方法 | top-K 向量搜尋 | 需看整份文件 |
| top-K 合理嗎？ | 合理 | 不合理，K 是任意決定 |

**兩種可行方案：**

**Option A：Map-reduce（query time）**
- 每個 chunk 獨立判斷「這段有沒有跌倒事件？是/否」
- 任一 chunk 回答「是」→ 文件有跌倒事件
- 不需決定 K，但 LLM 呼叫次數 = chunk 數量，**昂貴**

**Option B：ingest-time 事件標記**
- 文件進入系統時，一次性用 LLM 產出 `metadata["events"] = ["跌倒事件", ...]`
- 查詢時直接 `WHERE '跌倒事件' = ANY(events)`，不需任何 chunk 操作
- **衍生問題：** `events[]` 是開放式清單，每加一種事件類型就需要重跑所有文件 → 回到三角矛盾

---

### 問題 C：yes/no 無法計算同一文件內的多次跌倒

**提問：** 一份護理紀錄可能記錄「02:30 跌倒 + 14:00 再次跌倒」。目前只問「有沒有」，會把兩次算成一次。

**修正：** 把 yes/no 改成回傳次數

```
「這筆護理紀錄中記錄了幾次實際跌倒事件？（數字，0 表示沒有）」
```

**衍生問題：跨文件重複計算**

同一次跌倒可能出現在多份文件：

| 文件 | 內容 |
|------|------|
| 護理紀錄（夜班） | 「02:30 病人跌倒」|
| 護理紀錄（白班） | 「病人昨夜跌倒，今日持續觀察」|
| 出院摘要 | 「住院期間發生一次跌倒」|

三份文件各自回傳 1 → 加總為 3，但實際只有 1 次。

**解法：** 要求 LLM 同時回傳事件時間，再做 dedup

```python
unique_falls = {f.time for r in results for f in r.falls if f.time != "unknown"}
total_falls = len(unique_falls)
```

**衍生問題：** 每次查詢精度提高一點，prompt 設計就更複雜，且無法從 metadata filter 中獲益，每次都是 query-time LLM 處理。

---

## 七、核心取捨總結

### Ingest time vs Query time

| | Ingest time 萃取 | Query time 萃取 |
|---|---|---|
| 代表方法 | `events[]` 欄位、`record_date` | Map-reduce LLM、prompt 萃取事件時間 |
| Ingest 成本 | 高（每份文件跑 LLM） | 低 |
| Query 成本 | 低（純 DB filter） | 高（N × LLM 呼叫） |
| 新增查詢需求 | 需重跑 ingest | 不需要 |
| 適合情境 | 已知、穩定、高頻的查詢 | 臨時、探索性的查詢 |

### 實務建議

| 欄位 | 時機 | 理由 |
|------|------|------|
| `patient_id`, `file_type`, `page_count` | 結構型，ingest 時確定性產出 | 不需 LLM，永遠不需重跑 |
| `record_date` | ingest 時 LLM 萃取（一次性） | 每次查詢都需要，值得提前算 |
| `document_type` | ingest 時規則萃取（粗粒度） | 縮小 filter 範圍，不需精細 |
| `events[]` | 高頻法規必報項目考慮 ingest-time | 跌倒、壓瘡等法規要求查詢 |
| 精確事件時間、次數、dedup | query-time LLM 萃取 | 需求多變，維護成本低 |

### 不可繞過的前提

**`record_date` 是整個查詢架構能運作的單一必要前提。**

沒有 `record_date`，Step 1 的時間篩選失效，10 萬筆無法縮小，query-time LLM 成本不可行，整個架構崩潰。

---

## 八、問題鏈示意

```
Keywords 固定 10 個
  └─→ 長文件覆蓋率 2–3%
        └─→ 改為窮舉式 NER
              └─→ 仍無法解決語意消歧（跌倒衛教 ≠ 跌倒事件）
                    └─→ 改用 LLM 語意分類
                          └─→ 需要 metadata filter 先縮小範圍
                                └─→ record_date 缺失
                                      └─→ 補上 record_date
                                            └─→ 多種日期類型，存哪個？
                                                  └─→ 存 record_date（記錄日期）
                                                        └─→ 精確事件時間由 query-time LLM 萃取

LLM 語意分類
  └─→ 長文件 context window 問題
        └─→ per-document chunk 檢索
              └─→ top-K 任意，K 怎麼決定？
                    └─→ Map-reduce（不需決定 K）
                          └─→ LLM 呼叫次數 = chunk 數，昂貴
                                └─→ ingest-time events[]
                                      └─→ 開放式清單，新需求需重跑
                                            └─→ 三角矛盾，無解

Yes/No 分類
  └─→ 同一文件多次跌倒無法計數
        └─→ 改為回傳次數
              └─→ 跨文件同一事件被重複計算
                    └─→ 加上事件時間做 dedup
                          └─→ prompt 越來越複雜，每次都是 query-time 處理
```
