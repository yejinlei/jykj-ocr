# jykj_ocr 实测报告

**部署机**:`http://192.168.0.81:8000`
**代码 HEAD**:`4df31df`(含 `elapsed_ms` 修复)
**测试日期**:2026-09-04
**测试图**:`tests/兰亭序.jpeg`(750×1390,110 KB,中文古文竖排繁体)

---

## 1. Bug 修复验证:`elapsed_ms`

所有 `bestof*` 家族端点此前返回 `elapsed_ms=0`(根因:`BestofEngine.recognise`
没写计时,只 `StrategyEngine.recognise` 会)。修复后:

| 端点 | 修复前 | 修复后 |
|---|---:|---:|
| `POST /ocr/bestof/text` | **0** | **13346 ms** |
| `POST /ocr/bestof-smart/text` | **0** | **11449 ms** |
| `POST /ocr/bestof-fastest/text` | **0** | **11315 ms** |
| `POST /ocr/bestof-confidence/text` | **0** | **11498 ms** |
| `POST /ocr/bestof-longest/text` | **0** | **11311 ms** |
| `POST /ocr/bestof-fluency/text` | **0** | **11273 ms** |
| `POST /ocr/text + strategy_name=bestof-fastest` | **0** | **12266 ms** |
| `POST /ocr/text + strategy_name=bestof-smart` | **0** | **11495 ms** |
| `POST /ocr/bestof-fastest`(multipart) | **0** | **10781 ms** |
| `POST /ocr/rapidocr/text`(baseline) | 9611 | 9611 |

服务端 `elapsed_ms` 与客户端 wall-clock 相差 < 150 ms——计时对上了。

修复 commit:`4df31df fix: BestofEngine.recognise 写入 elapsed_ms(wall-clock)`

---

## 2. 兰亭序 OCR 结果对比

同一张图跑 9 个端点变体:

| 端点 | HTTP | engine | elapsed_ms | regions | text_chars | 内容形态 |
|---|:---:|---|---:|---:|---:|---|
| `/ocr/rapidocr/text` | 200 | rapidocr | 9611 | 166 | 488 | 竖排繁体,拆散成 166 段 |
| `/ocr/bestof/text` | 200 | siliconflow | 13346 | 1 | 324 | 连贯全文,含标点断句 |
| `/ocr/bestof-smart/text` | 200 | siliconflow | 11449 | 1 | 324 | 连贯全文,**推荐** |
| `/ocr/bestof-fastest/text` | 200 | rapidocr | 11315 | 166 | 488 | 竖排繁体(166 segments) |
| `/ocr/bestof-confidence/text` | 200 | siliconflow | 11498 | 1 | 324 | 连贯全文 |
| `/ocr/bestof-longest/text` | 200 | rapidocr | 11311 | 166 | 488 | 竖排繁体,完整 |
| `/ocr/bestof-fluency/text` | 200 | siliconflow | 11273 | 1 | 324 | 连贯全文 |
| `/ocr/text + strategy_name=bestof-fastest` | 200 | rapidocr | 12266 | 166 | 488 | 同上 |
| `/ocr/text + strategy_name=bestof-smart` | 200 | siliconflow | 11495 | 1 | 324 | 连贯全文 |

### 两种模式的核心差异

- **bestof-smart / bestof-confidence / bestof-fluency**(选 siliconflow):
  返回 324 字连贯段落,含"永和九年"到"有感于斯文"全部关键内容,
  标点断句自动加入,可直接读。
- **bestof-fastest / bestof-longest**(选 rapidocr):
  返回 166 个 region,488 字拆散成竖排片段,每个带 bbox 坐标,
  适合版面分析/坐标需求,不适合直接读。

---

## 3. 结论:哪个效果最好

**推荐 `/ocr/bestof-smart`**——综合置信度 + 长度 + 窜行惩罚 + 语义流畅度,
同一张图上效果最好。

### 场景选择表

| 场景 | 推荐预设 |
|---|---|
| 日常文档识别,要可读文本 | **`bestof-smart`** |
| 需要 region 坐标做版面分析 | `bestof-longest`(选 rapidocr) |
| 追求极致速度(可接受文本碎片) | `bestof-fastest`(选 rapidocr) |
| 只跑远程大模型,跳过 local | `vl` |
| 只跑本地引擎,离线 | `local` |
| 窜行降级 + 阅读顺序重排 | `quality`(= `seq-any`) |

---

## 4. 非 OCR 端点可达性

| 端点 | HTTP | 状态 |
|---|:---:|---|
| `GET /health` | 200 | `{status: ok, engines: [rapidocr, siliconflow, multimodal]}` |
| `GET /engines` | 200 | 3 引擎已注册 |
| `GET /presets` | 200 | 15 项预设(14 显式 + `bestof:<mode>`) |
| `GET /config` | 200 | 795 chars,无 API key 泄露 |
| `GET /openapi.json` | 200 | 8 path / 10 operation,全部带 tag |
| `GET /docs` | 200 | Swagger UI 页面加载正常 |

---

## 5. 本地测试基线

```
.venv/Scripts/python -m pytest tests -q
143 passed, 1 warning
```

比 137 基线多 6 个用例:
- 5 个 `TestPresetsEndpoint` 用例覆盖 `GET /presets` 端点
- 1 个 `test_winner_elapsed_ms_reflects_actual_wall_clock` 覆盖 bestof 计时

---

## 6. 已知次要问题(未修)

1. **`image_data` data URI 扩展名推断失败**——`engine/inputs.py` 把
   `data:image/png;base64,...` 内容写为 `.bin` 临时文件,触发
   `400 unsupported input type`。用 `image_b64` 字段传裸 base64 可绕开。
2. **top-level OpenAPI `tags` 字段缺失**——FastAPI ≤ 0.120 已知行为,
   operation 级 tags 都在,Swagger UI 侧栏分组正常渲染,不需要额外处理。
3. **`docs/index.html` 里 "142 passed" 未同步为 143**——纯文档同步。

---

## 附:测试脚本

用 `.venv/Scripts/python` 直接调 requests,输入 `tests/兰亭序.jpeg` 的
base64 编码(227480 chars),`timeout=180s`。所有 9 个 OCR 端点 + 5 个
非 OCR 端点在本轮都实测通过。
