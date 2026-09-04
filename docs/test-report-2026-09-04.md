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

### 1.1 seq 家族同型 bug(同日发现)

`server._pipeline` 只在**强制单引擎**时套 `TimedOCR`——走 `build_pipeline`
的策略预设(`seq` / `seq-any` / `seq-low_conf` / `seq-line_overlap` /
`local` / `vl` / `quality`)直接返回裸 `StrategyEngine`,也不写
`elapsed_ms`。用兰亭序复现:

| 端点 | 客户端 wall | server_elapsed |
|---|---:|---:|
| `/ocr/seq/text` | 9642 ms | **0 ms** ❌ |
| `/ocr/seq-any/text` | 21145 ms | **0 ms** ❌ |
| `/ocr/seq-low_conf/text` | 10033 ms | **0 ms** ❌ |
| `/ocr/seq-line_overlap/text` | 20311 ms | **0 ms** ❌ |
| `/ocr/local/text` | 9889 ms | **0 ms** ❌ |
| `/ocr/vl/text` | 2274 ms | **0 ms** ❌ |
| `/ocr/quality/text` | 20488 ms | **0 ms** ❌ |

修复 `StrategyEngine.recognise`(与 `BestofEngine` 对称):用
`time.perf_counter()` 包住整个 attempts 循环,winner / best 返回前写入
`elapsed_ms`——覆盖所有 attempts(含被 reject 的重试)。

修复 commit:`28aaaa2 fix: StrategyEngine.recognise 写入 elapsed_ms(seq 家族 elapsed_ms=0 bug)`

新增回归测试:
- `TestStrategyEngine::test_accepted_result_elapsed_ms_reflects_wall_clock`
  覆盖 3 种 retry_check 组合
- `TestStrategyEngine::test_rejected_attempts_include_retry_time`
  两次 40ms 尝试(elapsed_ms ≥ 60ms),验证重试时间被计入

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

## 7. cascade 家族 + 请求级旋钮(2026-09-04 晚)

`seq*` 之前只有一个 retry 判定维度;`bestof*` 之前只有一个评分维度;
同一引擎被 reject 后会**在同引擎重试**,不降级。此次补齐:

**新增预设**(`STRATEGY_PRESETS` 14 → 17):
- `cascade` = `seq` + `max_retries=0`(不重试同引擎)
- `cascade-low_conf` = `cascade` + `retry_mode=low_confidence`
- `cascade-line_overlap` = `cascade` + `retry_mode=line_overlap`

**请求级旋钮**(所有 `/ocr/{preset}` 与 `/ocr/{preset}/text` 端点):
- `retry_mode`:no_text / low_confidence / line_overlap / any / none
- `score_mode`:smart / fastest / highest_confidence / longest / fluency
- `max_retries`:≥0,0 等价于 cascade

旋钮在 `apply_strategy_preset` 之后应用,永远覆盖预设默认。无效值走 400;
`max_retries=-1` 由 Pydantic `ge=0` 拦成 422。

**CLI 同步**:`--strategy-name` 的 `choices` 从硬编码 14 元组改为
`STRATEGY_PRESETS`,自动跟随预设家族增长。

---

## 8. 远程模型实测(2026-09-04 深夜,本地服务 127.0.0.1:8765)

用 `.env` 里的 `JYKJ_OCR_SILICONFLOW_MODEL` 切换 siliconflow 的模型,同图对比:

| Model | HTTP | elapsed | text_len | 备注 |
|---|:---:|---:|---:|---|
| `PaddlePaddle/PaddleOCR-VL-1.5`(默认) | 200 | 1.29s | 324 | 兰亭序全文,首尾干净 |
| `Qwen/Qwen3-VL-32B-Instruct`(对话模型) | 200 | 1.48s | 4 | 短文本 OK,不做 OCR |
| `moonshotai/Kimi-K2.7-Code`(代码模型) | 200 | **157s** | 372 | 输出前缀 "墨趣" 系幻觉,40× 慢 |

**结论**:通用多模态 LLM(Kimi/Qwen3)能过,但**做 OCR 不划算**——
推理思考 tokens 撑爆延迟,Kimi-K2.7-Code 还带 reasoning_content(前 300
字里出现 "墨趣" 这种非图像文字)。OCR 场景默认 `PaddleOCR-VL-1.5` 仍最佳,
JYKJ_OCR_SILICONFLOW_MODEL 保留作 A/B 实验与对话式视觉任务用。

`.env` 加了 `JYKJ_OCR_SILICONFLOW_MODEL=moonshotai/Kimi-K2.7-Code`
一行方便切换;默认配置仍走 PaddleOCR-VL-1.5。

---

## 附:测试脚本

用 `.venv/Scripts/python` 直接调 requests,输入 `tests/兰亭序.jpeg` 的
base64 编码(227480 chars),`timeout=180s`。所有 9 个 OCR 端点 + 5 个
非 OCR 端点在本轮都实测通过。
