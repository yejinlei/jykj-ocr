# jykj_ocr 用户手册

本手册面向实际使用 jykj_ocr 的开发者与运维人员,涵盖安装、配置、三种调用方式
(CLI / Python API / HTTP)、策略与重试、Docker 部署,以及常见问题排查。

---

## 1. 前置要求

- Python 3.10+
- 本地 OCR 无需任何外部服务或 API key
- 远程多模态 OCR 需要一个 OpenAI 兼容端点 + 对应 API key

## 2. 安装

```bash
git clone <repo> jykj_ocr && cd jykj_ocr
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
pip install -e .                 # 可选,注册 jykj-ocr 命令行入口
```

`requirements.txt` 自包含:核心运行依赖 + 测试依赖(`pytest`、`httpx`、`ruff`)。
`rapidocr-onnxruntime` 已列入其中,首次运行会自动下载模型权重。

验证安装:

```bash
python -m jykj_ocr --list-engines
# rapidocr       本地 RapidOCR(ONNX,离线)
# siliconflow    硅基流动多模态 OCR
# multimodal     通用 OpenAI 兼容端点
```

## 3. 配置

### 3.1 环境变量(远程引擎)

API key 永远只走环境变量,绝不写入代码或配置文件:

```bash
cp .env.example .env
# 编辑 .env:
#   OPENAI_API_KEY=sk-xxxxxxxx
#   OPENAI_BASE_URL=https://api.siliconflow.cn/v1
```

| 变量 | 必需 | 作用 |
|------|:----:|------|
| `OPENAI_API_KEY` | 远程引擎必需 | 通用 key,适配任意 OpenAI 兼容平台 |
| `OPENAI_BASE_URL` | multimodal 必需 | 端点 URL;siliconflow 不设时用内置默认 |
| `SILICONFLOW_API_KEY` | 可选 | 仅当 siliconflow 需与 `OPENAI_*` 不同 key 时覆盖 |
| `JYKJ_OCR_CONFIG` | 可选 | 配置文件路径(默认 `config/config.yaml`) |
| `JYKJ_OCR_PORT` | 可选 | HTTP 服务端口(默认 8000) |
| `JYKJ_OCR_REMOTE_ENGINES` | 可选 | 逗号分隔,把新引擎追加进远程名单(`vl` 侧),见 §7.2 |

切换平台示例(改两个变量即可,代码与配置文件不动):

| 平台 | `OPENAI_BASE_URL` |
|------|-------------------|
| 硅基流动 | `https://api.siliconflow.cn/v1` |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` |
| 本地 vLLM | `http://localhost:8000/v1` |

### 3.2 配置文件 `config/config.yaml`

定义引擎顺序、模型、策略:

```yaml
strategy:
  name: fallback             # local | vl | fallback | quality(默认预设)
  max_retries: 1
  retry_mode: no_text        # no_text | low_confidence | line_overlap | any | none
  min_confidence: 0.7

engines:
  - name: rapidocr           # 本地,离线
    lang: ch
    enabled: true

  - name: siliconflow        # 硅基流动
    model: PaddlePaddle/PaddleOCR-VL-1.5
    base_url: https://api.siliconflow.cn/v1
    temperature: 0.0
    timeout: 180
    prompt: |
      请识别图片中的全部文字内容,按原文版面顺序输出。
      只输出文字,不要翻译、不要解释。

  - name: multimodal         # 通用 OpenAI 兼容(留空 base_url 走 OPENAI_BASE_URL)
    enabled: false
    model: qwen-vl-max
```

### 3.3 配置优先级

```
显式参数 (--engine / ocr(engine=...))
        ↓ 覆盖
环境变量  JYKJ_OCR_* / *_API_KEY / OPENAI_API_KEY / OPENAI_BASE_URL
        ↓ 覆盖
config/config.yaml
        ↓ 覆盖
内置默认值
```

## 4. CLI 用法

```bash
python -m jykj_ocr [source] [options]
```

| 选项 | 默认 | 说明 |
|------|------|------|
| `source` | — | 图片/PDF 路径或 `http(s)://` URL |
| `--engine` | 策略链 | 强制单个引擎 |
| `--format` | `text` | `text` / `markdown` / `json` |
| `-o` | stdout | 输出文件 |
| `--max-pages` | 配置 | PDF 页数上限 |
| `--dpi` | 200 | PDF 渲染 DPI |
| `-c` | `config/config.yaml` | 配置文件 |
| `--strategy-name` | 配置默认 | 一次性预设:`local`/`vl`/`fallback`/`quality`(见 §7) |
| `serve` | — | 启动 HTTP 服务(`--host`/`--port`) |
| `--list-engines` | — | 列出引擎后退出 |

示例:

```bash
# 离线识别
python -m jykj_ocr scan.png --engine rapidocr --format markdown -o out.md

# 远程识别
python -m jykj_ocr scan.png --engine siliconflow --format json

# PDF,只读前 5 页
python -m jykj_ocr report.pdf --max-pages 5 --dpi 300

# URL
python -m jykj_ocr https://example.com/scan.png

# 策略链(rapidocr 优先,失败回退 siliconflow)
python -m jykj_ocr scan.png

# 命名预设:本次强制走 VL 大模型
python -m jykj_ocr scan.png --strategy-name vl

# 命名预设:rapidocr 窜行自动降级 + 阅读顺序重排
python -m jykj_ocr scan.png --strategy-name quality
```

## 5. Python API

> 完整可运行示例见 `scripts/demo.py`(`python scripts/demo.py [图片路径]`),
> 四种场景一次跑通。

```python
import jykj_ocr

# 指定引擎
results = jykj_ocr.ocr("scan.png", engine="siliconflow")
for r in results:
    print(r.engine, r.model, len(r.regions), "regions")
    print(r.text)

# 策略链(默认引擎顺序)
results = jykj_ocr.ocr("report.pdf", max_pages=10, dpi=300)

# 一次性命名预设(不改配置文件)
results = jykj_ocr.ocr("stamp.png", strategy_name="quality")

# 只要拼接好的 markdown
text = jykj_ocr.ocr_to_text("scan.png", engine="rapidocr")
```

**签名**:

```python
ocr(source: str, *,
   engine: str | None = None,
   config: Config | None = None,
   config_path: str | None = None,
   max_pages: int | None = None,
   dpi: int = 200,
   retries: int = 1,
   strategy_name: str | None = None) -> List[OCRResult]
```

> `source` 必须是文件路径或 `http(s)://` URL,**不接受裸 bytes**。
> 如需识别内存图片,先写入临时文件再传路径。

`OCRResult` 字段:

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 全部文本(拼接) |
| `regions` | `List[TextRegion]` | 各文本块(含 bbox/confidence) |
| `engine` | `str` | 实际使用的引擎 |
| `model` | `str` | 模型名(本地引擎为空) |
| `elapsed_ms` | `int` | 耗时 |
| `width`/`height` | `int` | 页面尺寸 |
| `ok` | `bool` | 是否有可用文本 |

## 6. HTTP API

```bash
python -m jykj_ocr serve --port 8000
```

### 6.1 识别上传文件 `POST /ocr`

multipart 表单:

| 字段 | 必填 | 说明 |
|------|:----:|------|
| `file` | ✅ | 图片或 PDF |
| `engine` | — | 强制引擎 |
| `model` | — | 覆盖模型(仅远程引擎) |
| `prompt` | — | 覆盖 prompt(仅远程引擎) |
| `strategy` | — | JSON 字符串,临时策略 |
| `strategy_name` | — | 一次性命名预设:`local`/`vl`/`fallback`/`quality`(见 §7) |
| `max_pages` | — | PDF 页数上限 |
| `dpi` | 200 | PDF 渲染 DPI |
| `format` | `json` | `json` / `text` / `markdown` |

```bash
curl -s http://localhost:8000/ocr \
  -F "file=@scan.png" -F "engine=siliconflow" -F "format=json"
```

JSON 响应:

```json
{
  "pages": [{"text": "...", "regions": [...], "engine": "siliconflow", "model": "..."}],
  "text": "...",
  "engine": "siliconflow",
  "page_count": 1
}
```

### 6.2 识别图片 URL `POST /ocr/text`

```bash
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png","engine":"multimodal","format":"text"}'

# 也可以用命名预设;URL 无扩展名也能识别(按文件头魔数自动判型)
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/img/u=1234&fm=3074","strategy_name":"vl"}'
```

### 6.3 运行时配置 `GET/POST/DELETE /config`

不重启改模型或引擎顺序:

```bash
# 查看(不泄露 key)
curl -s http://localhost:8000/config

# 切换模型
curl -s -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"engines":[{"name":"siliconflow","model":"Qwen/Qwen2.5-VL-72B"}]}'

# 调整策略
curl -s -X POST http://localhost:8000/config \
  -d '{"strategy":{"max_retries":2,"retry_mode":"low_confidence","min_confidence":0.8}}'

# 还原
curl -s -X DELETE http://localhost:8000/config
```

`GET /config` 永远只返回 `has_api_key: true/false`,不返回 key 明文。

### 6.4 辅助端点

- `GET /health` → `{"status":"ok","engines":[...]}`
- `GET /engines` → 可用引擎 + 当前配置顺序

### 6.5 异常映射

| 异常 | HTTP |
|------|------|
| `InputError`(坏输入/空页) | 400 |
| `EngineNotAvailable`(缺依赖/key/URL) | 422 |
| `EngineError`(引擎失败) | 502 |
| `StrategyError`(链耗尽) | 422 |

## 7. 引擎使用策略

### 7.1 命名策略预设(strategy presets)

项目把常用引擎组合固化成四个**命名预设**,配置里固定默认策略,单次请求可临时切换
(一次性,不改动服务端 base config):

| 预设 | 引擎范围 | retry_mode | 阅读顺序重排 | 适用场景 |
|------|----------|------------|:----:|----------|
| `local` | 仅本地引擎(rapidocr 家族) | `no_text` | — | 离线、隐私敏感、批量低成本 |
| `vl` | 仅远程 VL 大模型(siliconflow/multimodal) | `no_text` | — | 版面复杂、手写、表格、**需要整理后的连贯文本** |
| `fallback` | 全部启用引擎,按配置顺序回退(**默认**) | `no_text` | — | 通用生产链路 |
| `quality` | 同 fallback + 窜行降级 + 阅读顺序重排 | `any` | ✅ | 盖章/倾斜导致 rapidocr 窜行,或本地优先但需要回退整理 |

`quality` 的完整逻辑:先用 `any` 模式判定 rapidocr 结果是否低置信度或**窜行**
(`detect_line_overlap`:超长宽比合并框 + 双轴重叠框),不合格则降级到 VL 引擎;
最终输出前按区域坐标做阅读顺序重排(`rebuild_text_from_regions`,
`output.reorder_lines`)。

**用法**:

```bash
# CLI
python -m jykj_ocr scan.png --strategy-name quality

# Python API
jykj_ocr.ocr("scan.png", strategy_name="vl")
```

```bash
# HTTP —— /ocr(multipart)与 /ocr/text(JSON)均支持
curl -s http://localhost:8000/ocr -F "file=@scan.png" -F "strategy_name=quality"
curl -s http://localhost:8000/ocr/text -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png","strategy_name":"vl"}'
```

未知名称:CLI 报 argparse 错并列出可选值;HTTP 返回 400
`unknown strategy 'xxx'; choose one of local, vl, fallback, quality`。

**优先级**:单次请求 `strategy_name` > `POST /config` 运行时覆盖 > `config.yaml`。
预设展开走 `apply_strategy_preset`,返回 deepcopy——输入配置永不被改动。

### 7.2 更多 OCR 引擎接入

本地/远程划分走 `remote_engines()`(内置 siliconflow、multimodal)。新注册引擎
**无需改代码**即被预设识别:`local` 保留它(除非列入远程名单)、`vl` 排除它、
`fallback`/`quality` 尊重其 `enabled` 标志。要把新厂商归入远程侧:

```bash
export JYKJ_OCR_REMOTE_ENGINES="paddlecloud,acme-vl"   # 逗号分隔,小写
```

### 7.3 底层重试模式(retry_mode)

预设最终都落到 `retry_mode`。策略引擎按 `config.engines` 顺序尝试,每个引擎最多
重试 `max_retries` 次,由下列谓词决定是否换引擎:

| `retry_mode` | 行为 |
|--------------|------|
| `no_text` | 结果无文本 → 重试/切换(默认) |
| `low_confidence` | 平均置信度 < `min_confidence` → 重试/切换 |
| `line_overlap` | 无文本**或**检测到窜行 → 重试/切换 |
| `any` | 低置信度**或**窜行任一命中 → 重试/切换(`combine_predicates` 组合) |
| `none` / `first_success` | 第一个成功结果即返回,不重试 |

链耗尽时:返回所有尝试中 `ok` 且文本最长的结果(best-of 兜底);若无任何 `ok`
结果,抛 `StrategyError`(HTTP 422)。手动调参示例:

```bash
curl -s -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"strategy":{"retry_mode":"any","min_confidence":0.75,"max_retries":2}}'
```

## 8. Docker 部署

```bash
# 构建并启动(含 healthcheck + 模型权重持久化)
docker compose up --build

# 单容器
docker build -t jykj_ocr .
docker run --rm -p 8000:8000 --env-file .env jykj_ocr

# 容器内一次性任务
docker run --rm --env-file .env -v "$PWD:/data" jykj_ocr \
    python -m jykj_ocr /data/scan.png --engine siliconflow
```

- 基础镜像 `python:3.11-slim`,非 root 用户运行
- `HEALTHCHECK` 探测 `/health`
- 配置/端口:`JYKJ_OCR_CONFIG` / `JYKJ_OCR_PORT`
- compose 持久化 RapidOCR 模型权重到 `rapidocr-models` 卷,避免重复下载

## 9. 测试

```bash
.venv/Scripts/python -m pytest tests -q
```

- **CI 基线**:98 个用例,全部离线运行,无真实 API 调用,monkeypatch 模拟引擎返回
- 覆盖:models、config、strategy、engines(multimodal OpenAI 响应解析、rapidocr
  1.x/1.4.x/2.x 返回形态)、策略预设(local/vl/fallback/quality、deepcopy 不变性、
  `JYKJ_OCR_REMOTE_ENGINES` 扩展)、窜行检测与阅读顺序重排、inputs 魔数识别

端到端接口测试(需联网 + key,非 CI):

```bash
export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://api.siliconflow.cn/v1
.venv/Scripts/python scripts/test_interfaces.py
```

## 10. 常见问题

**Q: multimodal 报 "no base URL"?**
A: 没设 `OPENAI_BASE_URL` 且 config.yaml 中 multimodal 的 `base_url` 留空。multimodal
   不会自动回退到硅基流动(避免把别家 key 发到硅基流动),必须显式给 URL。

**Q: siliconflow 报 HTTP 402 / 余额不足?**
A: 账号余额不足,需充值。key 格式正确,请求已到达模型端点。

**Q: rapidocr 返回乱码或全是数字?**
A: `rapidocr-onnxruntime` 1.4.x 返回 `(results, elapsed)` 2-tuple,与 1.x 不同。
   已在 `_run()` 用 `_looks_like_results_list()` 适配;若升级到新大版本需检查返回形态。

**Q: 怎么换平台?**
A: 改 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY` 两个环境变量,代码与配置不动。
   multimodal 引擎会自动走新端点。

**Q: 内存里的图片怎么识别?**
A: `ocr()` 只收路径/URL。先 `tempfile.NamedTemporaryFile` 写盘再传路径。

**Q: API key 会被泄露吗?**
A: 不会。`.env` 已 gitignore;`GET /config` 只返回 `has_api_key` 布尔;
   运行时覆盖 `POST /config` 接受 `api_key` 字段但同样不回显明文。
