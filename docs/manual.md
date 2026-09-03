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
  name: fallback             # local | vl | seq* | bestof* | fallback | quality(默认预设)
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
| `--strategy-name` | 配置默认 | 一次性预设:`local`/`vl`/`seq*`/`bestof*`(见 §7) |
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

# 最佳策略:所有引擎各跑一次,选识别质量最优
python -m jykj_ocr scan.png --strategy-name bestof
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

# 最佳策略(所有引擎各跑一次)
results = jykj_ocr.ocr("scan.png", strategy_name="bestof-smart")

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

> `source` 支持文件路径、`http(s)://` URL,以及 `data:` URI(如
> `data:image/png;base64,...`)。**不接受裸 bytes**。
> 如需识别内存图片,先 `base64.b64encode(img_bytes).decode()` 后拼接为
> `data:image/png;base64,<payload>` 传入,或先写入临时文件再传路径。

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

服务启动后访问 `http://localhost:8000/docs` 可查看交互式 Swagger 文档。

### 6.1 端点一览

| 方法 | 路径 | 输入方式 | 输出结构 |
|------|------|----------|:---:|
| `POST` | `/ocr` | multipart 上传文件 | ✅ 统一(见 §6.3) |
| `POST` | `/ocr/{preset}` | multipart 上传文件 | ✅ 统一(见 §6.3) |
| `POST` | `/ocr/text` | JSON body | ✅ 统一(见 §6.3) |
| `POST` | `/ocr/{preset}/text` | JSON body | ✅ 统一(见 §6.3) |
| `GET` | `/config` | — | ❌ 单独结构 |
| `POST` | `/config` | JSON body(运行时覆盖) | ❌ 单独结构 |
| `DELETE` | `/config` | — | ❌ 单独结构 |
| `GET` | `/health` | — | ❌ `{status, engines}` |
| `GET` | `/engines` | — | ❌ `{engines, configured}` |

**四个 OCR 端点(`/ocr`、`/ocr/{preset}`、`/ocr/text`、`/ocr/{preset}/text`)返回结构完全一致**。只有 `format=text`/`markdown` 时退化为纯文本。`{preset}` 路径参数见 §7.1(命名策略预设)。

---

### 6.2 输入格式

#### 6.2.1 multipart 上传(`POST /ocr`、`POST /ocr/{preset}`)

用于上传本地图片或 PDF 文件。

**表单字段**:

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `file` | File | ✅ | 图片或 PDF 文件 |
| `engine` | string | — | 强制指定引擎(如 `rapidocr`/`siliconflow`) |
| `model` | string | — | 覆盖模型名(仅远程引擎生效) |
| `prompt` | string | — | 覆盖 prompt(仅远程引擎生效) |
| `strategy` | JSON string | — | 临时策略对象(如 `{"retry_mode":"any","max_retries":2}`) |
| `strategy_name` | string | — | 一次性命名预设:`local`/`vl`/`seq*`/`bestof*` |
| `max_pages` | int | — | PDF 页数上限 |
| `dpi` | int | 200 | PDF 渲染 DPI |
| `format` | string | json | 输出格式:`json`/`text`/`markdown` |

```bash
# 上传本地图片,指定引擎和输出格式
curl -s http://localhost:8000/ocr \
  -F "file=@scan.png" \
  -F "engine=siliconflow" \
  -F "format=json"

# 上传 PDF,只处理前 5 页,300 DPI
curl -s http://localhost:8000/ocr \
  -F "file=@report.pdf" \
  -F "max_pages=5" \
  -F "dpi=300"

# 使用命名预设路由:所有引擎择优
curl -s http://localhost:8000/ocr/bestof \
  -F "file=@scan.png" \
  -F "format=json"

# 使用路由 + 覆盖模型
curl -s http://localhost:8000/ocr/siliconflow \
  -F "file=@scan.png" \
  -F "model=Qwen/Qwen2.5-VL-72B" \
  -F "format=text"
```

#### 6.2.2 JSON body(`POST /ocr/text`、`POST /ocr/{preset}/text`)

用于识别 URL 图片、data URI 或 base64 编码的图片字节。请求体为 JSON,
`Content-Type: application/json`。

**图片来源(三选一,只能传一个)**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_url` | string | 本地路径或 `http(s)://` URL |
| `image_b64` | string | 纯 base64 字符串(自动加 `data:` 前缀) |
| `image_data` | string | 完整 data URI,如 `data:image/png;base64,xxxx` |

**通用字段**:

| 字段 | 类型 | 默认 | 说明 |
|------|------|:----:|------|
| `engine` | string | — | 强制引擎 |
| `model` | string | — | 覆盖模型(仅远程引擎) |
| `prompt` | string | — | 覆盖 prompt(仅远程引擎) |
| `strategy` | object | — | 临时策略 JSON 对象 |
| `strategy_name` | string | — | 一次性命名预设 |
| `max_pages` | int | — | PDF 页数上限 |
| `dpi` | int | 200 | PDF 渲染 DPI |
| `format` | string | `json` | 输出格式 |

**示例**:

```bash
# 按 URL 识别(最常用)
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png","engine":"multimodal","format":"json"}'

# URL 无扩展名也能识别(按文件头魔数自动判型)
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/img/u=1234&fm=3074","strategy_name":"vl"}'

# 传 base64 字符串
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"iVBORw0KGgoAAAANSUhEUgAA...","engine":"rapidocr"}'

# 传完整 data URI
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_data":"data:image/png;base64,iVBORw0KGgoAAA...","engine":"rapidocr"}'

# 路由即策略:用 bestof 预设,base64 输入
curl -s http://localhost:8000/ocr/bestof/text \
  -H "Content-Type: application/json" \
  -d '{"image_b64":"iVBORw0KGgoAAA...","format":"json"}'

# 本地文件路径(容器环境挂载共享目录时可用)
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"/data/scan.png","engine":"rapidocr"}'
```

> **注意**:三个图片来源字段只能传一个,传零个或传多个都会返回 HTTP 400
> `{"detail":"provide exactly one of image_url, image_b64, or image_data"}`。

---

### 6.3 输出格式

#### 6.3.1 JSON 格式(`format=json`,默认)

所有四个 OCR 端点返回**完全相同**的结构:

```json
{
  "pages": [
    {
      "text": "永和九年,岁在癸丑,暮春之初...",
      "engine": "rapidocr",
      "model": "rapidocr-onnxruntime",
      "elapsed_ms": 10286,
      "width": 750,
      "height": 1390,
      "region_count": 166,
      "regions": [
        {
          "text": "永和九年岁在癸丑",
          "confidence": 0.97,
          "bbox": {
            "x1": 672,
            "y1": 12,
            "x2": 737,
            "y2": 1370,
            "width": 65,
            "height": 1358
          },
          "engine": "rapidocr"
        }
      ]
    }
  ],
  "text": "永和九年,岁在癸丑...\n\n(多页时用双换行拼接)",
  "engine": "rapidocr",
  "page_count": 1
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `pages` | array | 每页一个对象;单图也是 1 个元素 |
| `pages[].text` | string | 该页识别全文 |
| `pages[].engine` | string | 该页最终使用的引擎名 |
| `pages[].model` | string | 模型名(本地引擎为 onnx 版本号,远程为模型名) |
| `pages[].elapsed_ms` | int | 该页识别耗时(毫秒) |
| `pages[].width` / `height` | int | 页面像素尺寸 |
| `pages[].region_count` | int | 文本区域数量 |
| `pages[].regions` | array | 每个文本区域的详细信息 |
| `pages[].regions[].text` | string | 该区域的文字 |
| `pages[].regions[].confidence` | float | 置信度 0.0–1.0 |
| `pages[].regions[].bbox` | object | 边界框(含 x1/y1/x2/y2/width/height) |
| `pages[].regions[].engine` | string | 识别该区域的引擎 |
| `text` | string | 所有页拼接后的完整文本 |
| `engine` | string | 最终使用的引擎(单页时等于 `pages[0].engine`) |
| `page_count` | int | 页数 |

**多页 PDF 示例**:

```json
{
  "pages": [
    {
      "text": "第一章 引言",
      "engine": "rapidocr",
      "model": "rapidocr-onnxruntime",
      "elapsed_ms": 8500,
      "width": 595,
      "height": 842,
      "region_count": 42,
      "regions": [...]
    },
    {
      "text": "第二章 方法",
      "engine": "rapidocr",
      "model": "rapidocr-onnxruntime",
      "elapsed_ms": 9200,
      "width": 595,
      "height": 842,
      "region_count": 58,
      "regions": [...]
    }
  ],
  "text": "第一章 引言\n\n第二章 方法",
  "engine": "rapidocr",
  "page_count": 2
}
```

#### 6.3.2 text / markdown 格式(`format=text` 或 `format=markdown`)

直接返回拼接后的纯文本(每个页面的 markdown 卡片用双换行拼接),
无 JSON 包裹,`Content-Type: text/plain`:

```text
# 识别结果 - 第 1 页

- **engine**: rapidocr
- **model**: rapidocr-onnxruntime
- **elapsed_ms**: 10286
- **size**: 750 × 1390
- **regions**: 166

## 文本

永和九年,岁在癸丑,暮春之初,会于会稽山阴之兰亭...
```

两种格式内容完全相同,`text` 和 `markdown` 可互换使用。

---

### 6.4 完整案例

#### 案例 1:上传文件,JSON 输出

```bash
curl -s -X POST http://localhost:8000/ocr \
  -F "file=@tests/兰亭序.jpeg" \
  -F "engine=rapidocr" \
  -F "format=json"
```

```json
{
  "pages": [{
    "text": "永和九年岁在癸丑...",
    "engine": "rapidocr",
    "model": "rapidocr-onnxruntime",
    "elapsed_ms": 10286,
    "width": 750,
    "height": 1390,
    "region_count": 166,
    "regions": [
      {"text": "永和九年岁在癸丑", "confidence": 0.97, "bbox": {...}, "engine": "rapidocr"}
    ]
  }],
  "text": "永和九年岁在癸丑...",
  "engine": "rapidocr",
  "page_count": 1
}
```

#### 案例 2:URL 识别,markdown 输出

```bash
curl -s -X POST http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/scan.png",
    "engine": "siliconflow",
    "format": "markdown"
  }'
```

```text
# 识别结果 - 第 1 页

- **engine**: siliconflow
- **model**: PaddlePaddle/PaddleOCR-VL-1.5
- **elapsed_ms**: 8500
- **size**: 800 × 1200
- **regions**: 24

## 文本

永和九年,岁在癸丑,暮春之初...
```

#### 案例 3:base64 识别,JSON 输出

```bash
curl -s -X POST http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{
    "image_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "engine": "rapidocr",
    "format": "json"
  }'
```

#### 案例 4:策略预设路由 + 临时覆盖

```bash
curl -s -X POST http://localhost:8000/ocr/bestof \
  -F "file=@scan.png" \
  -F "model=Qwen/Qwen2.5-VL-72B" \
  -F "format=json"
```

路由参数 `bestof` 指定"所有引擎各跑一次,按 smart 评分选最佳";
`model` 覆盖远程引擎的模型为 Qwen2.5-VL-72B。

#### 案例 5:运行时调整策略 + 按 URL 识别

```bash
# 先调整策略:降低置信度阈值,增大重试次数
curl -s -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"strategy": {"retry_mode": "low_confidence", "min_confidence": 0.6, "max_retries": 2}}'

# 再识别,自动应用新策略
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/blurry.png", "engine": "rapidocr"}'
```

---

### 6.5 运行时配置 `GET/POST/DELETE /config`

不重启改模型或引擎顺序:

```bash
# 查看当前配置(不泄露 key)
curl -s http://localhost:8000/config
```

```json
{
  "engines": [
    {"name": "rapidocr", "enabled": true, "model": "", "base_url": ""},
    {"name": "siliconflow", "enabled": true, "model": "PaddlePaddle/PaddleOCR-VL-1.5", "base_url": "https://api.siliconflow.cn/v1"},
    {"name": "multimodal", "enabled": false}
  ],
  "strategy": {"max_retries": 1, "retry_mode": "no_text", "min_confidence": 0.7},
  "has_api_key": true,
  "overridden": false
}
```

```bash
# 切换模型(不回显 key)
curl -s -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"engines":[{"name":"siliconflow","model":"Qwen/Qwen2.5-VL-72B"}]}'

# 调整策略
curl -s -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"strategy":{"max_retries":2,"retry_mode":"low_confidence","min_confidence":0.8}}'

# 还原到 config.yaml 默认值
curl -s -X DELETE http://localhost:8000/config
```

> `GET /config` 永远只返回 `has_api_key: true/false` 布尔值,绝不返回 key 明文。

### 6.6 辅助端点

```bash
# 健康检查
curl -s http://localhost:8000/health
# {"status":"ok","engines":["rapidocr","siliconflow","multimodal"]}

# 引擎列表
curl -s http://localhost:8000/engines
# {"engines":{"rapidocr":"本地...","siliconflow":"硅基流动...","multimodal":"通用OpenAI兼容..."},"configured":["rapidocr","siliconflow","multimodal"]}
```

### 6.7 异常映射

所有异常统一返回 `{"detail": "错误描述"}` 格式。

| 异常 | HTTP 状态码 | 典型场景 | 响应示例 |
|------|:----:|------|------|
| `InputError` | 400 | 文件不存在、图片损坏、URL 无法下载、base64 解码失败 | `{"detail":"input not found: /tmp/x.png"}` |
| `EngineNotAvailable` | 422 | 缺少依赖库、API key 为空、base_url 未配置 | `{"detail":"siliconflow requires OPENAI_API_KEY"}` |
| `EngineError` | 502 | 引擎调用超时、HTTP 402 余额不足、网络错误 | `{"detail":"HTTP 402","engine":"siliconflow"}` |
| `StrategyError` | 422 | 策略链中所有引擎均失败且无可用结果 | `{"detail":"all engines exhausted"}` |
| 未知 preset | 404 | `/ocr/xxx` 路由既非引擎也非策略预设 | `{"detail":"unknown preset 'xxx'..."}` |
| 格式错误 | 400 | 图片三选一只传一个、format 非法、strategy JSON 非对象 | `{"detail":"provide exactly one of image_url, image_b64, or image_data"}` |

## 7. 引擎使用策略

### 7.1 命名策略预设(strategy presets)

项目把常用引擎组合固化为**命名预设**,配置里固定默认策略,单次请求可临时切换
(一次性,不改动服务端 base config):

**顺序预设**(`seq*`,走 `StrategyEngine`,按引擎顺序尝试,首个命中即返回):

| 预设 | retry_mode | 阅读顺序重排 | 适用场景 |
|------|------------|:----:|----------|
| `local` | `no_text` | — | 仅本地引擎,离线、隐私敏感 |
| `vl` | `no_text` | — | 仅远程 VL 大模型,版面复杂/手写/表格 |
| `seq` | `no_text` | — | 全部启用引擎,按配置顺序回退(**默认**) |
| `seq-any` | `any`(低置信度或窜行) | ✅ | 同 quality,需要整理后的连贯文本 |
| `seq-low_conf` | `low_confidence` | — | 低置信度时自动降级 |
| `seq-line_overlap` | `line_overlap` | — | 窜行(合并/重叠框)时自动降级 |

**最佳策略**(bestof*,走 `BestofEngine`,所有引擎各跑一次,按评分选最佳):

| 预设 | 评分函数 | 适用场景 |
|------|----------|----------|
| `bestof` / `bestof-smart` | 置信度 × 100 − 窜行惩罚 + 文本长度奖赏 + **语义流畅度** | 综合最优,**推荐** |
| `bestof-fastest` | 耗时(elapsed_ms)最低 | 追求速度 |
| `bestof-confidence` | 平均置信度最高 | 追求质量 |
| `bestof-longest` | 文本最长 | 追求完整性 |
| `bestof-fluency` | 语义流畅度(短语密度 + CJK 标点 − 单字碎片惩罚) | 追求"读起来像人话",适合对比本地碎片化结果与 VL 连贯结果 |
| `bestof:<mode>` | 同上任意 mode | 等价于 `bestof-mode`,冒号语法别名 |

**bestof-fluency 评分信号**:

- **短语密度**(上限 +15):每区域平均字符数,`mean_phrase_length`——句子越长越连贯
- **CJK 标点比例**(上限 +5):句子标记(`,。！？、；:()`等)占比——有标点即像自然语言
- **单字碎片惩罚**(上限 −25):`len(p)==1` 的区域计数 × 0.3——166 个单字的 rapidocr 输出会被明显扣减

> 综合分 smart 已集成 fluency:对兰亭序实测,rapidocr 输出 166 个单字/短词(fluency ≈ −23),
> siliconflow 输出完整古文句子(fluency ≈ +15),`bestof-fluency`/`bestof-smart` 均正确选硅基流动。

> `bestof` 比 `seq*` 慢(所有引擎都跑),但能拿到所有候选里最好的结果——
> 适合需要"不管用什么模型,只要识别质量最好"的场景。

**legacy 别名**(保留兼容,与 seq* 完全等价):

| 旧名 | 等价于 |
|------|--------|
| `fallback` | `seq` |
| `quality` | `seq-any` |

**用法**:

```bash
# CLI
python -m jykj_ocr scan.png --strategy-name quality

# 最佳策略:所有引擎各跑一次,选识别质量最优
python -m jykj_ocr scan.png --strategy-name bestof    # seq-any,窜行自动降级
python -m jykj_ocr scan.png --strategy-name bestof      # 所有引擎各跑一次,选最佳
python -m jykj_ocr scan.png --strategy-name bestof-fastest

# Python API
jykj_ocr.ocr("scan.png", strategy_name="seq-any")
jykj_ocr.ocr("scan.png", strategy_name="bestof-smart")

# HTTP —— /ocr(multipart)与 /ocr/text(JSON)均支持
curl -s http://localhost:8000/ocr -F "file=@scan.png" -F "strategy_name=bestof"
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png","strategy_name":"bestof-longest"}'
```

未知名称:CLI 报 argparse 错并列出可选值;HTTP 返回 400
`unknown strategy 'xxx'; choose one of local, vl, seq, seq-any, seq-low_conf, seq-line_overlap, bestof, ...`。

**优先级**:单次请求 `strategy_name` > `POST /config` 运行时覆盖 > `config.yaml`。
预设展开走 `apply_strategy_preset`,返回 deepcopy——输入配置永不被改动。

### 7.2 更多 OCR 引擎接入

本地/远程划分走 `remote_engines()`(内置 siliconflow、multimodal)。新注册引擎
**无需改代码**即被预设识别:`local` 保留它(除非列入远程名单)、`vl` 排除它、
`seq*`/`bestof*`/`fallback`/`quality` 尊重其 `enabled` 标志。要把新厂商归入远程侧:

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

- **CI 基线**:120 个用例,全部离线运行,无真实 API 调用,monkeypatch 模拟引擎返回
- 覆盖:models、config、strategy、engines(multimodal OpenAI 响应解析、rapidocr
  1.x/1.4.x/2.x 返回形态)、策略预设(local/vl/seq*/bestof*、deepcopy 不变性、
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
