# jykj_ocr

多引擎 OCR 服务:同时支持本地 **RapidOCR**(离线、无需 API key)与多模态 OCR 大模型
(硅基流动 / 任意 OpenAI 兼容端点),通过策略层编排调用顺序与重试逻辑,对外提供
CLI、Python API 与 FastAPI HTTP 接口。

```
┌──────────┐   ┌────────────┐   ┌──────────────────────────────────┐
│ CLI / API│──▶│ Strategy   │──▶│ rapidocr (local ONNX, offline)    │
│ / HTTP   │   │ (retry链)  │   │ siliconflow (PaddleOCR-VL-1.5)    │
└──────────┘   └────────────┘   │ multimodal (OpenAI-compatible)   │
                                └──────────────────────────────────┘
```

## 引擎

| 引擎 | 类型 | 需 API key | 默认模型 | 说明 |
|------|------|:----------:|---------|------|
| `rapidocr` | 本地 ONNX | ❌ | — | RapidOCR-onnxruntime,离线可用,中英文 |
| `siliconflow` | 远程多模态 | ✅ | `PaddlePaddle/PaddleOCR-VL-1.5` | 硅基流动,multimodal 的注册别名(同一引擎实例),默认模型/URL 注入于 config.py |
| `multimodal` | 远程多模态 | ✅ | 由 config 指定 | 通用 OpenAI 兼容端点,适配任意平台 |

**引擎别名**(`config.normalise_engine`):`rapid`/`rapid-ocr`/`rapidocr-onnx` → `rapidocr`;
`sf`/`silicon-flow`/`silicon_flow` → `siliconflow`;
`multi`/`openai`/`openai-compat`/`openai-compatible`/`llm` → `multimodal`。

远程引擎统一走 **OpenAI 兼容协议**(`POST /chat/completions`,`messages` 数组 +
`image_url` data-URI content parts),只依赖 `requests`,不引入 `openai` SDK。

## 策略预设

配置文件固定**默认策略**,API/CLI 可按请求**一次性切换**预设(不改动任何配置,
只在本次请求生效)。

**顺序预设**(`seq*`,按引擎顺序尝试,首个命中即返回):

| 预设 | 引擎范围 | retry_mode | 文本重排 |
|------|----------|------------|:--------:|
| `local` | 仅本地(rapidocr 等),远程禁用 | `no_text` | ❌ |
| `vl` | 仅 VL 大模型(siliconflow/multimodal),本地禁用 | `no_text` | ❌ |
| `seq` | 全部启用引擎,按配置顺序回退(默认) | `no_text` | ❌ |
| `seq-any` | 同 seq,但低置信度或窜行即降级 | `any` | ✅ 按坐标重建阅读顺序 |
| `seq-low_conf` | 低置信度时自动降级 | `low_confidence` | ❌ |
| `seq-line_overlap` | 窜行时自动降级 | `line_overlap` | ❌ |

**最佳策略**(`bestof*`,所有引擎各跑一次,按评分选最佳):

| 预设 | 评分函数 |
|------|----------|
| `bestof` / `bestof-smart` | 置信度 − 窜行惩罚 + 文本长度奖赏(综合最优) |
| `bestof-fastest` | 耗时最低 |
| `bestof-confidence` | 平均置信度最高 |
| `bestof-longest` | 文本最长 |
| `bestof-fluency` | 语义流畅度(短语密度 + CJK 标点 − 单字碎片惩罚) |
| `bestof:<mode>` | 等价于 `bestof-mode` |

`bestof` 比 `seq*` 慢(所有引擎都跑),但能拿到所有候选里最好的结果。

**legacy 别名**:`fallback` == `seq` / `quality` == `seq-any`(保留兼容)。

`retry_mode` 可选值:`no_text` / `low_confidence` / `line_overlap` / `any` / `none`;
`output.reorder_lines: true` 可单独开启阅读顺序重排。

**future-proof**:预设按「远程引擎白名单」划分引擎,新接入的引擎(PaddleOCR、
Tesseract、其他云厂商…)无需改预设代码——注册后默认归入本地侧;若新引擎是
远程端点,设 `JYKJ_OCR_REMOTE_ENGINES="a,b"` 把它标为远程即可被 `vl` 选中。

```bash
# CLI
python -m jykj_ocr image.png --strategy-name quality        # seq-any,窜行降级
python -m jykj_ocr image.png --strategy-name bestof         # 所有引擎各跑一次
python -m jykj_ocr image.png --strategy-name bestof-fastest

# HTTP
curl -s http://localhost:8000/ocr -F "file=@image.png" -F "strategy_name=bestof"

# HTTP JSON
curl -s http://localhost:8000/ocr/text -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png","strategy_name":"bestof-longest"}'

# Python API
results = jykj_ocr.ocr("image.png", strategy_name="bestof-smart")
```

优先级:**单次请求的 `strategy_name` > `POST /config` 运行时覆盖 > config.yaml**。
未知预设名返回 HTTP 400 并列出有效值。

## 安装

```bash
# 1. 创建虚拟环境(Python 3.10+)
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate      # Linux/macOS

# 2. 安装依赖(含测试所需 httpx;rapidocr-onnxruntime 已包含)
pip install -r requirements.txt

# 3. 以 editable 模式安装本包(可选,提供 jykj-ocr 命令)
pip install -e .
```

> `requirements.txt` 自包含,`pip install -r requirements.txt` 后即可通过
> `pytest tests -q` 运行全部测试。

### rapidocr 引擎的 OpenCV 依赖(headless)

`rapidocr-onnxruntime` 的传递依赖是 GUI 版 `opencv-python`,它在 **Linux**
上 import 时需要系统库 `libGL.so.1`/`libglib2.0`(slim 镜像与最小化服务器
默认没有)。缺失时 `import cv2` 抛 ImportError,被引擎层误报为
`"RapidOCR is not installed"`(HTTP 422)——即使 pip list 显示已安装。

本项目在 `requirements.txt` 中**先装 `opencv-python-headless`** 顶替 GUI 版
(功能对 OCR 完全等价,不链接 libGL),因此**无需任何 apt 系统包**,
Windows/Linux/Docker 统一一条 `pip install -r requirements.txt` 即可。

已经踩过这个坑的存量环境(如已装 GUI 版 opencv-python),二选一修复:

```bash
# 方案 A(推荐,与 requirements.txt 一致):换 headless,注意必须先卸载 GUI 版
pip uninstall -y opencv-python && pip install "opencv-python-headless>=4.9,<6"

# 方案 B:保留 GUI 版,补系统库(Debian / Ubuntu)
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

排查真实原因(逐层 import 并打印原始错误):

```bash
.venv/bin/python scripts/diag_rapidocr_import.py
```

## 配置

API key 只通过**环境变量**提供,绝不写入配置文件或代码:

```bash
cp .env.example .env              # 复制模板,填入真实 key
# .env 已在 .gitignore 中,不会被提交
```

远程引擎统一用一对环境变量,切换平台只需改这两个值:

| 变量 | 用途 |
|------|------|
| `OPENAI_API_KEY` | 通用 API key,适配任意 OpenAI 兼容平台 |
| `OPENAI_BASE_URL` | 端点 URL(硅基流动 / 阿里云百炼 / 智谱 / 本地 vLLM 等) |
| `SILICONFLOW_API_KEY` | 可选,仅当 siliconflow 需要与 `OPENAI_*` 不同的 key 时覆盖 |

**配置优先级**(高 → 低):显式参数 > 环境变量(`JYKJ_OCR_*`、`JYKJ_OCR_<NAME>_API_KEY`、
`<NAME>_API_KEY`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`)> `config/config.yaml` > 内置默认值。

## CLI

```bash
# 列出可用引擎
python -m jykj_ocr --list-engines

# 识别图片(指定引擎)
python -m jykj_ocr image.png --engine siliconflow --format json
python -m jykj_ocr doc.pdf   --engine rapidocr  --format markdown -o out.md

# 不指定引擎 → 走 config.yaml 中的策略链(rapidocr → siliconflow)
python -m jykj_ocr image.png

# 启动 HTTP 服务
python -m jykj_ocr serve --port 8000
```

| 参数 | 说明 |
|------|------|
| `source` | 图片/PDF 路径或 `http(s)://` URL |
| `-c` / `--config` | 配置文件路径(默认 `config/config.yaml`) |
| `--engine` | 强制使用某引擎,忽略策略链 |
| `--strategy-name` | 一次性策略预设:`local` \| `vl` \| `seq*` \| `bestof*` |
| `--format` | `text` \| `markdown` \| `json`(默认 `text`) |
| `-o` / `--output` | 输出文件;缺省打印到 stdout |
| `--max-pages` | PDF 最多处理页数 |
| `--dpi` | PDF 渲染 DPI(默认 200) |
| `serve` | 启动 FastAPI 服务(`--host` / `--port`) |

## Python API

```python
import jykj_ocr

# 指定引擎
results = jykj_ocr.ocr("image.png", engine="siliconflow")
print(results[0].text, results[0].model)

# 走策略链(配置文件中的引擎顺序)
results = jykj_ocr.ocr("doc.pdf")

# 只拿拼接好的 markdown 文本
text = jykj_ocr.ocr_to_text("image.png", engine="rapidocr")
```

`ocr(source, *, engine=None, config=None, config_path=None, max_pages=None, dpi=200, retries=1, strategy_name=None)`
→ `List[OCRResult]`,每页一个。`source` 必须是文件路径或 `http(s)://` URL
(不接受裸 bytes)。`strategy_name` 一次性应用命名预设(`local`/`vl`/`seq*`/`bestof*`),
只作用于本次调用的配置副本。

## HTTP API

```bash
python -m jykj_ocr serve          # 或 JYKJ_OCR_PORT=9000 python -m jykj_ocr serve
```

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/engines` | 可用引擎列表 + 当前配置的引擎顺序 |
| `GET` | `/presets` | 全部命名策略预设的元数据(评分模式、重试判定、引擎范围) |
| `GET` | `/config` | 当前生效配置(**不返回 API key 明文**,仅 `has_api_key` 布尔) |
| `POST` | `/config` | 运行时覆盖配置(引擎/模型/策略),无需重启 |
| `DELETE` | `/config` | 清除运行时覆盖,回到配置文件状态 |
| `POST` | `/ocr` | 上传图片/PDF 识别(multipart) |
| `POST` | `/ocr/text` | 按图片 URL 识别(JSON) |
| `POST` | `/ocr/{preset}` | 路由即策略:多部件上传(preset=引擎名 或 策略预设名) |
| `POST` | `/ocr/{preset}/text` | 路由即策略:按图片 URL(JSON) |

四个 OCR 端点返回结构完全一致:`{pages, text, engine, page_count}`;`format=text/markdown` 时退化为纯文本。

**POST /ocr** 示例:

```bash
curl -s http://localhost:8000/ocr \
  -F "file=@image.png" \
  -F "engine=siliconflow" \
  -F "format=json" | python -m json.tool
```

form 字段:`file`(必填)、`engine`、`model`、`prompt`、`strategy`(JSON 字符串)、
`strategy_name`(`local`/`vl`/`seq*`/`bestof*`,一次性预设,见策略预设章节)、
`max_pages`、`dpi`、`format`(`json`/`text`/`markdown`)。`model`/`prompt` 仅对远程引擎
(`siliconflow`/`multimodal`,或 `JYKJ_OCR_REMOTE_ENGINES` 标定的引擎)生效,
本地 `rapidocr` 不受影响。

**POST /ocr/text** 示例:

```bash
curl -s http://localhost:8000/ocr/text \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png","engine":"multimodal"}'
```

**运行时覆盖**(改模型不重启):

```bash
curl -s -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"engines":[{"name":"siliconflow","model":"Qwen/Qwen2.5-VL-72B"}]}'
```

**异常映射**:`InputError → 400`、`EngineNotAvailable → 422`、`EngineError → 502`、
`StrategyError → 422`。

## Docker

```bash
# 单服务
docker build -t jykj_ocr .
docker run --rm -p 8000:8000 --env-file .env jykj_ocr

# compose(含 healthcheck、模型权重持久化卷)
docker compose up --build

# 容器内一次性任务
docker run --rm --env-file .env -v "$PWD:/data" jykj_ocr \
    python -m jykj_ocr /data/image.png --engine siliconflow
```

镜像 `python:3.11-slim`、非 root 用户、含 `HEALTHCHECK`;配置与端口通过
`JYKJ_OCR_CONFIG` / `JYKJ_OCR_PORT` 注入。

## 测试

```bash
.venv/Scripts/python -m pytest tests -q     # 全部离线,无真实 API 调用
```

137 个用例覆盖:models(边界框/文本区域/置信度保留)、config(别名归一化/YAML/环境变量
优先级)、strategy(重试链/谓词)、engines(multimodal 的 OpenAI 响应解析 / rapidocr 的
1.x 3-tuple 与 1.4.x 2-tuple 返回形态)。

端到端接口测试(需联网与 API key,非 CI 基线):

```bash
export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://api.siliconflow.cn/v1
.venv/Scripts/python scripts/test_interfaces.py
```

## 项目布局

```
src/jykj_ocr/
├── __init__.py            # 顶层 API:ocr() / ocr_to_text()
├── config.py              # Config / EngineConfig / load_config / normalise_engine
├── models.py              # Point / BoundingBox / TextRegion / OCRResult
├── strategy.py            # StrategyEngine / BestofEngine / 重试谓词
├── engine/
│   ├── __init__.py        # 惰性注册(lazy import,不引入 PIL/rapidocr/openai)
│   ├── base.py            # BaseEngine / PageImage / EngineNotAvailable / registry
│   ├── inputs.py          # 图片/PDF/URL → PageImage
│   └── registry.py        # build_engine / build_pipeline / apply_strategy_preset / remote_engines / _SEQ_PRESETS
├── engines/
│   ├── rapidocr_engine.py     # RapidOCREngine(适配 1.x/1.4.x/2.x 返回形态)
│   └── multimodal_engine.py   # MultimodalEngine(OpenAI 兼容;同时注册 siliconflow 别名工厂)
├── cli.py               # argparse CLI
└── server.py            # FastAPI /ocr /ocr/text /config /engines /health
config/config.yaml       # 默认引擎 + 策略
tests/                   # pytest,120 passed
scripts/                 # 诊断与接口测试脚本
Dockerfile / docker-compose.yml
requirements.txt / pyproject.toml
.env.example            # 环境变量模板(真实 .env 已 gitignore)
```

## 架构要点

- **惰性注册**:`import jykj_ocr` 不加载 PIL/rapidocr/openai,引擎按需 import。
- **策略引擎**:`StrategyEngine` 按顺序尝试(首个命中即返回),`should_retry_no_text` /
  `should_retry_low_confidence` / `should_retry_line_overlap` 判定是否重试与切换;
  `BestofEngine` 所有引擎各跑一次,按 smart/fastest/highest_confidence/longest 评分选最佳。
- **命名预设**:`apply_strategy_preset` 把 `local`/`vl`/`seq*`/`bestof*`(含 legacy `fallback`/`quality` 别名)展开为
  一次性配置副本(deepcopy,输入 config 永不被改动);远程/本地划分走
  `remote_engines()` 白名单,新引擎零改动接入。
- **`TextRegion.from_parts`**:用 `_UNSET` 哨兵区分"调用方没传 confidence"与
  "真的传了 1.0"——引擎返回 `score: 0.88` 不会被静默抹平为 1.0。
- **`_PydanticBase`**:pydantic 可选;缺失时回退到 stdlib 轻量替代,离线容器可运行。
- **`RuntimeConfig`**:线程安全,`POST /config` 的运行时覆盖在 `snapshot()` 时与
  base config 合并,不返回 API key 明文(只暴露 `has_api_key` 布尔)。
- **rapidocr 1.4.x 适配**:`rapidocr-onnxruntime` 1.4.x 返回 `(results, elapsed)`
  2-tuple(`results` = `[box, text, score]` 三元组列表),与 1.x 的
  `(boxes, txts, scores)` 3-tuple、2.x 的 dict 均不同;`_run()` 用
  `_looks_like_results_list()` 检测并分别解析。

更详细的架构图见 `docs/architecture.mmd` / `docs/architecture.html`,
用户手册见 `docs/manual.md`。

## 许可

MIT
