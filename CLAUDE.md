# CLAUDE.md

jykj_ocr 是一个多引擎 OCR Python 项目:同时支持本地 RapidOCR(离线、
无需 API key)与多模态 OCR 大模型(硅基流动 / OpenAI 兼容端点),通过
策略层编排调用顺序与重试逻辑,对外提供 CLI 与 FastAPI HTTP 接口。

## 技术栈

- **语言/框架**:Python 3.10+,pydantic v2,FastAPI,Pillow。
- **引擎**:
  - `rapidocr` — RapidOCR ONNX(离线),适配 1.4.x `(results, elapsed)`、1.x tuple 与 2.x dict 返回形态。
  - `multimodal` — 通用 OpenAI 兼容 `/chat/completions`,只依赖 `requests`(不引入 `openai` SDK)。
  - `siliconflow` — `multimodal` 的注册别名(同一 MultimodalEngine 实例,无独立子类文件),默认模型 `PaddlePaddle/PaddleOCR-VL-1.5`、base URL `https://api.siliconflow.cn/v1`(注入于 `config.py`)。
- **引擎别名**:rapid/rapid-ocr/rapidocr-onnx → rapidocr;sf/silicon-flow/silicon_flow → siliconflow;
  multi/openai/openai-compat/openai-compatible/llm → multimodal(`config.normalise_engine`)。
- **Docker**:`python:3.11-slim` 非 root,含 `HEALTHCHECK`,通过 `JYKJ_OCR_CONFIG` / `JYKJ_OCR_PORT` 注入配置。

## 项目布局

```
.
├── src/jykj_ocr/
│   ├── __init__.py            # 顶层 API:ocr() / ocr_to_text()
│   ├── config.py              # Config / EngineConfig / from_mapping / load_config / normalise_engine
│   ├── models.py              # Point / BoundingBox / TextRegion / OCRResult
│   ├── strategy.py            # StrategyEngine / 重试谓词(no_text/low_confidence/line_overlap)/combine_predicates
│   ├── engine/
│   │   ├── __init__.py        # 惰性注册(lazy import,不引入 PIL/rapidocr/openai)
│   │   ├── base.py            # BaseEngine / PageImage / EngineNotAvailable / registry
│   │   ├── inputs.py          # 图片/PDF/URL → PageImage(PDF 尝试 pymupdf → pdf2image → Pillow)
│   │   └── registry.py        # build_engine / build_pipeline / apply_strategy_preset / remote_engines / resolve_retry_check
│   ├── engines/
│   │   ├── rapidocr_engine.py     # RapidOCREngine
│   │   └── multimodal_engine.py   # MultimodalEngine(OpenAI 兼容;同时注册 siliconflow 别名工厂)
│   ├── cli.py               # argparse CLI(`jykj-ocr` / `serve` / `--list-engines` / `--engine` / `--strategy-name` / `--format`)
│   └── server.py            # FastAPI /ocr /ocr/text /config /engines /health
├── config/config.yaml       # 默认引擎+策略,api_key 有意省略(走环境变量)
├── tests/                   # pytest,50 个用例(50 passed)
├── Dockerfile / docker-compose.yml
├── requirements.txt / pyproject.toml
├── .env                     # 存放真实 API key(已 gitignore,勿提交)
├── .env.example             # 占位符
└── .gitignore
```

## 命令(全部使用项目虚拟环境)

```bash
# 安装(仅外部依赖;rapidocr_onnxruntime 按需要单独装)
.venv/Scripts/python -m pip install -r requirements.txt

# 运行测试(目前 50 passed)
.venv/Scripts/python -m pytest tests -q

# CLI 识别
.venv/Scripts/python -m jykj_ocr ocr -i image.png --engine siliconflow --format json
.venv/Scripts/python -m jykj_ocr --list-engines

# 启动 HTTP 服务(端口默认 8000,可由 JYKJ_OCR_PORT 覆盖)
.venv/Scripts/python -m jykj_ocr serve
# 或
JYKJ_OCR_PORT=8000 .venv/Scripts/python -m uvicorn jykj_ocr.server:app --host 0.0.0.0

# Docker
docker compose up -d
```

## API 概要

- **FastAPI**:
  - `GET /health`,`GET /engines`,`GET /config`,`POST /config`(运行时覆盖模型/引擎/策略),`DELETE /config`
  - `POST /ocr`(multipart,文件 + 可选 `engine`/`model`/`prompt`/`strategy`/`strategy_name`/`max_pages`/`dpi`/`format`)
  - `POST /ocr/text`(image_url 文本请求,body 支持 `strategy_name`)
- **Python API**:`jykj_ocr.ocr(source, engine=None, config=None, config_path=None, max_pages=None, dpi=200, retries=1, strategy_name=None) -> List[OCRResult]`。
- **策略预设**(`strategy_name`,一次性):`local` 仅本地引擎 / `vl` 仅远程 VL 引擎 /
  `fallback` 按配置顺序回退(默认) / `quality` 回退+窜行降级(retry_mode `any`)+阅读顺序重排
  (`output.reorder_lines`)。`apply_strategy_preset` 返回 deepcopy,输入 config 不被改动;
  未知名称 CLI 报 argparse 错、HTTP 返回 400。远程/本地划分走 `remote_engines()`
  (内置 siliconflow/multimodal + 环境变量 `JYKJ_OCR_REMOTE_ENGINES="a,b"`),新引擎默认归本地侧。
- **retry_mode**:`no_text`(默认)/ `low_confidence` / `line_overlap`(无文字或窜行) /
  `any`(低置信度或窜行任一) / `none`。窜行检测在 `models.detect_line_overlap`
  (超长宽比合并框 + 双轴重叠框),重排在 `models.rebuild_text_from_regions`。
- **配置优先级**:显式参数 > 环境变量(`JYKJ_OCR_*`, `JYKJ_OCR_<NAME>_API_KEY`, `<NAME>_API_KEY`, `OPENAI_API_KEY`;base URL 同样支持 `OPENAI_BASE_URL`)> config.yaml > 默认值。远程引擎统一走 OpenAI 兼容协议:设一对 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 即可指向任意平台,siliconflow 引擎不设这对变量时仍用内置默认 URL。

## 架构要点

- **引擎注册表**:惰性 import,`import jykj_ocr` 不会加载 PIL/rapidocr/openai。
- **策略引擎**:按顺序尝试每个引擎,支持 `should_retry_no_text` / `should_retry_low_confidence` /
  `should_retry_line_overlap` 判定是否重试;`combine_predicates` 组合出 `any` 模式。
- **命名预设**:`apply_strategy_preset`(engine/registry.py)把 preset 展开为一次性配置副本;
  `build_pipeline` **不会**重新应用 preset(`strategy.name` 仅作记录)。
- **`TextRegion.from_parts`**:用 `_UNSET` 哨兵区分“调用方没传 confidence”与“真的传了 1.0”——
  引擎返回的 `score: 0.88` 不会被静默抹平为 1.0。
- **`_PydanticBase`**:pydantic 可选;缺失时回退到 stdlib 轻量替代,保持离线容器可运行。
- **`remote_engines()`**:内置 siliconflow/multimodal + `JYKJ_OCR_REMOTE_ENGINES` 环境变量追加;
  只有名单内的引擎响应 `model` / `prompt` 覆盖并被 `vl` 预设选中,其余引擎默认视为本地。
- **`RuntimeConfig`**:线程安全,`POST /config` 的运行时覆盖在 `snapshot()` 时与 base config 合并,
  不返回 API key 明文(只暴露 `has_api_key` 布尔)。
- **异常映射**:`InputError → 400`,`EngineNotAvailable → 422`,`EngineError → 502`,`StrategyError → 422`。

## 引擎实测状态(均通过)

三引擎已用 `tests/兰亭序.jpeg`(750×1390 中文古文)实测成功:

- `rapidocr` — 166 个区域,置信度 0.97+,离线可用。1.4.x 返回 `(results, elapsed)`,`_run()` 通过 `_looks_like_results_list()` 区分真实结果与计时数据(否则会把浮点计时误读为文本)。
- `siliconflow` — `PaddlePaddle/PaddleOCR-VL-1.5`,HTTP 200,返回完整全文。
- `multimodal` — 用 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 指向硅基流动同样跑通,验证了"统一 OpenAI 兼容端点"的可行性。

## 维护提示

1. 改 `engine/registry.py` 时注意 `from . import X as base`,不要写 `from . import engine as engine_pkg`
   (registry 本身在 engine 包内,`engine` 是自身)。
2. 改 `BoundingBox` / `TextRegion` 测试时用**关键字参数**(pydantic v2 禁止位置参数构造)。
3. 不要往 repo 提交真实 API key;`.env` 已 gitignore,新环境用 `.env.example` 起手。
4. 加新引擎:实现 `BaseEngine` 子类 + `_recognise_impl` + `_wrap`,用 `@register("name")` 装饰工厂函数;
   若要保留惰性 import,在 `engine/__init__.py` 里 `register_lazy` 即可。
5. 改 API 契约前跑一遍 `pytest tests -q`;当前 50 passed 是基线。
6. `engines_from_config` 不带显式 names 时只用 **enabled** 引擎(尊重 `enabled: false`);
   加新引擎后跑一遍预设测试确认 `local`/`vl` 归类正确(远程名单外的都进 local)。
