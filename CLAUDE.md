# CLAUDE.md

jykj_ocr 是一个多引擎 OCR Python 项目:同时支持本地 RapidOCR(离线、
无需 API key)与多模态 OCR 大模型(硅基流动 / OpenAI 兼容端点),通过
策略层编排调用顺序与重试逻辑,对外提供 CLI 与 FastAPI HTTP 接口。

## 技术栈

- **语言/框架**:Python 3.10+,pydantic v2,FastAPI,Pillow。
- **引擎**:
  - `rapidocr` — RapidOCR ONNX(离线),适配 1.4.x `(results, elapsed)`、1.x tuple 与 2.x dict 返回形态。
  - `multimodal` — 通用 OpenAI 兼容 `/chat/completions`,只依赖 `requests`(不引入 `openai` SDK)。**一个类型、无限实例**:config 里可写任意多条 `multimodal`,每条由 `(base_url, model, api_key)` 区分并各自实例化。
- **每实例环境变量(序号变量)**:`EngineConfig.instance` 是「同类型里的第 N 条」(1-based),
  由 `from_mapping` 里的 `_assign_instances()` **在 `_dedupe_engines()` 之前**按书写顺序编号,
  各类型独立计数(rapidocr 不占 multimodal 的号)。三个远程字段都走同一套序号模板
  (`config.py` 的 `_instance_env_names()`):
  `base_url` → 条目值 → `OPENAI_BASE_URL_<N>` → `OPENAI_BASE_URL` → `JYKJ_OCR_<NAME>_BASE_URL`;
  `api_key` → 条目值 → `JYKJ_OCR_<NAME>_<N>_API_KEY` → `<NAME>_<N>_API_KEY`(仅当 instance 已设)→
  `JYKJ_OCR_<NAME>_API_KEY` → `<NAME>_API_KEY` → `OPENAI_API_KEY`;
  `model` → 条目值 → `JYKJ_OCR_<NAME>_<N>_MODEL` → `JYKJ_OCR_<NAME>_MODEL`。
  不带序号的变量被同类型所有条目**共享**——这才是「同平台不同账号」和「多平台」
  共用一把 key、另一个平台 401 的根因。整条 entry(端点 + 模型 + key)都可以只靠
  序号环境变量描述,yaml 里一个字段都不用写;同平台不同账号也一样按序号区分。
  yaml 里可显式写 `instance: 1` 钉住序号(未钉住的条目自动跳过该号),防止条目增删
  导致 key 漂移。`dedupe_key()` 仍是不含 instance 的 6-tuple——序号相同但 key 不同的
  两条不会被合并。
- **引擎别名**:rapid/rapid-ocr/rapidocr-onnx → rapidocr;multi/openai/openai-compat/openai-compatible/llm → multimodal;
  sf/silicon-flow/silicon_flow/siliconflow → multimodal。**只有两个引擎类型**
  (`rapidocr` / `multimodal`)——平台不是引擎,是条目里的 `base_url` + `model`
  组合。
- **无平台默认值**:三个 `resolved_*` 都不再有任何厂商默认。`base_url` 留空且
  序号变量与共享的 `OPENAI_BASE_URL` 都没设 → `EngineNotAvailable`(不做回退:把 A 平台的
  key 发给 B 平台只会得到说不清原因的 HTTP 401);`model` 留空退化为裸 ID
  `PaddleOCR-VL-1.5`,其他平台请显式写全(模力方舟用裸 ID,硅基流动要带厂商前缀)。
  两处 `EngineNotAvailable` 的报错文案都列出该条目实际探测过的变量名
  (`config.base_url_env_names()` / `config.api_key_env_names()`),直接告诉运维该 export 什么。
- **多实例去重**:`dedupe_key()` = `(resolved_name, resolved_base_url, resolved_model, resolved_api_key, lang, prompt)`,`from_mapping` 解析时折叠完全相同的条目(第一条胜出);同厂商不同模型、不同厂商、同厂商不同账号三种搭配都各自实例化。
- **Docker**:`python:3.11-slim` 非 root,含 `HEALTHCHECK`,通过 `JYKJ_OCR_CONFIG` / `JYKJ_OCR_PORT` 注入配置。

## 项目布局

```
.
├── src/jykj_ocr/
│   ├── __init__.py            # 顶层 API:ocr() / ocr_to_text()
│   ├── config.py              # Config / EngineConfig / from_mapping / load_config / normalise_engine
│   ├── models.py              # Point / BoundingBox / TextRegion / OCRResult
│   ├── strategy.py            # StrategyEngine / BestofEngine / 重试谓词(no_text/low_confidence/line_overlap)/combine_predicates
│   ├── engine/
│   │   ├── __init__.py        # 惰性注册(lazy import,不引入 PIL/rapidocr/openai)
│   │   ├── base.py            # BaseEngine / PageImage / EngineNotAvailable / registry
│   │   ├── inputs.py          # 图片/PDF/URL → PageImage(PDF 尝试 pymupdf → pdf2image → Pillow)
│   │   └── registry.py        # build_engine / build_pipeline / apply_strategy_preset / remote_engines / describe_presets / _SEQ_PRESETS
│   ├── engines/
│   │   ├── rapidocr_engine.py     # RapidOCREngine
│   │   └── multimodal_engine.py   # MultimodalEngine(OpenAI 兼容,唯一远程类型)
│   ├── cli.py               # argparse CLI(`jykj-ocr` / `serve` / `--list-engines` / `--engine` / `--strategy-name` / `--format`)
│   └── server.py            # FastAPI /ocr /ocr/text /ocr/{preset} /ocr/{preset}/text /config /engines /presets /health
├── config/config.yaml       # 默认引擎+策略,api_key 有意省略(走环境变量);多实例示例已注释
│   ├── config.local.yaml    # 示例 1:只本地 rapidocr(零凭据)
│   ├── config.vl.yaml       # 示例 2:只远程 VL(硅基流动 + 模力方舟)
│   ├── config.seq.yaml      # 示例 3:local + 1 远程兜底
│   └── config.bestof.yaml   # 示例 4:local + 2 远程,bestof_mode: smart
├── tests/                   # pytest,237 个用例(237 passed)
├── Dockerfile / docker-compose.yml
├── requirements.txt / pyproject.toml
├── .env.example             # 占位符模板(真实 key 走 export / Docker -e,本仓库不保留 .env)
└── .gitignore
```

## 命令(全部使用项目虚拟环境)

```bash
# 安装(仅外部依赖;rapidocr_onnxruntime 按需要单独装)
.venv/Scripts/python -m pip install -r requirements.txt

# 运行测试(目前 237 passed)
.venv/Scripts/python -m pytest tests -q

# CLI 识别(source 是位置参数,没有 ocr 子命令,也没有 -i)
.venv/Scripts/python -m jykj_ocr image.png --engine multimodal --format json
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
  - `GET /health`,`GET /engines`,`GET /presets`(全部命名预设的元数据),`GET /config`,`POST /config`(运行时覆盖模型/引擎/策略),`DELETE /config`
  - `POST /ocr`(multipart,文件 + 可选 `engine`/`model`/`prompt`/`strategy`/`strategy_name`/`max_pages`/`dpi`/`format`)
  - `POST /ocr/{preset}`(multipart,路由即策略)
  - `POST /ocr/text`(JSON body,图片来源三选一:`image_url`(路径/URL) / `image_b64`(base64 字符串) / `image_data`(完整 data URI))
  - `POST /ocr/{preset}/text`(同 `/ocr/text`,路由参数指定预设)
  - 所有 OCR 端点输出结构统一:`{pages:[{text,engine,model,elapsed_ms,width,height,region_count,regions}], text, engine, page_count}`;`format=text/markdown` 时退化为纯文本
- **Python API**:`jykj_ocr.ocr(source, engine=None, config=None, config_path=None, max_pages=None, dpi=200, retries=1, strategy_name=None) -> List[OCRResult]`。
- **策略预设**(`strategy_name`,一次性):
  - 顺序预设 `seq*`(走 `StrategyEngine`,首个命中即返回):
    `seq`(retry=no_text)/ `seq-any`(retry=any,reorder=on)= quality /
    `seq-low_conf`(retry=low_confidence)/ `seq-line_overlap`(retry=line_overlap)。
    `local` 仅本地引擎 / `vl` 仅 1 条远程 VL 引擎(config 里第一条已启用的,
    其余远程禁用——多远程进重试链会让返回的模型随 retry 结果漂移,不可预测)
    (同 `StrategyEngine`,改 enabled 标志)。
  - `cascade*` 家族:同 `StrategyEngine` 但 `max_retries=0`——被 reject 的尝试立刻降级
    下一引擎,不重试同一引擎。`cascade`(= seq 语义)/ `cascade-low_conf` /
    `cascade-line_overlap`。
  - 最佳预设 `bestof*`(走 `BestofEngine`,所有引擎各跑一次,按评分选最佳):
    `bestof`/`bestof-smart`(置信度−窜行惩罚+文本长度奖赏+语义流畅度)/
    `bestof-fastest`(elapsed_ms 最低)/`bestof-confidence`(平均置信度最高)/
    `bestof-longest`(文本最长)/`bestof-fluency`(短语密度+CJK 标点−单字碎片惩罚)/
    `bestof:<mode>`(语法别名)。
  - legacy 别名:`fallback` == `seq` / `quality` == `seq-any`(保留兼容)。
  - `apply_strategy_preset` 返回 deepcopy,输入 config 不被改动;未知名称 CLI 报
    argparse 错、HTTP 返回 400。远程/本地划分走 `remote_engines()`(内置
    multimodal + 环境变量 `JYKJ_OCR_REMOTE_ENGINES="a,b"`),新引擎默认归
    本地侧。
  - `build_pipeline` 检测到 `strategy["bestof_mode"]` 时组装 `BestofEngine`;
    `engine_name` 强制单个引擎时绕过 bestof,仍走 `StrategyEngine`。
- **retry_mode**:`no_text`(默认)/ `low_confidence` / `line_overlap`(无文字或窜行) /
  `any`(低置信度或窜行任一) / `none`。窜行检测在 `models.detect_line_overlap`
  (超长宽比合并框 + 双轴重叠框),重排在 `models.rebuild_text_from_regions`。
- **配置优先级**:**显式参数 > config.yaml > 环境变量 > 默认值**。
  `config.py` 的三个 `resolved_*` 都是「yaml 里有值就用 yaml,留空才回退环境变量」,
  且都优先序号变量、再落共享变量:
  `base_url` 走 yaml → `OPENAI_BASE_URL_<N>` → `OPENAI_BASE_URL` → `JYKJ_OCR_<NAME>_BASE_URL`;
  `model` 走 yaml → `JYKJ_OCR_<NAME>_<N>_MODEL` → `JYKJ_OCR_<NAME>_MODEL`;
  `api_key` 走 yaml → `JYKJ_OCR_<NAME>_<N>_API_KEY` / `<NAME>_<N>_API_KEY`(仅当
  instance 已设)→ `JYKJ_OCR_<NAME>_API_KEY` → `<NAME>_API_KEY` → `OPENAI_API_KEY`
  (见「每实例环境变量」;`<N>` 是 `EngineConfig.instance`,由 `_assign_instances` 编号)。
  **环境变量不会覆盖 yaml 里已写的值**——`config.yaml` 一旦写了 `base_url`,
  `OPENAI_BASE_URL` 就被忽略。想让环境变量接管平台,把 yaml 里的
  `base_url` 留空(当前 `config/config.yaml` 就是这么配的)。
  远程引擎统一走 OpenAI 兼容协议:设一对 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 即可指向任意平台。
  **没有厂商默认值**——base URL 解析不出来就 `EngineNotAvailable`(fail-fast);
  静默回退到某个平台会把 A 平台的 key 发给 B,只会表现为一个说不清原因的 HTTP 401。
  环境变量只在进程启动时读入,改完必须重启服务才生效。

## 架构要点

- **引擎注册表**:惰性 import,`import jykj_ocr` 不会加载 PIL/rapidocr/openai。
- **策略引擎**:按顺序尝试每个引擎,支持 `should_retry_no_text` / `should_retry_low_confidence` /
  `should_retry_line_overlap` 判定是否重试;`combine_predicates` 组合出 `any` 模式。
- **命名预设**:`apply_strategy_preset`(engine/registry.py)把 preset 展开为一次性配置副本;
  `build_pipeline` **不会**重新应用 preset(`strategy.name` 仅作记录)。
- **yaml 里的 `strategy.name` 是文档字段,不生效**——实测 `name: local` 仍会加载远程引擎,
  `name: bestof` 仍组装 `StrategyEngine`(`test_build_pipeline_does_not_reapply_preset` 锁死了
  这一行为)。真正生效的只有 `bestof_mode`(单独触发 `BestofEngine`)、`max_retries`、
  `retry_mode`、`min_confidence`;"只用本地/只用远程"靠的是条目里的 `enabled: false`。
  预设交给接口(`--strategy-name` / `strategy_name` / `POST /ocr/{preset}`),示例配置
  `config/config.{local,vl,seq,bestof}.yaml` 因此都不再写 `name`。
- **无平台默认值**;`JYKJ_OCR_SILICONFLOW_*` / `SILICONFLOW_API_KEY` 也是死变量——
  `resolved_name` normalise 后恒为 `MULTIMODAL`,所以只有 `JYKJ_OCR_MULTIMODAL_*` 会被读取。
- **`TextRegion.from_parts`**:用 `_UNSET` 哨兵区分“调用方没传 confidence”与“真的传了 1.0”——
  引擎返回的 `score: 0.88` 不会被静默抹平为 1.0。
- **`_PydanticBase`**:pydantic 可选;缺失时回退到 stdlib 轻量替代,保持离线容器可运行。
- **`remote_engines()`**:内置 multimodal + `JYKJ_OCR_REMOTE_ENGINES` 环境变量追加;
  只有名单内的引擎响应 `model` / `prompt` 覆盖并被 `vl` 预设选中,其余引擎默认视为本地。
- **`RuntimeConfig`**:线程安全,`POST /config` 的运行时覆盖在 `snapshot()` 时与 base config 合并,
  不返回 API key 明文(只暴露 `has_api_key` 布尔)。
- **离线引擎不显示凭证**:`_engine_view`(即 `GET /config`)与 `GET /engines` 的 `configured`
  对本地引擎(判据 `normalise_engine(name) in remote_engines()` 取反,见 `_is_remote`)把
  `model`/`base_url` 置 `""`、`has_api_key` 置 `False`。因为 `resolved_base_url` /
  `resolved_api_key` 即便对 rapidocr 也会回退到共享的 `OPENAI_BASE_URL` / `OPENAI_API_KEY`,
  原样回显会谎报「本地引擎也在打远程接口」。字段仍保留(值置空),不按引擎类型删键——
  `test_local_engine_shows_no_credentials` / `test_engines_endpoint_blank_for_local_engine`
  锁死了该行为。
- **异常映射**:`InputError → 400`,`EngineNotAvailable → 422`,`EngineError → 502`,`StrategyError → 422`。

## 引擎实测状态(均通过)

两引擎已用 `tests/兰亭序.jpeg`(750×1390 中文古文)实测成功:

- `rapidocr` — 166 个区域,置信度 0.97+,离线可用。1.4.x 返回 `(results, elapsed)`,`_run()` 通过 `_looks_like_results_list()` 区分真实结果与计时数据(否则会把浮点计时误读为文本)。
- `multimodal` — 用 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 指向任意平台均可跑通,
  验证了"统一 OpenAI 兼容端点"的可行性。硅基流动是其中一个平台示例
  (`base_url: https://api.siliconflow.cn/v1` + `model: PaddlePaddle/PaddleOCR-VL-1.5`),
  不是引擎类型。

### 真实模型 E2E(`scripts/real_model_e2e.py`,34 项全部通过,~436s)

凭据走**通用** `OPENAI_API_KEY`/`OPENAI_BASE_URL` 一对,当前指向模力方舟
`https://api.moark.com/v1`(裸模型 ID,不带厂商前缀)。34/34 通过:

- **接口**(16 项):`/health` `/engines` `/presets`(18 项)/`/config` GET/POST/DELETE
  (`key_leaked=False`)/`/ocr`(multipart)/`/ocr/text`(image_url / image_b64 / image_data
  三选一)/`/ocr/{preset}` `/ocr/{preset}/text`/`format=text` 与 `format=markdown`
  返回 `text/plain`(不是 JSON 信封)/缺 file→422/未知预设→400。
- **策略**(18 项):local/vl/seq*/cascade*/bestof* 全部、`bestof:<mode>` 冒号别名、
  `fallback`/`quality` legacy 别名。实测结果与预期一致:
  - `seq`/`cascade`/`fallback` → 首个命中即 `rapidocr`(488 字,~7.6-10.1s)
  - `seq-any`/`seq-line_overlap`/`cascade-line_overlap`/`quality` → 被 reject 后降级
    `multimodal`(324 字,~16-26s),窜行判定确实生效
  - `bestof*` 各评分函数选出不同赢家:`-fastest`/`-longest` → `rapidocr`(488 字),
    `-smart`/`-fluency` → `multimodal` 418 字(Qwen3-VL-30B,简体+标点),
    `-confidence` → `multimodal` 324 字(PaddleOCR-VL-1.5)
- 该轮抓到的缺陷:`TextRequest.image_data` 文档写「完整 data URI」,实现却无条件再加
  一层 `data:` 前缀,合法输入被 400 拒绝。已修复(前缀存在则原样透传),
  并加 `TestTextRequestSourceResolution` 5 个回归用例。

## 维护提示

1. 改 `engine/registry.py` 时注意 `from . import X as base`,不要写 `from . import engine as engine_pkg`
   (registry 本身在 engine 包内,`engine` 是自身)。
2. 改 `BoundingBox` / `TextRegion` 测试时用**关键字参数**(pydantic v2 禁止位置参数构造)。
3. 不要往 repo 提交真实 API key;`.env` 已 gitignore,新环境用 `.env.example` 起手。
4. 加新引擎:实现 `BaseEngine` 子类 + `_recognise_impl` + `_wrap`,用 `@register("name")` 装饰工厂函数;
   若要保留惰性 import,在 `engine/__init__.py` 里 `register_lazy` 即可。
5. 改 API 契约前跑一遍 `pytest tests -q`;当前 237 passed 是基线。
6. `engines_from_config` 不带显式 names 时只用 **enabled** 引擎(尊重 `enabled: false`);
   加新引擎后跑一遍预设测试确认 `local`/`vl` 归类正确(远程名单外的都进 local)。
