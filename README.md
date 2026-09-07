# jykj_ocr

多引擎 OCR 服务:本地 **RapidOCR**(离线、零凭据)+ 远程多模态大模型(任意 OpenAI
兼容端点),策略层编排调用顺序与重试,对外提供 CLI、Python API 与 FastAPI 接口。

```
┌──────────┐   ┌────────────┐   ┌──────────────────────────────────┐
│ CLI / API│──▶│  Strategy  │──▶│ rapidocr   (本地 ONNX,离线)      │
│  / HTTP  │   │ (重试链)   │   │ multimodal (OpenAI 兼容,任意平台) │
└──────────┘   └────────────┘   └──────────────────────────────────┘
```

## 引擎

只有两个引擎类型:

| 引擎 | 类型 | API key | 留空时的默认模型 |
|------|------|:-------:|-----------------|
| `rapidocr` | 本地 ONNX | ❌ | — |
| `multimodal` | 远程多模态 | ✅ | `PaddleOCR-VL-1.5` |

**平台不是引擎。** 硅基流动、模力方舟、阿里云百炼、火山方舟、智谱、本地 vLLM 都只是
平台,由条目里的 `base_url` + `model` 区分,不是额外的引擎类型。`multimodal` 是
「类型」不是「实例 id」——config 里可以写任意多条,由 `(base_url, model, api_key)` 区分:

```yaml
engines:
  - name: multimodal                 # 硅基流动:模型 ID 带厂商前缀
    base_url: https://api.siliconflow.cn/v1
    model: PaddlePaddle/PaddleOCR-VL-1.5
  - name: multimodal                 # 模力方舟:裸模型 ID;base_url 留空走 OPENAI_BASE_URL
    model: Qwen3-VL-30B-A3B-Instruct
```

完全相同的条目解析时合并为一条(第一条胜出)。`rapid` / `openai` / `siliconflow` 等
别名一律归一化到上面两个类型,结果里的 `engine` 字段只有 `rapidocr` 或 `multimodal`。
远程引擎走 OpenAI 兼容 `/chat/completions`,只依赖 `requests`(不引入 `openai` SDK)。

## 策略预设

配置文件定**默认策略**;CLI / HTTP 可按请求**一次性切换**(不动配置,只作用于本次请求)。

**顺序预设** `seq*` / `cascade*` — 按引擎顺序尝试,首个命中即返回;`cascade*` 与
`seq*` 共用同一套逻辑,区别只在被 reject 后是否重试同一引擎(`max_retries=0`)。

| 预设 | 引擎范围 | retry_mode | 文本重排 |
|------|----------|------------|:--------:|
| `local` | 仅本地,远程禁用 | `no_text` | ❌ |
| `vl` | 仅 1 条远程 VL(config 里第一条已启用的),其余禁用 | `no_text` | ❌ |
| `seq` | 全部启用引擎(默认) | `no_text` | ❌ |
| `seq-any` | 低置信度或窜行即降级 | `any` | ✅ 按坐标重建阅读顺序 |
| `seq-low_conf` | 低置信度降级 | `low_confidence` | ❌ |
| `seq-line_overlap` | 窜行降级 | `line_overlap` | ❌ |
| `cascade*` | 同上三个,但不重试同一引擎 | 见括号后缀 | ❌ |

**最佳预设** `bestof*` — 所有引擎各跑一次,按评分选最佳(比 `seq*` 慢,但拿到候选里
最好的结果):`bestof` / `bestof-smart`(置信度 − 窜行惩罚 + 文本长度 + 语义流畅度,
默认)/ `bestof-fastest` / `bestof-confidence` / `bestof-longest` / `bestof-fluency` /
`bestof:<mode>`(冒号别名)。

legacy 别名:`fallback` == `seq`,`quality` == `seq-any`。`retry_mode` 可选
`no_text` / `low_confidence` / `line_overlap` / `any` / `none`。

## 安装

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows;Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                    # 可选,提供 jykj-ocr 命令
```

`requirements.txt` 自包含,无需任何系统包(见下)。

**rapidocr 的 OpenCV 依赖**:传递依赖默认是 GUI 版 `opencv-python`,在 Linux 最小化镜像
上 import 需要系统库 `libGL.so.1`。本项目改装 `opencv-python-headless`(OCR 功能等价,
不链接 libGL),因此 Windows / Linux / Docker 都是同一条 `pip install`。存量环境若已装
GUI 版,先 `pip uninstall -y opencv-python` 再装 headless,或用
`scripts/diag_rapidocr_import.py` 逐层打印真实 import 错误。

## 配置

一份配置文件搞定一切:`engines`(引擎条目)+ `strategy`(默认策略)+ `output` + `pdf`。
仓库自带 4 份示例,用 `-c` 选:

| 文件 | 内容 |
|------|------|
| `config/config.local.yaml` | 示例 1:只本地 rapidocr,零凭据 |
| `config/config.vl.yaml` | 示例 2:只远程 VL(硅基流动 + 模力方舟) |
| `config/config.seq.yaml` | 示例 3:本地 + 1 个远程兜底 |
| `config/config.bestof.yaml` | 示例 4:本地 + 2 个远程,bestof 评分选最佳 |

**环境变量按序号,每条各读各的** —— 这是多平台的唯一方式。配置里有 N 条
`multimodal`,就按 yaml 书写顺序读 `*_1`、`*_2`……数量不限,**yaml 里可以完全不写
URL / 模型 / key**:

| 变量 | 用途 |
|------|------|
| `OPENAI_BASE_URL_<N>` | 第 N 条 `multimodal` 的端点 |
| `JYKJ_OCR_MULTIMODAL_<N>_API_KEY` | 第 N 条的 key,N 不限(简写 `MULTIMODAL_<N>_API_KEY` 同样生效) |
| `JYKJ_OCR_MULTIMODAL_<N>_MODEL` | 第 N 条的模型 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `JYKJ_OCR_MULTIMODAL_MODEL` | 不带序号的回退,该类型**所有条目共享**——只有单平台才够用 |
| `JYKJ_OCR_CONFIG` / `JYKJ_OCR_PORT` | 配置文件路径 / 服务端口 |

```bash
export OPENAI_BASE_URL_1=https://api.siliconflow.cn/v1     # 第 1 条
export OPENAI_BASE_URL_2=https://api.moark.com/v1          # 第 2 条
export JYKJ_OCR_MULTIMODAL_1_API_KEY=***
export JYKJ_OCR_MULTIMODAL_2_API_KEY=***
export JYKJ_OCR_MULTIMODAL_1_MODEL=PaddlePaddle/PaddleOCR-VL-1.5
export JYKJ_OCR_MULTIMODAL_2_MODEL=PaddleOCR-VL-1.5
```

yaml 字段优先级最高:条目里写了 `base_url` / `model` / `api_key` 就用条目,留空才读
环境变量,两种写法可以混用。想让某条的序号不随条目增删漂移,在 yaml 里钉住
`instance: 1`(未钉住的条目自动跳过该号)。

**优先级**(高 → 低):单次请求参数(`--strategy-name`、`strategy_name`、`/ocr/{preset}`
路由) > `POST /config` 运行时覆盖 > yaml 字段 > 环境变量 > 内置默认值。
方向是「yaml 有值就用 yaml,留空才回退环境变量」——**环境变量不会覆盖 yaml 里已写的值**。
唯一保留的默认是模型名 `PaddleOCR-VL-1.5`;base URL 没有厂商默认,解析不出来直接
`EngineNotAvailable`(静默回退到某个平台会把 A 平台的 key 发给 B,只表现为一个说不清
原因的 HTTP 401)。

**yaml 里的 `strategy.name` 不生效**,它只是文档字段——预设交给接口。真正生效的只有
`bestof_mode`(单独触发 `BestofEngine`)、`max_retries`、`retry_mode`、`min_confidence`;
「只用本地 / 只用远程」靠条目里的 `enabled: false`。环境变量只在进程启动时读入,
改完必须重启。

## CLI

```bash
python -m jykj_ocr --list-engines
python -m jykj_ocr image.png --engine multimodal --format json
python -m jykj_ocr doc.pdf   --engine rapidocr  --format markdown -o out.md
python -m jykj_ocr image.png --strategy-name bestof   # 一次性预设
python -m jykj_ocr image.png                          # 不指定引擎 → 走策略链
python -m jykj_ocr serve --port 8000                  # 启动 HTTP 服务
```

| 参数 | 说明 |
|------|------|
| `source` | 图片/PDF 路径或 `http(s)://` URL |
| `-c` / `--config` | 配置文件路径(默认 `config/config.yaml`) |
| `--engine` | 强制某引擎,绕过策略链 |
| `--strategy-name` | 一次性预设:`local` \| `vl` \| `seq*` \| `bestof*` |
| `--format` | `text` \| `markdown` \| `json`(默认 `text`) |
| `-o` / `--output` | 输出文件;缺省打印到 stdout |
| `--max-pages` / `--dpi` | PDF 最多页数 / 渲染 DPI(默认 200) |
| `serve` | 关键字(非子命令),放任意位置均可;此时 `--host` / `--port` 生效 |

## Python API

```python
import jykj_ocr

results = jykj_ocr.ocr("image.png", engine="multimodal")   # 指定引擎
results = jykj_ocr.ocr("doc.pdf")                          # 走策略链
text    = jykj_ocr.ocr_to_text("image.png", engine="rapidocr")

jykj_ocr.ocr("image.png", strategy_name="bestof-smart")
```

`ocr(source, *, engine=None, config=None, config_path=None, max_pages=None, dpi=200,
retries=1, strategy_name=None) -> List[OCRResult]`,每页一个。`source` 接受路径或
`http(s)://` URL,不接受裸 bytes。

## HTTP API

```bash
python -m jykj_ocr serve          # 或 JYKJ_OCR_PORT=9000 ...
```

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/engines` | 可用引擎 + 当前引擎顺序 |
| `GET` | `/presets` | 全部命名预设的元数据 |
| `GET` / `POST` / `DELETE` | `/config` | 查看 / 运行时覆盖 / 清除覆盖(不返回 key 明文) |
| `POST` | `/ocr` | multipart 上传识别 |
| `POST` | `/ocr/text` | JSON body,按 `image_url` / `image_b64` / `image_data` |
| `POST` | `/ocr/{preset}` | 路由即策略(multipart) |
| `POST` | `/ocr/{preset}/text` | 路由即策略(JSON) |

四个 OCR 端点返回结构一致:`{pages, text, engine, page_count}`;`format=text/markdown`
时退化为纯文本。form / JSON 字段:`file` 或 `image_url`/`image_b64`/`image_data`、
`engine`、`model`、`base_url`、`api_key`、`prompt`、`strategy`(JSON 字符串)、
`strategy_name`、`retry_mode`、`score_mode`、`max_retries`、`max_pages`、`dpi`、`format`。
`model` / `base_url` / `api_key` / `prompt` 只对远程引擎生效,且都是**一次性**覆盖——
`base_url` + `api_key` 留空时仍按原顺序回退环境变量或配置文件里的值,所以 `/vl` 可以在
单次请求内切到另一个平台或账号,不需要动 `POST /config`。异常映射:`InputError → 400`、
`EngineNotAvailable` / `StrategyError → 422`、`EngineError → 502`。

```bash
curl -s http://localhost:8000/ocr -F "file=@image.png" -F "strategy_name=bestof"
curl -s http://localhost:8000/ocr/bestof-fluency -F "file=@image.png"
curl -s http://localhost:8000/ocr/text -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/scan.png"}'
# 单次请求切平台/账号:key 省略时用 export 的环境变量
curl -s http://localhost:8000/ocr/vl -F "file=@image.png" \
  -F "model=Qwen/Qwen3-VL-30B-A3B-Instruct" \
  -F "base_url=https://api.siliconflow.cn/v1"
curl -s -X POST http://localhost:8000/config -H "Content-Type: application/json" \
  -d '{"engines":[{"name":"multimodal","model":"Qwen3-VL-30B-A3B-Instruct"}]}'
```

## Docker

```bash
docker build -t jykj_ocr .
docker run --rm -p 8000:8000 \
    -e OPENAI_API_KEY=sk-... -e OPENAI_BASE_URL=https://api.siliconflow.cn/v1 jykj_ocr

docker compose up --build                      # 含 healthcheck、权重持久化卷

docker run --rm -e OPENAI_API_KEY=sk-... -v "$PWD:/data" jykj_ocr \
    python -m jykj_ocr /data/image.png --engine multimodal
```

镜像 `python:3.11-slim`、非 root、含 `HEALTHCHECK`;配置与端口通过
`JYKJ_OCR_CONFIG` / `JYKJ_OCR_PORT` 注入。

## 测试

```bash
.venv/Scripts/python -m pytest tests -q       # 237 个用例,全部离线,无真实 API 调用
```

覆盖 models、config(别名归一化 / YAML / 环境变量优先级 / 多实例去重)、strategy
(重试链 / 谓词)、engines(multimodal 的 OpenAI 响应解析、rapidocr 的 1.x / 1.4.x 返回
形态)、server(HTTP 路由与预设、三种图片来源)、presets(`seq*` / `cascade*` / `bestof*`)、
cli(`serve` 关键字解析、JSON 输出、退出码)。

真实模型回归(需联网与真实 key,非 CI 基线):

```bash
export OPENAI_API_KEY=...  OPENAI_BASE_URL=https://api.siliconflow.cn/v1
.venv/Scripts/python scripts/demo.py --ci                    # 全场景演示
JYKJ_OCR_PORT=8010 .venv/Scripts/python -m jykj_ocr serve &  # 起服务
.venv/Scripts/python scripts/real_model_e2e.py \
    http://127.0.0.1:8010 tests/兰亭序.jpeg                   # 34 项,约 7 分钟
```

## 项目布局

```
src/jykj_ocr/
├── __init__.py            # 顶层 API:ocr() / ocr_to_text()
├── config.py              # Config / EngineConfig / load_config / normalise_engine
├── models.py              # Point / BoundingBox / TextRegion / OCRResult
├── strategy.py            # StrategyEngine / BestofEngine / 重试谓词
├── engine/                # 惰性注册 / base / inputs / registry
├── engines/               # rapidocr_engine.py / multimodal_engine.py
├── cli.py                 # argparse CLI
└── server.py              # FastAPI 路由
config/                    # config.yaml + 4 份示例(见「配置」章节)
tests/                     # pytest,237 passed
scripts/                   # 诊断与真实模型回归脚本
Dockerfile / docker-compose.yml / requirements.txt / pyproject.toml
```

架构细节、端到端排查与接口实测记录见 `docs/manual.md`;架构图见
`docs/architecture.mmd` / `docs/architecture.html`。

## 许可

MIT
