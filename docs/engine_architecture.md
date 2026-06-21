# 引擎架构文档

## 插件式设计

ASRGo 采用插件式引擎架构。添加新引擎不需要修改已有代码，只需按约定创建文件。

核心机制：

1. `engine/asr_engine.py` 定义基类 `AsrEngine`
2. `asrgo_export.py` 扫描 `engine/{引擎名}/engine_{引擎名}.py` 文件名，列出可用引擎
3. 只导入用户选择的引擎文件，不扫描其他引擎代码
4. 每个引擎文件中的 `AsrEngine` 子类被加载使用

```mermaid
flowchart TD
    AsrEngine("AsrEngine（引擎基类）")
    FireRedEngine("FireRedEngine（高精度引擎）")
    FunAsrEngine("FunAsrEngine（轻量级引擎）")
    YourEngine("YourEngine（你添加的引擎）")
    AsrResult("AsrResult（统一返回结果）")

    AsrEngine -->|继承| FireRedEngine
    AsrEngine -->|继承| FunAsrEngine
    AsrEngine -->|继承| YourEngine
    FireRedEngine -.->|返回| AsrResult
    FunAsrEngine -.->|返回| AsrResult
    YourEngine -.->|返回| AsrResult

    style AsrEngine fill:#2e3440,stroke:#6c7086,stroke-width:3px,color:#d8dee9
    style FireRedEngine fill:#3d2e2e,stroke:#8a6e6e,stroke-width:3px,color:#d8dee9
    style FunAsrEngine fill:#2e3838,stroke:#6e8888,stroke-width:3px,color:#d8dee9
    style AsrResult fill:#3a3828,stroke:#8a7a5e,stroke-width:3px,color:#d8dee9
    style YourEngine fill:#3a2e44,stroke:#7a6a8a,stroke-width:3px,color:#d8dee9
```

### 引擎对比

| 特性 | FireRedASR2 (默认) | FunASR |
|---|---|---|
| 来源 | 小红书开源 | 阿里达摩院开源 |
| 核心模型 | FireRedASR2-AED | Paraformer-zh |
| 误字率 | **CER 3.05%** (业界领先) | CER ~5% |
| 显存需求 | ~8 GB | **~2 GB** (轻量) |
| 时间戳粒度 | **句子级 + 词级** | 句子级 |
| VAD 模型 | FireRedVAD | FSMN-VAD |
| 标点模型 | FireRedPunc | CT-Transformer |
| 管道流程 | VAD → AED → Punc | VAD → Paraformer → Punc |
| 适用场景 | 追求最高精度 | 显存受限/快速部署 |

### 为什么有两个引擎？

**FireRedASR2** — 高精度首选
- CER 3.05%，是目前国产开源 ASR 的天花板
- 词级时间戳，字幕对齐更精确
- 适合：有声书转写、专业字幕制作、精度要求高的场景

**FunASR** — 轻量级之选
- 仅需 2GB 显存，没有 GPU 也能用 CPU 跑
- 阿里巴巴出品，社区活跃、文档完善
- 适合：嵌入式部署、低配机器、快速验证

### 统一返回格式

所有引擎返回 `AsrResult` 对象，包含三个字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | str | 完整识别文本 |
| `sentences` | list[dict] | 句子列表，每句含 `start_ms`、`end_ms`、`text` |
| `words` | list[dict] | 词列表，每词含 `start_ms`、`end_ms`、`text` |

## 添加新引擎（以 Whisper 为例）

### 第 1 步：创建目录和文件

```
engine/
└── whisper/
    └── engine_whisper.py
```

### 第 2 步：编写引擎类

```python
# engine/whisper/engine_whisper.py
import whisper
from engine.asr_engine import AsrEngine, AsrResult


class WhisperEngine(AsrEngine):
    engine_id = "whisper"  # 对应 --engine whisper

    def __init__(self, device="cuda:0"):
        # 在引擎初始化时加载模型
        self.model = whisper.load_model("large-v3", device=device)

    def transcribe(self, audio_path):
        # 调用 Whisper 做识别
        result = self.model.transcribe(audio_path, language="zh")

        # 组装成 AsrResult 格式
        sentences = []
        words = []
        for seg in result["segments"]:
            sentences.append({
                "start_ms": int(seg["start"] * 1000),
                "end_ms": int(seg["end"] * 1000),
                "text": seg["text"].strip(),
            })

        return AsrResult(
            text=result["text"].strip(),
            sentences=sentences,
            words=words,
        )
```

### 第 3 步：使用

```bash
python asrgo_export.py --engine whisper --audio sample/input.wav
```

`_list_available_engines()` 会自动发现 `WhisperEngine`，无需修改任何已有文件。

## 引擎自动发现流程

```mermaid
flowchart TD
    subgraph s1[发现阶段]
        start["asrgo_export.py 启动"] --> discover["_list_available_engines()"]
        discover --> scan["扫描 engine/{name}/engine_{name}.py 文件名"]
        scan --> reg["列出可用引擎"]
        reg --> dict["可用引擎: fireredasr2s, funasr"]
    end
    style s1 fill:#1e3a8a,stroke:#648fff,stroke-width:2px,color:#ffffff

    subgraph s2[执行阶段]
        dict --> parse["解析 --engine 参数"]
        parse --> create["创建引擎实例"]
        create --> load["加载模型到显存"]

        load --> firered_load["FireRed: 加载 VAD + AED + Punc"]
        load --> funasr_load["FunASR: 加载 VAD + Paraformer + Punc"]

        firered_load --> transcribe["调用 transcribe(audio_path)"]
        funasr_load --> transcribe

        transcribe --> firered_pipe["FireRed 管道"]
        transcribe --> funasr_pipe["FunASR 管道"]

        firered_pipe --> result["AsrResult(text, sentences, words)"]
        funasr_pipe --> result

        result --> export["导出 TXT / SRT / VTT / JSON"]
    end
    style s2 fill:#455a64,stroke:#90a4ae,stroke-width:2px,color:#ffffff

    subgraph s3[FireRed管道]
        firered_pipe --> v1["1. FireRedVAD<br/>检测人声片段"]
        v1 --> aed["2. FireRedASR2-AED<br/>语音转文字 + 词级时间戳"]
        aed --> p1["3. FireRedPunc<br/>加标点符号"]
    end
    style s3 fill:#c62828,stroke:#ef9a9a,stroke-width:2px,color:#ffffff

    subgraph s4[FunASR管道]
        funasr_pipe --> v2["1. FSMN-VAD<br/>检测人声片段"]
        v2 --> pf["2. Paraformer-zh<br/>语音转文字"]
        pf --> p2["3. CT-Transformer<br/>加标点符号"]
    end
    style s4 fill:#00695c,stroke:#80cbc4,stroke-width:2px,color:#ffffff

    classDef builtin fill:#dce5ef,stroke:#5c6b7e,stroke-width:2px,color:#2e3440
    classDef custom fill:#e8e0ee,stroke:#7a6a8a,stroke-width:2px,color:#3a2e44
    classDef firered fill:#f0e0e0,stroke:#7a5e5e,stroke-width:2px,color:#3a2a2a
    classDef funasr fill:#dceeec,stroke:#5e7e7a,stroke-width:2px,color:#2a3a38
    classDef result fill:#f0ece0,stroke:#8a7a5e,stroke-width:2px,color:#3a3828

    class start builtin
    class discover builtin
    class scan builtin
    class reg builtin
    class dict builtin
    class parse builtin
    class create builtin
    class load builtin
    class firered_load firered
    class firered_pipe firered
    class v1 firered
    class aed firered
    class p1 firered
    class funasr_load funasr
    class funasr_pipe funasr
    class v2 funasr
    class pf funasr
    class p2 funasr
    class transcribe result
    class result result
    class export result
```

## 内置引擎

### FireRedASR2-AED（高精度引擎）

位于 `engine/fireredasr2s/engine_fireredasr2s.py`。

| 属性 | 值 |
|---|---|
| 引擎 ID | `fireredasr2s` |
| 来源 | 小红书开源 |
| 核心模型 | FireRedASR2-AED (端到端) |
| 辅助模型 | FireRedVAD + FireRedPunc |
| 误字率 | **CER 3.05%** |
| 显存 | ~8 GB |
| 时间戳 | 句子级 + **词级** |

**管道流程：**
```
音频 → FireRedVAD(切分人声) → FireRedASR2-AED(转文字+时间戳) → FireRedPunc(加标点) → AsrResult
```

### FunASR Paraformer（轻量级引擎）

位于 `engine/funasr/engine_funasr.py`。

| 属性 | 值 |
|---|---|
| 引擎 ID | `funasr` |
| 来源 | 阿里达摩院开源 |
| 核心模型 | Paraformer-zh |
| 辅助模型 | FSMN-VAD + CT-Transformer |
| 误字率 | ~5% CER |
| 显存 | **~2 GB** |
| 时间戳 | 句子级 |

**管道流程：**
```
音频 → FSMN-VAD(切分人声) → Paraformer-zh(转文字) → CT-Transformer(加标点) → AsrResult
```

### 如何选择？

| 场景 | 推荐引擎 | 理由 |
|---|---|---|
| 有声书转写 | FireRedASR2 | 词级时间戳，对齐精确 |
| 专业字幕制作 | FireRedASR2 | CER 3.05%，业界领先 |
| 低配机器 | FunASR | 仅需 2GB 显存 |
| 嵌入式部署 | FunASR | 轻量级，CPU 也能跑 |
| 快速验证 | FunASR | 启动快，社区活跃 |

## 规范约定

| 项目 | 要求 |
|---|---|
| 文件路径 | `engine/{引擎名}/engine_{引擎名}.py` |
| 类名 | 任意，但需继承 `AsrEngine` |
| 目录名 | 目录名即为 `--engine` 参数值，如 `fireredasr2s`、`funasr` |
| engine_id | 可选类属性，用于旧版兼容 |
| \_\_init\_\_ | 接收 `device="cuda:0"` 参数，加载模型 |
| transcribe() | 接收 `audio_path`，返回 `AsrResult` |
