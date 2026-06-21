"""
FireRedASR2-AED 引擎适配器（适配器 = 把复杂的东西包装成简单接口）

FireRed 是什么？
- 小红书开源的语音识别系统
- 精度高，是目前项目里效果最好的引擎（默认引擎）

AED 是什么？
- 全称是"端到端模型"（End-to-End）
- 音频输入，文字输出，不需要中间步骤
- 另一种是 LLM（大语言模型），但 8GB 显存跑不动

这个文件做了什么？
- FireRed 原本的调用方式很复杂（要配置好几个模型）
- 这个文件把它包装成一个简单的 transcribe() 函数
- 只需要调用 transcribe(音频路径) 就行
"""

# ------ 第 1 部分：导入需要用到的工具 ------
#
# import 的作用：把别人写好的代码拿过来用

import sys     # 和 Python 解释器交互，比如修改模块搜索路径
import os      # 操作文件和路径，比如拼接路径、判断文件存在
import subprocess  # 执行系统命令，比如 git clone

# ------- 自动下载 FireRedASR2S 源码（如果本地没有） -------
# fireredasr2s 包来自 GitHub 仓库 FireRedTeam/FireRedASR2S
# 首次使用时会自动 clone 到 models/fireredasr2s/ 目录
_fireredasr2s_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "fireredasr2s")
_fireredasr2s_path = os.path.abspath(_fireredasr2s_path)

# fireredasr2s 包内部目录结构：fireredasr2s/fireredasr2s/__init__.py
# 检查内层 fireredasr2s 包目录是否存在
if not os.path.isdir(os.path.join(_fireredasr2s_path, "fireredasr2s")):
    print(f"[自动下载] 正在从 GitHub 克隆 FireRedASR2S 源码到 {_fireredasr2s_path} ...")
    os.makedirs(os.path.dirname(_fireredasr2s_path), exist_ok=True)
    subprocess.check_call([
        "git", "clone",
        "https://github.com/FireRedTeam/FireRedASR2S.git",
        _fireredasr2s_path
    ])

# 把 fireredasr2s 源码目录加到 Python 模块搜索路径
sys.path.insert(0, _fireredasr2s_path)

# 导入 FireRed 自身的模块
from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
# FireRedAsr2System：FireRed 系统的核心，负责把三个模型串起来运行
# FireRedAsr2SystemConfig：系统的配置，指定用哪个模型、什么参数

from fireredasr2s.fireredasr2 import FireRedAsr2Config
# FireRedAsr2Config：ASR 模型（语音转文字）的配置

from fireredasr2s.fireredvad import FireRedVadConfig
# FireRedVadConfig：VAD 模型的配置
# VAD = Voice Activity Detection = 语音活动检测
# 检测"哪些时间段有人在说话"

from fireredasr2s.fireredpunc import FireRedPuncConfig
# FireRedPuncConfig：标点模型的配置
# 给转出来的文字加逗号、句号、问号等标点

from fireredasr2s.fireredlid import FireRedLidConfig
# FireRedLidConfig：语种识别模型的配置
# LID = Language Identification = 识别说的是哪种语言
# 注意：我们没启用 LID，但配置里必须填一个路径

# 导入模型自动下载工具（从魔搭社区下载模型）
from modelscope.hub.snapshot_download import snapshot_download

# 导入项目自己的基类
from engine.asr_engine import AsrEngine, AsrResult

# 从 engine/fireredasr2s/ 回到 engine/ 再导入 asr_engine


# ------ 第 2 部分：配置模型路径 ------
#
# PRETRAINED 是常量（值不会变的变量）
# 全大写是 Python 的命名习惯，看到全大写就知道是常量

# os.path.dirname(__file__) 是 engine/fireredasr2s/ 目录
# os.path.join(..., "..", "..", "models", "fireredasr2s", "pretrained_models")
PRETRAINED = os.path.join(os.path.dirname(__file__), "..", "..", "models", "fireredasr2s", "pretrained_models")


# ------ 第 3 部分：定义 FireRed 引擎 ------
#
# class FireRedEngine(AsrEngine)：定义一个新类 FireRedEngine
# 括号里的 AsrEngine 表示"继承自 AsrEngine"
# 继承的语法：class 子类名(父类名):
# FireRedEngine 是子类，AsrEngine 是父类


class FireRedEngine(AsrEngine):
    """
    FireRedASR2-AED 引擎。

    工作流程（三步）：
    第一步：FireRedVAD
        先检测音频里哪些时间段有人在说话
        把静音部分切掉，只保留有人声的片段
        这样后面的模型只需要处理有效片段，速度快、准确率高
    第二步：FireRedASR2-AED
        把每个说话片段转成文字
        同时算出每个字在音频里的开始时间和结束时间
        这是核心步骤
    第三步：FireRedPunc
        给文字加标点符号
        ASR 转出来的文字没有标点："今天天气真好我们出去玩吧"
        标点模型加上标点后："今天天气真好，我们出去玩吧。"
    """
    engine_id = "fireredasr2s"  # 用于 CLI 参数和输出目录命名

    def __init__(self, device="cuda:0"):
        """
        初始化 FireRed 引擎。

        这是 Python 类的构造方法（constructor）。
        当写 FireRedEngine() 时，会自动调用这个方法。
        作用：把引擎需要用到的三个模型（VAD、ASR、标点）都配置好。

        ------ 参数 ------
        device：用 CPU 还是 GPU 运行模型
          "cuda:0"  用第 1 块 NVIDIA 显卡（速度快 5-10 倍）
          "cuda:1"  用第 2 块显卡（如果有多个显卡）
          "cpu"     用 CPU（没显卡也能跑，但慢很多）

        ------ self 是什么 ------
        self 是 Python 类里的特殊参数，代表对象自身。
        通过 self.xxx 可以存取这个对象的属性和方法。
        """

        # ------- 自动下载模型（如果本地没有） -------
        # 把模型 ID 和期望的子目录名对应起来
        _models = {
            "FireRedASR2-AED": "FireRedTeam/FireRedASR2-AED",
            "FireRedVAD":      "xukaituo/FireRedVAD",
            "FireRedPunc":     "xukaituo/FireRedPunc",
        }
        # 缓存目录放在 models/fireredasr2s/.modelcache/ 下
        _cache = os.path.join(os.path.dirname(__file__), "..", "..", "models", "fireredasr2s", ".modelcache")
        for _subdir, _model_id in _models.items():
            _target = os.path.join(PRETRAINED, _subdir)
            if not os.path.isdir(_target):
                print(f"[自动下载] 正在从魔搭下载 {_model_id} ...")
                _path = snapshot_download(_model_id, cache_dir=_cache)
                os.makedirs(os.path.dirname(_target), exist_ok=True)
                os.symlink(_path, _target, target_is_directory=True)
        # LID 目录即使不用也要存在，否则配置会报错
        os.makedirs(os.path.join(PRETRAINED, "FireRedLID"), exist_ok=True)

        config = FireRedAsr2SystemConfig(
            # ---- VAD 模型配置 ----
            # VAD = 语音活动检测
            # vad_model_dir 指向 VAD 模型文件所在的文件夹
            vad_model_dir=os.path.join(PRETRAINED, "FireRedVAD", "VAD"),

            # ---- LID 模型配置 ----
            # LID = 语种识别（没启用）
            # 但 FireRed 要求必须填一个路径
            lid_model_dir=os.path.join(PRETRAINED, "FireRedLID"),

            # ---- ASR 模型类型 ----
            # 两种选择：
            # "aed" — 端到端模型（我们在用的，8GB 显卡跑得动）
            # "llm" — 大语言模型（OOM = 显存溢出，8GB 跑不动）
            asr_type="aed",

            # ---- ASR 模型配置 ----
            # 核心模型：把语音转成文字
            asr_model_dir=os.path.join(PRETRAINED, "FireRedASR2-AED"),

            # ---- 标点模型配置 ----
            # 给文字加逗号句号
            punc_model_dir=os.path.join(PRETRAINED, "FireRedPunc"),

            # ---- VAD 详细配置 ----
            vad_config=FireRedVadConfig(
                use_gpu=device != "cpu",  # True=用显卡，False=用 CPU
            ),

            # ---- LID 详细配置 ----
            # 没启用，但必须传
            lid_config=FireRedLidConfig(
                use_gpu=device != "cpu",
                use_half=False,  # float32 精度
            ),

            # ---- ASR 详细配置 ----
            asr_config=FireRedAsr2Config(
                use_gpu=device != "cpu",   # True 用显卡，False 用 CPU
                use_half=False,             # float32（更准但更耗显存）
                                            # float16（省显存但精度稍降）
                beam_size=3,                # 搜索宽度，越大越准但越慢
                                            # 1 最快但不准，3 是平衡点
                                            # 10 会很慢但提升不大
                return_timestamp=True,      # True = 返回每个字的时间戳
                                            # False = 只返回文字，不要时间
                                            # 字幕工具需要时间戳
            ),

            # ---- 标点详细配置 ----
            punc_config=FireRedPuncConfig(
                use_gpu=device != "cpu",
            ),

            # ---- 启用的功能开关 ----
            enable_vad=True,   # True = 先做语音检测，分段后再识别
                               # 推荐开启，长音频效果更好
                               # False = 整段音频直接扔给 ASR
            enable_lid=False,  # True = 自动识别语种，我们用不到
                               # False = 不识别语种
            enable_punc=True,  # True = 加标点符号
                               # False = 纯文字不带标点
                               # 推荐开启，否则结果没有逗号句号
        )

        # self._system 是引擎内部的 FireRed 系统
        # 下划线 _ 开头的变量名表示"内部使用，外部不要直接访问"
        self._system = FireRedAsr2System(config)

    def transcribe(self, audio_path):
        """
        识别一段音频，返回统一格式的 AsrResult。

        这是最重要的方法，作用是把音频转成文字。

        ------ 参数 ------
        audio_path：音频文件的路径
          比如 "sample/dp.wav" 或 "C:/音乐/录音.wav"
          推荐用 16kHz 采样率的 WAV 文件
          MP3 或其他格式可能报错

        ------ 返回值 ------
        AsrResult 对象，包含：
          .text       - 字符串，完整文字
          .sentences  - 列表，每句话带时间戳
          .words      - 列表，每个词带时间戳

        ------ 工作流程 ------
        1. 调用 self._system.process(audio_path)
           FireRed 自己的方法，返回原始结果（字典格式）
        2. 从原始结果中提取 text / sentences / words
        3. 封装成统一的 AsrResult 格式返回
        """
        # 调用 FireRed 系统处理音频，返回字典
        raw = self._system.process(audio_path)

        # 把 FireRed 的句子列表转成统一格式
        # raw.get("sentences", []) 从字典取 sentences 的值，不存在则用空列表
        sentences = []
        for s in raw.get("sentences", []):
            sentences.append({
                "start_ms": s["start_ms"],  # 句子开始时间（毫秒）
                "end_ms":   s["end_ms"],    # 句子结束时间（毫秒）
                "text":     s["text"],       # 句子文字
            })

        # words 粒度比 sentences 更细，每个词一个条目
        words = []
        for w in raw.get("words", []):
            words.append({
                "start_ms": w["start_ms"],
                "end_ms":   w["end_ms"],
                "text":     w["text"],
            })

        return AsrResult(
            text=raw.get("text", ""),
            sentences=sentences,
            words=words,
        )

    def name(self):
        """
        返回引擎的名字。

        覆盖（override）了父类 AsrEngine 的 name() 方法。
        覆盖的意思是：子类重新实现父类的方法。
        不覆盖的话，父类返回的是 "FireRedEngine"（类名）。
        覆盖后返回 "FireRedASR2-AED"（模型的实际名字）。
        """
        return "FireRedASR2-AED"
