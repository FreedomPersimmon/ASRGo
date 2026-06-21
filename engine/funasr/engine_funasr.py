"""
FunASR（阿里巴巴语音识别工具箱）引擎适配器。

FunASR 是阿里巴巴达摩院开源的语音识别工具包。
我们用的是 Paraformer-zh（中文语音识别模型）。

这个文件做了什么？
- 把 FunASR 的调用方式，包装成统一的 transcribe() 函数
- 这样 asrgo_export.py 不需要知道底层是 FireRed 还是 FunASR

里面用了三个模型：
1. FSMN VAD — 语音活动检测
   找出音频里哪些时间段有人在说话
   把沉默部分切掉，只保留有人声的片段
2. Paraformer-zh — 把语音转成文字
   核心模型，把音频信号转成中文文字
3. 标点模型 — 给文字加逗号句号
   ASR 转出来的文字没有标点："今天天气真好我们出去玩吧"
   加上标点后："今天天气真好，我们出去玩吧。"
"""
import os
import shutil
from funasr import AutoModel
from modelscope.hub.snapshot_download import snapshot_download
from engine.asr_engine import AsrEngine, AsrResult

# 从 engine/funasr/ 回到 engine/ 再导入 asr_engine


class FunAsrEngine(AsrEngine):
    """
    FunASR 引擎（Paraformer + VAD + 标点）。

    工作流程：
    第一步：FSMN VAD — 检测人声片段
    第二步：Paraformer — 把每个片段转成文字
    第三步：标点模型 — 加上逗号句号
    """
    engine_id = "funasr"  # 用于 CLI 参数和输出目录命名

    def __init__(self, device="cuda:0",
                 asr_model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                 vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                 punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"):
        """
        初始化 FunASR 引擎。

        构造方法，写 FunAsrEngine() 时自动调用。
        任务：加载三个模型（VAD、ASR、标点）。

        ------ 模型来源 ------
        模型优先从 models/funasr/ 加载（本地缓存）。
        如果 models/funasr/ 下没有，自动从魔塔下载并复制到 models/funasr/。
        """
        
        # 模型在 models/funasr/ 下的目录名（和魔塔上的 ID 一致）
        _base = os.path.join(os.path.dirname(__file__), "..", "..", "models", "funasr")
        _local_asr = os.path.join(_base, "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
        _local_vad = os.path.join(_base, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
        _local_punc = os.path.join(_base, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")

        # ------ 判断模型在不在本地 ------
        # 检查 models/funasr/ 目录下有没有 ASR 模型（三个模型中最大的那个）
        # 如果 ASR 模型目录存在，说明三个模型之前都已下载好了
        if os.path.isdir(_local_asr):

            # --- 情况 1：模型已在本地 ---
            # 直接把本地路径传给 AutoModel，AutoModel 去这个路径读文件加载
            # 本地路径示例：models/funasr/xxx/ 下放着 model.pth.tar 等文件
            # AutoModel 看到是本地路径就不会联网，直接读文件
            self._model = AutoModel(
                model=_local_asr,      # 传本地路径，不是魔塔 ID
                vad_model=_local_vad,  # VAD 模型本地路径
                punc_model=_local_punc,# 标点模型本地路径
                disable_update=True,   # 不检查更新，加快启动
                device=device,
            )

        else:

            # --- 情况 2：模型不在本地 ---
            # 三步走：下载 → 复制到项目目录 → 从项目目录加载

            print("[自动下载] FunASR 模型不存在，正在从魔塔下载最新版...")

            # 第 1 步：确保 models/funasr/ 目录存在
            os.makedirs(_base, exist_ok=True)

            # 第 2 步：下载三个模型并复制到项目目录
            # snapshot_download(魔塔ID) 下载到 ~/.cache/modelscope/hub/，返回缓存路径
            # shutil.copytree(缓存路径, 项目目录路径) 把模型文件复制到 models/funasr/
            for _model_id, _local_dir in [
                (asr_model, _local_asr),   # ASR 模型
                (vad_model, _local_vad),   # VAD 模型
                (punc_model, _local_punc), # 标点模型
            ]:
                _cache_path = snapshot_download(_model_id)   # 先下载到缓存
                shutil.copytree(_cache_path, _local_dir)     # 再复制到项目目录

            # 第 3 步：从项目目录加载（和情况 1 一样的代码）
            self._model = AutoModel(
                model=_local_asr,      # 本地路径，不联网
                vad_model=_local_vad,
                punc_model=_local_punc,
                disable_update=True,
                device=device,
            )

    def transcribe(self, audio_path):
        """
        识别一段音频，返回统一格式的 AsrResult。

        ------ 参数 ------
        audio_path：音频文件的路径，比如 "sample/dp.wav"

        ------ 返回值 ------
        AsrResult 对象，包含 text / sentences / words

        ------ 工作流程 ------
        1. 调用 self._model.generate() 让 FunASR 模型处理音频
        2. 从结果中提取文字和时间戳
        3. 封装成 AsrResult 返回
        """
        # batch_size_s=60 表示每次处理 60 秒的音频
        # 值小省内存但慢，值大快但吃内存
        raw = self._model.generate(input=audio_path, batch_size_s=60)

        # FunASR 返回的结果可能是列表，也可能是单个字典
        # 列表的每个元素是一段话（VAD 切出来的）
        # 每段话格式：
        # {
        #   "text": "今天天气真好",
        #   "timestamp": [[1200, 1500], [1500, 1800], [1800, 2200]]
        # }
        # timestamp 是每个字的开始和结束时间（毫秒）
        segments = raw if isinstance(raw, list) else [raw]

        full_text = ""
        sentences = []
        words = []

        for seg in segments:
            text = seg.get("text", "")
            if text:
                full_text += text + "\n"

            ts_list = seg.get("timestamp", [])

            if ts_list and len(ts_list) > 0:
                # 取第一个字的开始时间和最后一个字的结束时间
                # ts_list[0][0] — 第一个字的开始毫秒
                # ts_list[-1][1] — 最后一个字的结束毫秒
                # Python 里 -1 表示最后一个元素
                start_ms = ts_list[0][0]
                end_ms = ts_list[-1][1]

                sentences.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                })

                for ts in ts_list:
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        words.append({
                            "start_ms": ts[0],
                            "end_ms": ts[1],
                            "text": "",  # FunASR 不返回每个字的具体文字
                        })
            else:
                # 没有时间戳时用默认时间 0
                sentences.append({
                    "start_ms": 0,
                    "end_ms": 0,
                    "text": text,
                })

        return AsrResult(text=full_text.strip(), sentences=sentences, words=words)

    def name(self):
        """
        返回引擎的名字。
        覆盖父类的 name() 方法。
        """
        return "Paraformer-zh"
