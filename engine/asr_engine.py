"""
ASR 引擎基类 + 统一结果格式。

引擎发现机制在 asrgo_export.py 中实现（_list_available_engines + _load_engine_class）。
引擎文件命名约定：engine/{引擎名}/engine_{引擎名}.py
"""

import os


class AsrResult:
    """
    统一的识别结果格式。

    不管用哪个引擎（FireRed 还是 FunASR），
    返回的结果都是这个格式，方便统一处理。
    """

    def __init__(self, text="", sentences=None, words=None):
        """
        创建一个识别结果对象。

        ------ 参数说明 ------
        text：整段音频转出来的完整文字
            比如 "今天天气真好。我们出去玩吧。"

        sentences：句子列表（每句话带时间戳）
            列表里的每个元素是一个字典：
            {
                "start_ms": 1200,  这句话在音频第 1200 毫秒开始说
                "end_ms":   3500,  这句话在第 3500 毫秒说完
                "text": "今天天气真好。"  这句话的文字
            }

        words：词列表（每个词带时间戳，粒度比 sentences 更细）
            列表里的每个元素是一个字典：
            {
                "start_ms": 1200,  这个词在第 1200 毫秒开始说
                "end_ms":   1500,  这个词在第 1500 毫秒说完
                "text": "今天"     这个词的文字
            }

        ------ 参数默认值说明 ------
        sentences=None 的意思是：如果不传这个参数，默认就是 None
        or [] 的意思是：如果值是 None，就变成空列表
        这样不传参数也不会报错
        """
        self.text = text
        self.sentences = sentences or []
        self.words = words or []

    def to_dict(self):
        """
        把结果转成字典，方便存成 JSON 文件。

        JSON 是一种通用的数据格式，别的程序也能读取。
        Python 的字典和 JSON 格式很接近，可以用 json.dump() 直接写文件。
        """
        return {
            "text": self.text,
            "sentences": self.sentences,
            "words": self.words,
        }


class AsrEngine:
    """
    ASR 引擎的基类（也叫父类）。

    什么是"继承"？
    - FireRedEngine 和 FunAsrEngine 都继承自 AsrEngine
    - 继承后，子类自动获得父类已有的方法（比如 name()）
    - 子类必须自己实现 transcribe() 方法
    - 如果子类不实现 transcribe()，调用时会报错
    """

    def transcribe(self, audio_path):
        """
        让引擎识别一段音频，返回 AsrResult 对象。

        ------ 参数 ------
        audio_path：音频文件的路径
          比如 "sample/dp.wav" 或 "C:\\Users\\xxx\\audio.wav"
          推荐用 16kHz 采样率的 WAV 格式

        ------ 返回值 ------
        AsrResult 对象，包含：
          .text       - 完整文字（字符串）
          .sentences  - 句子列表（每个元素带开始时间、结束时间、文字）
          .words      - 词列表（每个元素带开始时间、结束时间、文字）

        ------ raise NotImplementedError 的作用 ------
        这行代码是占位符，表示方法还没实现。
        子类（FireRedEngine / FunAsrEngine）必须重写这个方法。
        如果直接调用基类的 transcribe()，会报错提醒你还没实现。
        """
        raise NotImplementedError

    def name(self):
        """
        返回引擎的名字。

        self.__class__.__name__ 的含义：
        - self：对象自身
        - self.__class__：对象所属的类
          比如 FireRedEngine 的对象，__class__ 就是 FireRedEngine
        - self.__class__.__name__：类的名字的字符串形式
          比如 FireRedEngine 的 __name__ 就是 "FireRedEngine"
        """
        return self.__class__.__name__

