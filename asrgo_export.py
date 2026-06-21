"""
ASRGo 语音识别系统 — 唯一入口脚本。

这个文件是整个项目的"大门"。
用户运行这个文件，就可以完成从语音到文字的全部工作。
不需要关心底层用的是什么引擎、模型怎么加载的。

工作流程：
第 1 步：自动发现所有引擎（FireRed、FunASR 等）
第 2 步：解析用户输入的命令行参数
第 3 步：检查音频文件是否存在
第 4 步：创建引擎实例（这时才会真正加载模型到显存）
第 5 步：调用引擎的 transcribe() 方法，把音频转成文字
第 6 步：导出 4 种格式：TXT / SRT / VTT / JSON
"""
import os
import sys
import json
import argparse
import importlib.util
import inspect

from engine.asr_engine import AsrEngine


def fmt_srt(ms):
    """
    把毫秒转成 SRT 字幕格式的时间字符串。

    SRT 时间格式：HH:MM:SS,mmm
    比如 3661000 毫秒 → 01:01:01,000

    ------ divmod 的作用 ------
    divmod(a, b) 返回 (a // b, a % b) 两个值
    比如 divmod(3661000, 3600000) → (1, 61000)
    1 是小时数，61000 是剩下的毫秒
    这样写比手动算 // 和 % 更简洁

    ------ 转换步骤 ------
    ms = 3661000
    divmod(3661000, 3600000) → hours=1, r=61000
    divmod(61000, 60000)     → minutes=1, r=1000
    divmod(1000, 1000)       → seconds=1, ms=0
    结果：01:01:01,000
    """
    hours, r = divmod(int(ms), 3600000)
    minutes, r = divmod(r, 60000)
    seconds, ms = divmod(r, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def fmt_vtt(ms):
    """
    把毫秒转成 VTT 字幕格式的时间字符串。

    VTT 时间格式：HH:MM:SS.mmm
    和 SRT 的唯一区别：毫秒前用 . 代替 ,
    SRT: 01:01:01,000
    VTT: 01:01:01.000
    """
    hours, r = divmod(int(ms), 3600000)
    minutes, r = divmod(r, 60000)
    seconds, ms = divmod(r, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _list_available_engines():
    """
    扫描 engine/ 子目录，根据文件名列出可用的引擎。

    命名约定：engine/{引擎名}/engine_{引擎名}.py
    比如 engine/funasr/engine_funasr.py → 引擎名 "funasr"

    只检查文件是否存在，不导入任何 Python 代码。
    """
    engines = {}
    base_dir = os.path.join(os.path.dirname(__file__), "engine")
    if not os.path.isdir(base_dir):
        return engines
    for entry in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, entry)
        if os.path.isdir(subdir):
            expected_file = os.path.join(subdir, f"engine_{entry}.py")
            if os.path.exists(expected_file):
                engines[entry] = expected_file
    return engines


def _load_engine_class(engine_name):
    """
    根据引擎名加载对应的引擎类。

    只导入指定引擎的文件，不扫描其他引擎代码。
    """
    base_dir = os.path.join(os.path.dirname(__file__), "engine")
    engine_file = os.path.join(base_dir, engine_name, f"engine_{engine_name}.py")

    if not os.path.exists(engine_file):
        return None

    spec = importlib.util.spec_from_file_location(f"engine_{engine_name}", engine_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if issubclass(cls, AsrEngine) and cls is not AsrEngine:
            return cls
    return None


def _split_result_sentences(result):
    """
    把 AsrResult 中跨度太长的句子拆分成多个短句。

    FireRed 引擎靠 VAD 切出 27 个自然语音段，每段 1-3 秒。
    FunASR 引擎的 VAD 经常只返回 1 个整段，时间戳覆盖整段音频。
    这种长句子做字幕时几十秒才显示一句，没法用。

    这个函数做得更细：
    1. 先按句末标点（。！？.!?）拆分
    2. 超过 20 字的长段，再按逗号（，、;）拆分
    3. 按字数比例估算每句话的时间戳

    ------ 参数 ------
    result：AsrResult 对象，直接修改它的 sentences 和 words
    """
    import re

    def _split_text_into_parts(text):
        """先把文本按句末标点切，长段再按逗号切"""
        parts = re.split(r'(?<=[。！？.!?])', text)
        parts = [p.strip() for p in parts if p.strip()]

        result_parts = []
        for p in parts:
            if len(p) > 20:
                sub = re.split(r'(?<=[，、；;,])', p)
                sub = [s.strip() for s in sub if s.strip()]
                result_parts.extend(sub)
            else:
                result_parts.append(p)
        return result_parts

    new_sentences = []
    new_words = []
    word_idx = 0

    for s in result.sentences:
        duration = s["end_ms"] - s["start_ms"]
        text = s["text"]

        # 只有跨度超过 5 秒时才拆
        if duration <= 5000:
            new_sentences.append(s)
            seg_word_count = _count_words_for_sentence(s, result.sentences, result.words)
            for _ in range(seg_word_count):
                if word_idx < len(result.words):
                    new_words.append(result.words[word_idx])
                    word_idx += 1
            continue

        parts = _split_text_into_parts(text)
        if len(parts) <= 1:
            new_sentences.append(s)
            seg_word_count = _count_words_for_sentence(s, result.sentences, result.words)
            for _ in range(seg_word_count):
                if word_idx < len(result.words):
                    new_words.append(result.words[word_idx])
                    word_idx += 1
            continue

        total_chars = len(text)
        char_offset = 0

        for part in parts:
            part_len = len(part)
            if part_len == 0:
                continue

            start_ratio = char_offset / total_chars
            end_ratio = (char_offset + part_len) / total_chars

            total_words = len(result.words)
            w_start = int(start_ratio * total_words) if total_words > 0 else 0
            w_end = int(end_ratio * total_words) if total_words > 0 else 0
            if w_end >= total_words:
                w_end = total_words - 1
            if w_end < w_start:
                w_end = w_start

            if total_words > 0 and w_start < total_words:
                new_sentences.append({
                    "start_ms": result.words[w_start]["start_ms"],
                    "end_ms": result.words[w_end]["end_ms"],
                    "text": part,
                })
            else:
                new_sentences.append({
                    "start_ms": s["start_ms"],
                    "end_ms": s["end_ms"],
                    "text": part,
                })

            for i in range(w_start, w_end + 1):
                if i < len(result.words):
                    new_words.append(result.words[i])

            char_offset += part_len

    if new_sentences != result.sentences:
        result.sentences = new_sentences
        result.words = new_words


def _count_words_for_sentence(sentence, sentences, words):
    """估算一个句子对应的 words 数量"""
    if not words or not sentences:
        return 0
    idx = sentences.index(sentence) if sentence in sentences else -1
    if idx < 0:
        return 0
    total = len(words)
    seg_count = len(sentences)
    base = total // seg_count
    remainder = total % seg_count
    return base + (1 if idx < remainder else 0)


def main():
    """
    主函数——程序的真正入口。

    引擎加载方式：
    扫描 engine/ 子目录，根据 engine_{引擎名}.py 文件名判断可用引擎。
    只导入用户选择的引擎文件，不扫描其他引擎代码。
    """
    available = _list_available_engines()
    if not available:
        print("错误：未找到任何引擎文件 (engine/*/engine_*.py)")
        sys.exit(1)

    # ------ 第 2 步：解析命令行参数 ------
    # argparse 是 Python 自带的命令行参数解析器
    # 用户可以在命令行传参数，覆盖默认值
    parser = argparse.ArgumentParser(
        description="ASRGo 语音识别系统 — 音频转文字并导出字幕"
    )

    # --audio：音频文件路径
    # default 设了一个合理的默认值，不传参也能直接跑
    # 用户可以用 --audio sample/dp.wav 指定其他文件
    parser.add_argument("--audio", "-a",
                        default=os.path.join(os.path.dirname(__file__),
                                             "sample", "input.wav"),
                        help="音频文件路径（推荐 16kHz WAV 格式）")

    # --output：输出目录
    # 最终的输出路径是 output/{引擎}/{音频名}/transcript.*
    parser.add_argument("--output", "-o",
                        default="output",
                        help="输出根目录")

    # --device：用 CPU 还是 GPU
    # "cuda:0" = 第 1 块 NVIDIA 显卡
    # "cpu"    = 只用 CPU（没显卡也能跑，慢很多）
    parser.add_argument("--device",
                        default="cuda:0",
                        help="运行设备（cuda:0 / cpu）")

    # --engine：选择哪个引擎
    # choices 来自文件名扫描结果，只检查文件存在性，不导入代码
    parser.add_argument("--engine",
                        choices=list(available.keys()),
                        default=list(available.keys())[0],
                        help="选择引擎")

    # parse_args() 读取真实的命令行输入
    # 如果用户什么都没传，全部用上面的 default 值
    args = parser.parse_args()

    # ------ 第 3 步：检查音频文件 ------
    # os.path.exists 判断文件或目录是否存在
    # 如果文件不存在，输出错误并退出，避免后面报奇怪的错误
    if not os.path.exists(args.audio):
        print(f"错误：音频文件不存在 — {args.audio}")
        print("请用 --audio 参数指定正确的音频文件路径。")
        sys.exit(1)

    # ------ 第 4 步：准备输出目录 ------
    # os.path.basename 从路径中提取文件名部分
    #   比如 "/home/audio/sample/dp.wav" → "dp.wav"
    # os.path.splitext 把文件名和扩展名分开
    #   比如 "dp.wav" → ("dp", ".wav")
    # 最终 audio_name = "dp"
    audio_name = os.path.splitext(os.path.basename(args.audio))[0]
    output_dir = os.path.join(args.output, args.engine, audio_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------ 第 5 步：导入引擎类并创建实例 ------
    # 只导入用户指定的引擎文件，不扫描其他引擎
    # 真正的模型加载在 __init__ 构造方法里
    # 这一步最慢，FireRed 引擎约 10~30 秒
    EngineClass = _load_engine_class(args.engine)
    if EngineClass is None:
        print(f"错误：引擎 {args.engine} 加载失败")
        sys.exit(1)
    print(f"加载引擎：{args.engine}")
    engine = EngineClass(device=args.device)

    # 调用引擎的 transcribe() 方法做语音识别
    # 返回 AsrResult 对象，包含 text / sentences / words
    print(f"识别：{args.audio}")
    result = engine.transcribe(args.audio)

    # 有些引擎（如 FunASR）返回的句子时间跨度太长
    # 按句末标点拆分成多条，字幕才实用
    _split_result_sentences(result)

    # ------ 第 6 步：导出 4 种格式 ------
    # 所有文件都放在 output/{引擎}/{音频名}/ 目录下

    # TXT — 纯文本，每句一行，段落间空行（和 SRT 换行规则一致）
    txt_path = os.path.join(output_dir, "transcript.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(result.sentences):
            if i > 0:
                f.write("\n")
            f.write(s["text"] + "\n")

    # SRT — SubRip 字幕格式，最通用的字幕格式
    # 支持导入剪映、Premiere Pro、PotPlayer 等
    srt_path = os.path.join(output_dir, "transcript.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(result.sentences):
            start = fmt_srt(s["start_ms"])
            end = fmt_srt(s["end_ms"])
            f.write(f"{i+1}\n{start} --> {end}\n{s['text']}\n\n")

    # VTT — WebVTT 网页字幕格式
    # 主要用于 HTML5 video 标签
    # 和 SRT 的区别：毫秒分隔符是 . 不是 ,，没有序号
    vtt_path = os.path.join(output_dir, "transcript.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in result.sentences:
            start = fmt_vtt(s["start_ms"])
            end = fmt_vtt(s["end_ms"])
            f.write(f"{start} --> {end}\n{s['text']}\n\n")

    # JSON — 结构化数据格式
    # 包含 text / sentences / words 全部信息
    # 方便其他程序读取和处理
    json_path = os.path.join(output_dir, "transcript.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    # 打印结果摘要，告诉用户文件都生成在哪里了
    print(f"\n输出目录：{output_dir}")
    print(f"  transcript.txt  — 纯文本 ({len(result.text)} 字)")
    print(f"  transcript.srt  — SRT 字幕 ({len(result.sentences)} 条)")
    print(f"  transcript.vtt  — VTT 字幕")
    print(f"  transcript.json — JSON 数据")


if __name__ == "__main__":
    """
    Python 文件的入口点。

    每个 .py 文件都有一个内置变量 __name__：
    - 直接运行时（python asrgo_export.py）
      __name__ 被 Python 设为 "__main__"
    - 被其他文件导入时（from asrgo_export import xxx）
      __name__ 设为文件名 "asrgo_export"

    if __name__ == "__main__" 的作用：
    只有直接运行这个文件时才执行 main()。
    被其他文件 import 时，只提供函数和变量，不会自动执行 main()。
    """
    main()
