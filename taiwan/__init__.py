"""台灣念法：把中文文字轉成送進 VoxCPM2 的文字，讓它照台灣的念法念。

教育部《重編國語辭典修訂本》＋g2pW（看上下文判斷破音字）＋台灣日常念法決定每個字怎麼念；
跟模型預設念法（大陸念法）不同的字寫成 {拼音}，其餘轉成簡體（模型幾乎只學過簡體，繁體字會念錯、甚至跑成粵語）。

tw_reading.py 是 jt-live-whisper（github.com/ChimingLu/jt-live-whisper，commit 4ac4e6d，Apache-2.0）的
jtlw_tts/tw_reading.py，原封不動複製；要更新就整份重新複製，不要在這裡改它。

    from taiwan import load, spoken
    R = load()                       # 第一次用之前先跑 python -m taiwan.fetch_data
    spoken(R, "垃圾要分類")           # → "{le4}{se4}要分类"
"""
import os
import re

from . import tw_reading as T

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STYLE = "用標準的台灣國語普通話"      # 加在控制指令裡，進一步避免念成粵語

# 實際配音時 VoxCPM2 念得怪的字（以前用同音字硬換：側→測、築→住、桿→趕、縮→梭）：一律加念法提示
T._TTS_ALWAYS_HINT = T._TTS_ALWAYS_HINT | frozenset("側築桿縮")

HINT = re.compile(r"\{[a-z]+[1-5]\}")


def load(data_dir=DATA_DIR, model_dir=None):
    """載入辭典、g2pW 等資源（約 10 秒）。model_dir：VoxCPM2 模型資料夾，給了的話，模型詞彙裡沒有的字一律加念法提示"""
    if not os.path.exists(os.path.join(data_dir, "moe_words.tsv")):
        raise FileNotFoundError(f"找不到台灣念法資源（{data_dir}），請先在 VoxCPM 資料夾執行：python -m taiwan.fetch_data")
    R = T._tts_load_text(data_dir)
    if model_dir:
        R["vocab"] = T._tts_vocab_chars(os.path.join(model_dir, "tokenizer.json"))
    return R


def spoken(R, text, custom=None):
    """原文 → 送進模型的文字。custom：自訂發音 {"和": "ㄏㄢˋ", "垃圾": "ㄌㄜˋ ㄙㄜˋ"}，比辭典優先。
    送進模型時要關掉 VoxCPM 的文字規範化（normalize=False），不然 {拼音} 會被改掉"""
    return T._tts_spoken(R, text, custom)


def plain(R, text):
    """跟 spoken() 同樣的數字與簡體轉換、但不加 {拼音}：拿來跟語音辨識的結果比對錯字率"""
    return R["t2s"].convert(T._tts_numbers(text))


def split(text, limit=80):
    """切句：句末標點與換行，太長再依逗號切（不切斷數字與英文）"""
    return T._tts_split(text, limit)
