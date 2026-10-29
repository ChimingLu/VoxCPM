"""下載台灣念法要用的資源到 taiwan/data/（已經有的不重下）：
- G2PWModel/：g2pW 模型（約 600 MB，判斷破音字），Apache-2.0。原本的 storage.googleapis.com/esun-ai 已經不在了，
  改從作者在 Hugging Face 的備份（YC-Chen/g2pW）下載並核對 SHA256
- bert-base-chinese/：g2pW 用的分詞器
- moe_words.tsv：教育部《重編國語辭典修訂本》（g0v/moedict-data）轉成「詞→讀音」對照。
  著作權屬教育部，CC BY-ND 3.0 TW；只做格式轉換
用法（在 VoxCPM 資料夾）：python -m taiwan.fetch_data [資料夾，預設 taiwan/data]"""
import hashlib
import json
import lzma
import os
import re
import shutil
import sys
import urllib.request
import zipfile

from .tw_reading import _TTS_HAN, _TTS_MAX_WORD, _tts_syl

G2PW_URL = "https://huggingface.co/YC-Chen/g2pW/resolve/fb4784f7dce8466a53dd4f3552f5a4c5b657b90c/G2PWModel-v2-onnx.zip"
G2PW_SHA256 = "699f3c1fd7fb0e2c2d49ed2486826fd5bff233fee7759350a91c3b49aedc4ed2"
BERT = ("google-bert/bert-base-chinese", "8f23c25b06e129b6c986331a13d8d025a92cf0ea")
MOE_URL = "https://raw.githubusercontent.com/g0v/moedict-data/a6dc997417507eb510fc29822bc514de2c92728c/dict-revised.json.xz"


def moe_lines(entries):
    """jt-live-whisper remote_whisper_server.py 的 _tts_moe_lines：只留 2～8 個漢字的詞，同一詞的多個讀音全留"""
    rows = {}
    for e in entries:
        t = e.get("title", "")
        if not (2 <= len(t) <= _TTS_MAX_WORD) or not _TTS_HAN.fullmatch(t):
            continue
        for h in e.get("heteronyms") or []:
            b = re.split(r"[（(]", h.get("bopomofo") or "")[0].strip()
            syl = [x for x in re.split(r"[\s　]+", b) if x]
            if len(syl) == len(t):
                r = " ".join(_tts_syl(x) for x in syl)
                if r not in rows.setdefault(t, []):
                    rows[t].append(r)
    return [t + "\t" + "|".join(rs) for t, rs in rows.items() if rs]


def download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dest)
    return dest


def main(d):
    os.makedirs(d, exist_ok=True)
    g = os.path.join(d, "G2PWModel")
    if not os.path.exists(os.path.join(g, "version")):
        print("下載 g2pW 模型（約 600 MB）...", flush=True)
        z = os.path.join(d, "G2PWModel.zip")
        if not os.path.exists(z):
            download(G2PW_URL, z)
        h = hashlib.sha256()
        with open(z, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != G2PW_SHA256:
            os.remove(z)
            sys.exit(f"g2pW 壓縮檔的 SHA256 不符（{h.hexdigest()}），已刪除，請重跑")
        shutil.rmtree(g, ignore_errors=True)
        with zipfile.ZipFile(z) as zf:
            top = zf.namelist()[0].split("/")[0]
            zf.extractall(d)
        if top != "G2PWModel":
            os.replace(os.path.join(d, top), g)
        os.remove(z)
    b = os.path.join(d, "bert-base-chinese")
    os.makedirs(b, exist_ok=True)
    for f in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "config.json"):
        if not os.path.exists(os.path.join(b, f)):
            download(f"https://huggingface.co/{BERT[0]}/resolve/{BERT[1]}/{f}", os.path.join(b, f))
    m = os.path.join(d, "moe_words.tsv")
    if not os.path.exists(m):
        print("下載教育部《重編國語辭典修訂本》（約 15 MB）並轉成讀音對照...", flush=True)
        x = download(MOE_URL, os.path.join(d, "dict-revised.json.xz"))
        with lzma.open(x, "rt", encoding="utf-8") as f:
            lines = moe_lines(json.load(f))
        with open(m + ".tmp", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(m + ".tmp", m)
        os.remove(x)
        print(f"教育部辭典對照檔：{len(lines)} 個詞")
    print(f"資源就緒：{d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
