"""用台灣念法把文字念成 wav。在 VoxCPM 資料夾執行：

    python -m taiwan.say "垃圾要分類，星期三收。" out.wav
    python -m taiwan.say --file 稿子.txt out.wav --ref 我的聲音.wav --style "充滿活力、語速稍快"
    python -m taiwan.say "..." out.wav --ref 我的聲音.wav --check      # 每句用 SenseVoice 聽一次，念成粵語或錯字多就換 seed 重念

長文字會先切句，每句分開念再接起來。--ref 只用參考錄音的音色（不傳逐字稿，語氣才不會照抄原錄音）。
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

from . import STYLE, load, plain, spoken, split

SEEDS = (42, 7, 123, 2024, 99)


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


def han(x):
    return re.sub(r"[^一-鿿]", "", x.translate(str.maketrans("0123456789", "零一二三四五六七八九")))


def main():
    ap = argparse.ArgumentParser(description="VoxCPM2 台灣念法朗讀")
    ap.add_argument("text", nargs="?", help="要念的文字（或用 --file）")
    ap.add_argument("out", help="輸出的 wav")
    ap.add_argument("--file", help="從文字檔讀")
    ap.add_argument("--ref", help="參考錄音（只取音色）")
    ap.add_argument("--style", default="", help="風格指示，例如「充滿活力、開朗熱情，語速稍快」；前面會自動加上「用標準的台灣國語普通話」")
    ap.add_argument("--model", default="openbmb/VoxCPM2", help="模型（Hugging Face 名稱或本機資料夾）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--custom", help="自訂發音 JSON 檔，例如 {\"和\": \"ㄏㄢˋ\", \"垃圾\": \"ㄌㄜˋ ㄙㄜˋ\"}")
    ap.add_argument("--check", action="store_true", help="用 SenseVoice 檢查每句：不是華語或錯字率超過 --max-cer 就換 seed 重念")
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--show", action="store_true", help="只印出轉換後送進模型的文字，不念")
    a = ap.parse_args()

    text = open(a.file, encoding="utf-8").read() if a.file else a.text
    if not text:
        ap.error("請給要念的文字或 --file")
    custom = json.load(open(a.custom, encoding="utf-8")) if a.custom else None

    from voxcpm import VoxCPM
    from voxcpm.core import resolve_model_path

    if os.path.isdir(a.model):
        model_dir = a.model
    elif a.show:                               # 只看轉換結果：模型已經下載過才拿它的詞彙表，不為了這個去下載
        try:
            model_dir = resolve_model_path(a.model, local_files_only=True)
        except Exception:
            model_dir = None
    else:
        model_dir = resolve_model_path(a.model)
    R = load(model_dir=model_dir)
    sents = split(text)
    if a.show:
        for s in sents:
            print(spoken(R, s, custom))
        return

    model = VoxCPM.from_pretrained(model_dir, load_denoiser=False)
    sr = model.tts_model.sample_rate
    style = "(" + "，".join(x for x in (STYLE, re.sub(r"[()（）]", "", a.style).strip()) if x) + ")"
    asr = None
    if a.check:
        from funasr import AutoModel
        asr = AutoModel(model="iic/SenseVoiceSmall", disable_update=True, device="cpu")

    parts = []
    for i, s in enumerate(sents, 1):
        sent = spoken(R, s, custom)
        best = None
        for seed in ((a.seed,) + tuple(x for x in SEEDS if x != a.seed)) if a.check else (a.seed,):
            wav = model.generate(text=style + sent, reference_wav_path=a.ref, cfg_value=a.cfg,
                                 inference_timesteps=a.steps, normalize=False, seed=seed)
            if not asr:
                best = (0, wav, "")
                break
            tmp = a.out + ".check.wav"
            sf.write(tmp, wav, sr)
            r = asr.generate(input=tmp, language="auto", use_itn=False)[0]["text"]
            os.remove(tmp)
            m = re.match(r"<\|(\w+)\|>", r)
            lang = m.group(1) if m else "?"
            want = han(plain(R, s))
            cer = edit_distance(want, han(R["t2s"].convert(re.sub(r"<\|[^|]*\|>", "", r)))) / max(1, len(want))
            score = cer + (0 if lang == "zh" else 1)
            note = f" seed {seed}：{lang} 錯字率 {cer:.2f}"
            if best is None or score < best[0]:
                best = (score, wav, note)
            if lang == "zh" and cer <= a.max_cer:
                break
        parts.append(best[1])
        print(f"[{i}/{len(sents)}]{best[2]} {sent}", flush=True)
    gap = np.zeros(int(sr * 0.25), dtype=np.float32)
    out = np.concatenate([x for p in parts for x in (p.astype(np.float32), gap)][:-1])
    sf.write(a.out, out, sr)
    print(f"{a.out}：{len(out) / sr:.1f} 秒")


if __name__ == "__main__":
    sys.exit(main())
