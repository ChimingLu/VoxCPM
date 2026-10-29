# 台灣念法（taiwan/）

VoxCPM2 主要學的是大陸念法，繁體字還可能念成粵語。這個資料夾讓它照台灣的念法念：

- **每個字怎麼念**：自訂發音 ＞ 台灣日常念法 ＞ 教育部《重編國語辭典修訂本》 ＞ g2pW（看上下文判斷破音字）。
  例如垃圾念ㄌㄜˋ ㄙㄜˋ、星期的期念ㄑㄧˊ、伺服器的伺念ㄙˋ、角色念ㄐㄧㄠˇ ㄙㄜˋ。
- **跟模型預設念法不同的字**寫成 `{拼音}` 送進模型（`{le4}{se4}`），其餘轉成簡體（模型幾乎只學過簡體）。
- 控制指令前面自動加上「用標準的台灣國語普通話」。
- 念法規則來自 [jt-live-whisper](https://github.com/ChimingLu/jt-live-whisper)（`tw_reading.py` 原封不動複製自它的 `jtlw_tts/tw_reading.py`，commit 4ac4e6d），
  那邊對 Common Voice 四千多句逐一比對、試聽調過。要更新就整份重新複製。

## 安裝（第一次）

在 VoxCPM 資料夾：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[taiwan]"
python -m taiwan.fetch_data      # 下載 g2pW（約 600 MB）與教育部辭典，放在 taiwan/data/
```

## 使用

**網頁介面**：`python app.py`，「🇹🇼 台灣念法」預設開啟。

**命令列**：

```bash
# 念一段話
python -m taiwan.say "垃圾要分類，星期三收。" out.wav

# 用自己的聲音（只取音色）、加風格、每句用 SenseVoice 檢查，念成粵語或音節不對（例如垃圾念成ㄌㄚ ㄐㄧ）就換 seed 重念
python -m taiwan.say --file 稿子.txt out.wav --ref 我的聲音.wav --style "充滿活力、開朗熱情，語調上揚，語速稍快" --check

# 只看送進模型的文字
python -m taiwan.say "角色扮演著重要的工作" x.wav --show
```

**自訂發音**：寫一個 JSON 檔，用 `--custom` 帶入，例如 `{"和": "ㄏㄢˋ", "垃圾": "ㄌㄜˋ ㄙㄜˋ"}`。

**在自己的程式裡**：

```python
import taiwan
R = taiwan.load()
model.generate(text="(用標準的台灣國語普通話)" + taiwan.spoken(R, "垃圾要分類"), normalize=False, ...)
```

`normalize` 一定要關掉，不然 `{拼音}` 會被文字規範化改掉。

## 授權

- g2pW：Apache-2.0。原本放在 `storage.googleapis.com/esun-ai`，那裡已經不在了，改從作者在 Hugging Face 的備份 `YC-Chen/g2pW` 下載並核對 SHA256。
- `moe_words.tsv`：由教育部《重編國語辭典修訂本》（g0v/moedict-data 整理）轉成「詞→讀音」對照，只做格式轉換。
  著作權屬教育部，創用 CC 姓名標示-禁止改作 3.0 臺灣。下載後只存在你的電腦（`taiwan/data/` 不進 git）。
