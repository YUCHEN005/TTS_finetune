import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, load_wav_16k
import torch, torchaudio
import json, time, logging
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--shard", type=int, default=0)
args = parser.parse_args()

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)

### data processing and save
dataset, dur, idx = [], 0, 1
with open("/fs-computility/INTERN6/zhangziyang/libriheavy/libriheavy_cuts_large.jsonl", "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        data = json.loads(line)

        start = float(data.get("start"))
        duration = float(data.get("duration"))
        source = "/fs-computility/INTERN6/zhangziyang/libriheavy/" + data.get("recording", {}).get("sources", [{}])[0].get("source")
        text = data.get("supervisions", [{}])[0].get("custom", {}).get("texts", [None])[0]
        speech = load_wav_16k(source, start, duration)

        dp = cosyvoice.frontend.frontend_data_processing(text, speech)
        if dp is not None:  
            dataset.append(dp)
            dur += duration
        
        if len(dataset) == 100000:
            logging.info(f'processed {len(dataset)} samples, duration = {dur//3600} h')
            torch.save(dataset, f'Libriheavy_cosy/proc_libriheavy_{idx:03d}_{dur//3600}h.pt')
            del dataset; dataset = []; dur = 0
            idx += 1

