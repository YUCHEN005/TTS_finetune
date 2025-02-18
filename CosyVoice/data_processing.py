from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, load_wav_16k
from transformers import AutoTokenizer, AutoModel
import torch, torchaudio
import json, time, os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, device=device)

wav_scp_path = "data/wav.scp"
text_path = "data/text"
save_path = 'data.pt'

### data processing and save
dataset, idx = [], 0
with open(wav_scp_path, "r") as w_file, open(text_path, "r") as t_file:
    for w_line, t_line in zip(w_file.readlines(), t_file.readlines()):
        source = w_line.strip().split()[1]
        text = ' '.join(t_line.strip().split()[1:])
        speech = load_wav(source, 16000)

        dp = cosyvoice.frontend.data_processing(text, speech)
        if dp is not None:  
            dataset.append(dp)
        
torch.save(dataset, save_path)

