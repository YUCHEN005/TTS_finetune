import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, load_wav_16k
from transformers import AutoModel, AutoTokenizer
import torch, torchaudio
import json, time, logging
from argparse import ArgumentParser

# initialize cosyvoice model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, device=device)

# load finetuned checkpoint
ft_ckpt = torch.load('results/epoch_020.pth')
new_ft_ckpt = {k.replace("module.", ""): v for k, v in ft_ckpt.items()}
cosyvoice.model.llm.load_state_dict(new_ft_ckpt, strict=True)
print(f'loaded finetuned checkpoint!')

# test sample
prompt_speech = load_wav('p232_007.wav', 16000)
prompt_text = "The rainbow is a division of white light into many beautiful colors."
text = 'We all have a dream, where the world lives in peace forever'

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=False)):
    torchaudio.save('result_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
