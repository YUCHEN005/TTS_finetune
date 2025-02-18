import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, load_wav_16k
import torch, torchaudio
import json, time, logging
from argparse import ArgumentParser

## initialize cosyvoice model
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
## load finetuned checkpoint
# ft_ckpt = torch.load('/fs-computility/INTERN6/shared/yuchen/CosyVoice/results/epoch_020.pth')
# new_ft_ckpt = {k.replace("module.", ""): v for k, v in ft_ckpt.items()}
# cosyvoice.model.llm.load_state_dict(new_ft_ckpt, strict=True)
# print(f'loaded finetuned checkpoint!')

## dataset
with open("/fs-computility/INTERN6/zhangziyang/libriheavy/libriheavy_cuts_large.jsonl", "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        data = json.loads(line)

        if i == 6001:
            start = float(data.get("start"))
            duration = float(data.get("duration"))
            source = "/fs-computility/INTERN6/zhangziyang/libriheavy/" + data.get("recording", {}).get("sources", [{}])[0].get("source")
            text = data.get("supervisions", [{}])[0].get("custom", {}).get("texts", [None])[0]
            speech = load_wav_16k(source, start, duration)
            
            prompt_speech = speech[:, :124800]
            prompt_text = "What though the one that thou hast builded lies Where sinks the sun to its enchanted rest."
            torchaudio.save(f'speech.wav', speech, 16000)
            torchaudio.save(f'prompt_speech.wav', prompt_speech, 16000)
            # text = "If, on each breeze that bloweth east or west, To thee, on swiftest wing, my spirit flies"
            text = 'We all have a dream, where the world lives in peace forever, no wars, no criminal and no poverty'

            break

# prompt_speech = load_wav('p232_007.wav', 16000)
# prompt_text = "The rainbow is a division of white light into many beautiful colors."
# text = 'We all have a dream, where the world lives in peace forever, no wars, no criminal and no poverty'

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=False)):
    torchaudio.save('ft_result_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

