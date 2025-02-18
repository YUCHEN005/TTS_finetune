from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio, torch, os, glob
## export PYTHONPATH=/fs-computility/INTERN6/shared/yuchen/CosyVoice_qwen0.5b/third_party/Matcha-TTS

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=True)):
    torchaudio.save('zero_shot_stream_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

audio_files = sorted(glob.glob("zero_shot_stream_*.wav"))
combined = []
for file in audio_files:
    audio = torchaudio.load(file)[0]
    combined.append(audio)
os.system('rm -rf zero_shot_stream_*.wav')
torchaudio.save('zero_shot_stream_0.wav', torch.cat(combined, dim=-1), cosyvoice.sample_rate)
