# Finetune Your Own TTS Model based on CosyVoice2

This repo is built based on **CosyVoice2** ([Paper](https://arxiv.org/abs/2412.10117); [Modelscope](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B); [HuggingFace](https://huggingface.co/spaces/FunAudioLLM/CosyVoice2-0.5B))

There are 4 supported languages: Chinese, English, Japanese, and Korean.


## Conda Environment Configuration

- Update submodules
``` sh
git submodule update --init --recursive
```

- Create Conda env:

``` sh
conda create -n tts_finetune -y python=3.10
conda activate tts_finetune
# pynini is required by WeTextProcessing, use conda to install it as it can be executed on all platform.
conda install -y -c conda-forge pynini==2.1.5
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

- Download pretrained models

``` python
# Option 1: SDK download
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```

``` sh
# Option 2: git download，make sure you have already installed git-lfs (https://stackoverflow.com/questions/48734119/git-lfs-is-not-a-git-command-unclear)
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd

# Then put the cosyvoice.yaml into pretrained_models/CosyVoice2-0.5B
mv ./cosyvoice.yaml pretrained_models/CosyVoice2-0.5B
```

Optionally, you can unzip `ttsfrd` resouce and install `ttsfrd` package for better text normalization performance.

Notice that this step is not necessary. If you do not install `ttsfrd` package, we will use WeTextProcessing by default.

``` sh
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

- Import third-party modules:

Run the following command before running any experiments:

``` sh
export PYTHONPATH=$(pwd)/third_party/Matcha-TTS
```

## Data preprocessing

Before finetuning CosyVoice2, we need to process both text and speech data into discrete tokens, and then save them for subsequent training process.

Please enter the file `data_processing.py` and set your original data files, including `wav.scp` and `text`. Then specifiy the saved file path.

Run the following command and the text & speech data will be saved as `.pt`.

```sh
torchrun --nproc_per_node=1 data_processing.py
```

## Model training

Please enter the file `train.py` and specify your training data (line 72), and modify the hyperparameters if needed. Then run the following command to launch distributed training:

```sh
torchrun --nproc_per_node=8 train.py > train.log
```

The trained checkpoints will be saved under `results/`

## Model inference

Please enter the file `infer.py`, specify your trained weights and testing sample (including prompt speech, prompt text, and target speech). Or you can write a loop to evaluate your own test set.

Then run the following command to launch inference:

```sh
torchrun --nproc_per_node=1 infer.py > infer.log
```

