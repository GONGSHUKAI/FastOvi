<p align="center">
<h1 align="center">FastOvi</h1>
<a href="https://github.com/GONGSHUKAI/step_distillation"><img src="https://img.shields.io/badge/GitHub-Repository-0066cc.svg" alt="GitHub"></a>
<a href="https://huggingface.co/ShukaiGong/OviDMD"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<!-- <a href="https://huggingface.co/datasets/quanhaol/MagicData"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a> -->

## Configuration
```bash
conda create -n mediagen python=3.10 -y
conda activate mediagen
# It is recommended that you should install Flash Attention after installing other dependencies
pip install -r requirements.txt
pip install flash-attn==2.7.3 --use-pep517 --no-build-isolation
```

## Download weights

```bash
cd OviDMD
export HF_ENDPOINT=https://hf-mirror.com
# FastOvi checkpoint
huggingface-cli download ShukaiGong/OviDMD --local-dir ./weights --resume-download
# Other checkpoints
python ovi_weight_download.py
```
Your weights directory should look like
```
OviDMD
├── weights
│   ├── MMAudio
│   │   └── ext_weights
│   │       ├── best_netG.pt
│   │       └── v1-16.pth
│   ├── Wan2.2-TI2V-5B
│   │   ├── google
│   │   │   └── umt5-xxl
│   │   │       ├── special_tokens_map.json
│   │   │       ├── spiece.model
│   │   │       ├── tokenizer_config.json
│   │   │       └── tokenizer.json
│   │   ├── models_t5_umt5-xxl-enc-bf16.pth
│   │   └── Wan2.2_VAE.pth
│   └── model_ema.pt
```
## Inference
```bash
bash running_scripts/inference/i2av_fewstep.sh
```
