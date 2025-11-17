# Noise Robustness Project


## Environment Setup
```
conda create --name enhancement_correction python=3.10
conda activate enhancement_correction

conda install -c conda-forge ffmpeg
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torchcodec
pip install scoreq[gpu]
```

```
python main.py \
    --noisydir [PATH to the noisy wavs directory] \
    --enhanceddir [PATH to the enhanced wavs directory] \
    --txtfiledir [PATH to inference config file directory] \
    --list [File name of the config file] \
    --savedir [PATH to the output directory]
```

Example script:
```
CUDA_VISIBLE_DEVICES=3, python main.py         --noisydir ../NRSER2/data/urgent2024/noisy_wav         --enhanceddir ../NRSER2/data/urgent2024/SEMamba         --txtfiledir ../NRSER2/data/urgent2024/txtfiles         --list config.txt         --savedir data/urgent2025/SEMambaC
```

## Downloading Datasets
To replicate our the experiments shown in our paper:
