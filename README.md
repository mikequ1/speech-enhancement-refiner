# Noise Robustness Project

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
