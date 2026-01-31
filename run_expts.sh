# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2024/noisy_nb_wav \
#     --enhanceddir ./data/urgent2024/enhanced_CMGAN_nb_wav \
#     --listpath ../NRSER2/data/urgent2024/txtfiles/config-test.txt \
#     --savedir ./data/urgent2024/Tcorrected_CMGAN_nb_wav \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2024/noisy_nb_wav \
#     --enhanceddir ./data/urgent2024/enhanced_CMGAN_nb_wav \
#     --listpath ../NRSER2/data/urgent2024/txtfiles/config.txt \
#     --savedir ./data/urgent2024/TFcorrected_CMGAN_nb_wav \
#     --tf

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2024/noisy_nb_wav \
#     --enhanceddir ./data/urgent2024/enhanced_SEMamba_nb_wav \
#     --listpath ../NRSER2/data/urgent2024/txtfiles/config.txt \
#     --savedir ./data/urgent2024/Tcorrected_SEMamba_nb_wav \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2024/noisy_nb_wav \
#     --enhanceddir ./data/urgent2024/enhanced_SEMamba_nb_wav \
#     --listpath ../NRSER2/data/urgent2024/txtfiles/config.txt \
#     --savedir ./data/urgent2024/TFcorrected_SEMamba_nb_wav \
#     --tf

CUDA_VISIBLE_DEVICES=3 python main.py \
    --noisydir ./data/urgent2024/noisy_nb_wav \
    --enhanceddir ./data/urgent2024/enhanced_SGMSE_nb_wav \
    --listpath ../NRSER2/data/urgent2024/txtfiles/config.txt \
    --savedir ./data/urgent2024/Tcorrected_SGMSE_nb_wav \

CUDA_VISIBLE_DEVICES=3 python main.py \
    --noisydir ./data/urgent2024/noisy_nb_wav \
    --enhanceddir ./data/urgent2024/enhanced_SGMSE_nb_wav \
    --listpath ../NRSER2/data/urgent2024/txtfiles/config.txt \
    --savedir ./data/urgent2024/TFcorrected_SGMSE_nb_wav \
    --tf



# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2025/noisy_nb_wav \
#     --enhanceddir ./data/urgent2025/enhanced_CMGAN_nb_wav \
#     --listpath ../NRSER2/data/urgent2025/txtfiles/config.txt \
#     --savedir ./data/urgent2025/Tcorrected_CMGAN_nb_wav \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2025/noisy_nb_wav \
#     --enhanceddir ./data/urgent2025/enhanced_CMGAN_nb_wav \
#     --listpath ../NRSER2/data/urgent2025/txtfiles/config.txt \
#     --savedir ./data/urgent2025/TFcorrected_CMGAN_nb_wav \
#     --tf

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2025/noisy_nb_wav \
#     --enhanceddir ./data/urgent2025/enhanced_SEMamba_nb_wav \
#     --listpath ../NRSER2/data/urgent2025/txtfiles/config.txt \
#     --savedir ./data/urgent2025/Tcorrected_SEMamba_nb_wav \

# CUDA_VISIBLE_DEVICES=1 python main.py \
#     --noisydir ./data/urgent2025/noisy_nb_wav \
#     --enhanceddir ./data/urgent2025/enhanced_SEMamba_nb_wav \
#     --listpath ../NRSER2/data/urgent2025/txtfiles/config.txt \
#     --savedir ./data/urgent2025/TFcorrected_SEMamba_nb_wav \
#     --tf

CUDA_VISIBLE_DEVICES=3 python main.py \
    --noisydir ./data/urgent2025/noisy_nb_wav \
    --enhanceddir ./data/urgent2025/enhanced_SGMSE_nb_wav \
    --listpath ../NRSER2/data/urgent2025/txtfiles/config.txt \
    --savedir ./data/urgent2025/Tcorrected_SGMSE_nb_wav \

CUDA_VISIBLE_DEVICES=3 python main.py \
    --noisydir ./data/urgent2025/noisy_nb_wav \
    --enhanceddir ./data/urgent2025/enhanced_SGMSE_nb_wav \
    --listpath ../NRSER2/data/urgent2025/txtfiles/config.txt \
    --savedir ./data/urgent2025/TFcorrected_SGMSE_nb_wav \
    --tf
