# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TC-CMGAN \
#     --datadir ./data/urgent2024/Tcorrected_CMGAN_nb_wav \
#     --cleandir ../NRSER2/data/urgent2024/clean_wav \
#     --list urgent2024.txt \
#     --txtfiledir ../NRSER2/data/urgent2024/txtfiles

# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TFC-CMGAN \
#     --datadir ./data/urgent2024/TFcorrected_CMGAN_nb_wav \
#     --cleandir ../NRSER2/data/urgent2024/clean_wav \
#     --list urgent2024.txt \
#     --txtfiledir ../NRSER2/data/urgent2024/txtfiles

# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TC-SEMamba \
#     --datadir ./data/urgent2024/Tcorrected_SEMamba_nb_wav \
#     --cleandir ../NRSER2/data/urgent2024/clean_wav \
#     --list urgent2024.txt \
#     --txtfiledir ../NRSER2/data/urgent2024/txtfiles

# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TFC-SEMamba \
#     --datadir ./data/urgent2024/TFcorrected_SEMamba_nb_wav \
#     --cleandir ../NRSER2/data/urgent2024/clean_wav \
#     --list urgent2024.txt \
#     --txtfiledir ../NRSER2/data/urgent2024/txtfiles



# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TC-CMGAN \
#     --datadir ./data/urgent2025/Tcorrected_CMGAN_nb_wav \
#     --cleandir ../NRSER2/data/urgent2025/clean_wav \
#     --list urgent2025.txt \
#     --txtfiledir ../NRSER2/data/urgent2025/txtfiles

# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TFC-CMGAN \
#     --datadir ./data/urgent2025/TFcorrected_CMGAN_nb_wav \
#     --cleandir ../NRSER2/data/urgent2025/clean_wav \
#     --list urgent2025.txt \
#     --txtfiledir ../NRSER2/data/urgent2025/txtfiles

# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TC-SEMamba \
#     --datadir ./data/urgent2025/Tcorrected_SEMamba_nb_wav \
#     --cleandir ../NRSER2/data/urgent2025/clean_wav \
#     --list urgent2025.txt \
#     --txtfiledir ../NRSER2/data/urgent2025/txtfiles

# CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
#     --model TFC-SEMamba \
#     --datadir ./data/urgent2025/TFcorrected_SEMamba_nb_wav \
#     --cleandir ../NRSER2/data/urgent2025/clean_wav \
#     --list urgent2025.txt \
#     --txtfiledir ../NRSER2/data/urgent2025/txtfiles

CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
    --model enhanced-SGMSE \
    --datadir ./data/urgent2025/enhanced_SGMSE_nb_wav \
    --cleandir ../NRSER2/data/urgent2025/clean_wav \
    --list urgent2025.txt \
    --txtfiledir ../NRSER2/data/urgent2025/txtfiles

CUDA_VISIBLE_DEVICES=1 python speech_evaluation.py \
    --model TC-SGMSE \
    --datadir ./data/urgent2025/Tcorrected_SGMSE_nb_wav \
    --cleandir ../NRSER2/data/urgent2025/clean_wav \
    --list urgent2025.txt \
    --txtfiledir ../NRSER2/data/urgent2025/txtfiles