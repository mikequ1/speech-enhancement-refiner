mkdir -p data/urgent2024
mkdir -p data/urgent2025
mkdir -p data/urgent2026
mkdir -p data/VCTK-DEMAND

#########################################
# GOOGLE DRIVE DATASET DOWNLOAD HANDLER #
#########################################

IDS=(
  "1fvlrzw1K1YPZcC9GxY7nMNPIxu3-Dx78"
  "1Bcy7vtsZF5kzwrkklQwj6Um4AMlaUHe2"
  "1RarjxOgWkaDV8EjH_eLX169y89PVa3sg"
  "1rxV6RgA4LAp2I1EnHsln7wI7-UCP6Qer"
  "1lDgJmJkxBsxEV8sF58D9Fyk4aRY7dloV"
)

OUTFILES=(
  "data/urgent2024/clean_nb_flac.zip"
  "data/urgent2024/noisy_nb_flac.zip"
  "data/urgent2025/clean_nb_flac.zip"
  "data/urgent2025/noisy_nb_flac.zip"
  "data/urgent2026/clean_nb_flac.zip"
)

for i in "${!IDS[@]}"; do
    ID="${IDS[$i]}"
    OUTFILE="${OUTFILES[$i]}"
    mkdir -p "$(dirname "$OUTFILE")"
    echo "Downloading $ID → $OUTFILE"
    wget --no-check-certificate \
      "https://drive.usercontent.google.com/download?id=${ID}&export=download&confirm=t" \
      -O "$OUTFILE"
done

#################################
# WGET DATASET DOWNLOAD HANDLER #
#################################

IDS=(
  "https://huggingface.co/datasets/urgent-challenge/urgent2026_leaderboard/resolve/main/track1/nonblind_test_noisy.zip"
  "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
  "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"
)

OUTFILES=(
  "data/urgent2026/noisy_nb_flac.zip"
  "data/VCTK-DEMAND/clean_test.zip"
  "data/VCTK-DEMAND/noisy_test.zip"
)

for i in "${!IDS[@]}"; do
    ID="${IDS[$i]}"
    OUTFILE="${OUTFILES[$i]}"
    mkdir -p "$(dirname "$OUTFILE")"
    echo "Downloading $ID → $OUTFILE"
    wget --no-check-certificate \
      "$ID" \
      -O "$OUTFILE"
done



#####################
# EXTRACT ZIP FILES #
#####################

unzip -j -o data/urgent2024/clean_nb_flac.zip -d data/urgent2024/clean_nb_flac
unzip -j -o data/urgent2024/noisy_nb_flac.zip -d data/urgent2024/noisy_nb_flac
unzip -j -o data/urgent2025/clean_nb_flac.zip -d data/urgent2025/clean_nb_flac
unzip -j -o data/urgent2025/noisy_nb_flac.zip -d data/urgent2025/noisy_nb_flac
unzip -j -o data/urgent2026/clean_nb_flac.zip -d data/urgent2026/clean_nb_flac
unzip -j -o data/urgent2026/noisy_nb_flac.zip -d data/urgent2026/noisy_nb_flac
unzip -j -o data/VCTK-DEMAND/clean_test.zip -d data/VCTK-DEMAND/clean_test_wav
unzip -j -o data/VCTK-DEMAND/noisy_test.zip -d data/VCTK-DEMAND/noisy_test_wav


###########################################
# FLAC TO WAV CONVERSION / RESAMPLE 16KHZ #
###########################################

INS=(
    "data/urgent2024/clean_nb_flac"
    "data/urgent2024/noisy_nb_flac"
    "data/urgent2025/clean_nb_flac"
    "data/urgent2025/noisy_nb_flac"
    "data/urgent2026/clean_nb_flac"
    "data/urgent2026/noisy_nb_flac"
)
OUTS=(
    "data/urgent2024/clean_nb_wav"
    "data/urgent2024/noisy_nb_wav"
    "data/urgent2025/clean_nb_wav"
    "data/urgent2025/noisy_nb_wav"
    "data/urgent2026/clean_nb_wav"
    "data/urgent2026/noisy_nb_wav"
)

for i in "${!INS[@]}"; do
    IN="${INS[$i]}"
    OUT="${OUTS[$i]}"

    echo "Converting:"
    echo "  IN  = $IN"
    echo "  OUT = $OUT"

    mkdir -p "$OUT"

    for f in "$IN"/*.flac; do
        base=$(basename "$f" .flac)
        ffmpeg -y -i "$f" -ar 16000 -acodec pcm_s16le "$OUT/${base}.wav"
    done
done