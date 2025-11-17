import sys
import os
sys.path.append('./src')
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import soundfile as sf
from aligner import Aligner
from mixer import TMixer, TFMixer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
NOISY_WAV_PATH = './tests/test_noisy_audios.txt'
ENHANCED_WAV_PATH = './tests/test_enhanced_audios.txt'
TEMP_OUT_PATH = './temp'

def load_test_audio(audio_path):
    test_wavs = []
    with open(audio_path, 'r') as file:
        for line in file:
            test_wavs.append(line.rstrip('\n'))
    return test_wavs

# Whisper-Timestamped Aligner Tests
def test_whisper_timestamped():
    noisy_wav_paths = load_test_audio(NOISY_WAV_PATH)
    enhanced_wav_paths = load_test_audio(ENHANCED_WAV_PATH)
    model_name = 'whisper-timestamped'
    wt_aligner = Aligner(model=model_name, device=DEVICE)
    
    noisy_wavs = []
    enhanced_wavs = []
    intervals = []
    metadata = [] # file name, original wav lengths
    for i in range(len(noisy_wav_paths)):
        noisy_wav_path, enhanced_wav_path = noisy_wav_paths[i], enhanced_wav_paths[i]
        print(f'{i+1}/{len(noisy_wav_paths)} : {noisy_wav_path} | {enhanced_wav_path}')
        wav_noi, _ = torchaudio.load(noisy_wav_path)
        wav_enh, _ = torchaudio.load(enhanced_wav_path)
        wav_noi = wav_noi.squeeze(0).to(DEVICE)
        wav_enh = wav_enh.squeeze(0).to(DEVICE)

        wav_len = max(wav_noi.shape[-1], wav_enh.shape[-1])
        if wav_noi.shape[-1] < wav_len:
            wav_noi = F.pad(wav_noi, (0, wav_len - wav_noi.shape[-1]))
        if wav_enh.shape[-1] < wav_len:
            wav_enh = F.pad(wav_enh, (0, wav_len - wav_enh.shape[-1]))

        noisy_wavs.append(wav_noi)
        enhanced_wavs.append(wav_enh)

        wav_dict = wt_aligner.align_whisper_timestamped(noisy_wav_path)
        wav_segments = wt_aligner.get_segments(wav_dict)
        intervals.append(wav_segments)
        metadata.append((os.path.basename(noisy_wav_path), len(wav_noi)))
    batched_noisy_wavs = pad_sequence(noisy_wavs, batch_first=True, padding_value=0.0)
    batched_enhanced_wavs = pad_sequence(enhanced_wavs, batch_first=True, padding_value=0.0)

    # print(batched_noisy_wavs.shape)
    # print(batched_enhanced_wavs.shape)
    # print(intervals)
    print(metadata)

    config_data = {
        "objective_params":{
            "quality": ["scoreq", -5],
            "robustness": ["whisper-embsim", 1]
        }
    }
    mixer = TMixer(config_data, DEVICE)
    out = mixer.mix_and_repair(batched_noisy_wavs, batched_enhanced_wavs, intervals)
    print(out.shape)

    out_wav = out.detach().cpu().numpy()

    if not os.path.exists(TEMP_OUT_PATH):
        os.makedirs(TEMP_OUT_PATH)
    
    for i, entry in enumerate(metadata):
        basename = entry[0]
        wav_len = min(entry[1], out.shape[-1])
        sf.write(f'{TEMP_OUT_PATH}/{basename}', out_wav[i,0:wav_len], 16000)

if __name__ == '__main__':
    test_whisper_timestamped()