import os
import sys
import torch
sys.path.append('./src')
from aligner import Aligner

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
TEST_WAV_PATH = './tests/test_noisy_audios.txt'

def load_test_audio():
    test_wavs = []
    with open(TEST_WAV_PATH, 'r') as file:
        for line in file:
            test_wavs.append(line.rstrip('\n'))
    return test_wavs

# Whisper-Timestamped Aligner Tests
def test_whisper_timestamped():
    test_wavs = load_test_audio()
    model_name = 'whisper-timestamped'
    wt_aligner = Aligner(model=model_name, device=DEVICE)
    for i, test_wav in enumerate(test_wavs):
        print(f'\n{i+1}/{len(test_wavs)} : {test_wav}')
        wav_dict = wt_aligner.align_whisper_timestamped(test_wav)
        print(wt_aligner.get_segments(wav_dict))

if __name__ == '__main__':
    test_whisper_timestamped()