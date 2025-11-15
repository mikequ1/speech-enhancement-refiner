import os
import sys
import torch
import torchaudio
sys.path.append('./src')
from embedder import WhisperEmbedder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
TEST_WAV_PATH = './tests/test_noisy_audios.txt'

def load_test_audio():
    test_wavs = []
    with open(TEST_WAV_PATH, 'r') as file:
        for line in file:
            test_wavs.append(line.rstrip('\n'))
    return test_wavs

# Whisper-Timestamped Aligner Tests
def test_whisper_embedder():
    test_wavs = load_test_audio()
    whisper_embedder = WhisperEmbedder('openai/whisper-base', device=DEVICE)
    for i, test_wav in enumerate(test_wavs):
        print(f'\n{i+1}/{len(test_wavs)} : {test_wav}')
        wav, _ = torchaudio.load(test_wav)
        wav = wav.to(DEVICE)
        print(wav.shape)
        wav_melspec = whisper_embedder.transform_melspec(wav)
        print(wav_melspec.shape)
        wav_emb = whisper_embedder.embed(wav_melspec)
        print(wav_emb.shape)

if __name__ == '__main__':
    test_whisper_embedder()