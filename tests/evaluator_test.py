import os
import sys
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
sys.path.append('./src')
from evaluator import ScoreqEvaluator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
TEST_WAV_PATH = './tests/test_noisy_audios.txt'

def load_test_audio():
    test_wavs = []
    with open(TEST_WAV_PATH, 'r') as file:
        for line in file:
            test_wavs.append(line.rstrip('\n'))
    return test_wavs

# Whisper-Timestamped Batched Evaluation Tests
def test_evaluator():
    print('RUNNING TESTS')
    test_wavs = load_test_audio()
    scoreq_evaluator = ScoreqEvaluator(device=DEVICE)
    wavs = []
    for i, test_wav in enumerate(test_wavs):
        print(f'\n{i+1}/{len(test_wavs)} : {test_wav}')
        wav, _ = torchaudio.load(test_wav)
        wav = wav.to(DEVICE)
        wav = wav.squeeze(0)
        print(wav.shape)
        wavs.append(wav)
    batched_wavs = pad_sequence(wavs, batch_first=True, padding_value=0.0)
    scoreq_score = scoreq_evaluator.evaluate(batched_wavs)
    print(f'scoreq score: {scoreq_score}')

if __name__ == '__main__':
    test_evaluator()