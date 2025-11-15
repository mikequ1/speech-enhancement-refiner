import argparse
import sys
import os
from tqdm import tqdm

import torch
import torchaudio
import soundfile as sf

sys.path.append('./src')
from aligner import Aligner
from evaluator import ScoreqEvaluator
from embedder import WhisperEmbedder
from mixer import Mixer
from utils import batchify, load_batch_audio

def main(args):
    model_name = 'whisper-timestamped'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wt_aligner = Aligner(model=model_name, device=device)

    config_file = open(os.path.join(args.txtfiledir, args.list), 'r').read().splitlines()

    # preprocessing metadata
    metadata = []
    for line in tqdm(config_file, desc='Preprocessing...'):
        config_parts = line.split(' ')
        noisy_path = os.path.join(args.noisydir, config_parts[0])
        enhanced_path = os.path.join(args.enhanceddir, config_parts[1])
        wav_name = os.path.basename(noisy_path)
        if not os.path.exists(noisy_path):
            print(f'Skipping... | noisy path [{noisy_path}] not found')
            continue
        if not os.path.exists(enhanced_path):
            print(f'Skipping... | enhanced path [{noisy_path}] not found')
            continue

        noisy_alignment_dict = wt_aligner.align_whisper_timestamped(noisy_path)
        enhanced_alignment_dict = wt_aligner.align_whisper_timestamped(enhanced_path)
        noisy_segments = wt_aligner.get_segments(noisy_alignment_dict)
        enhanced_segments = wt_aligner.get_segments(enhanced_alignment_dict)

        metadata.append([noisy_path, enhanced_path, noisy_segments, enhanced_segments, wav_name])

    mos_evaluator = ScoreqEvaluator(device)
    whisper_embedder = WhisperEmbedder("openai/whisper-base", device)
    mixer = Mixer(mos_evaluator, whisper_embedder, device)
    
    batch_size = 8
    # batch inference
    for batch_meta in tqdm(batchify(metadata, batch_size), desc='Processing batch...'):
        noisy_wavs, enhanced_wavs, noisy_intervals, enhanced_intervals, wav_lengths, wav_names = load_batch_audio(batch_meta, device)
        mix_on_noi = mixer.mix_and_repair(noisy_wavs, enhanced_wavs, noisy_intervals)
        out_wav = mix_on_noi.detach().cpu().numpy()

        if not os.path.exists(args.savedir):
            os.makedirs(args.savedir)

        for i in range(len(wav_lengths)):
            wav_len = wav_lengths[i]
            wav_name = wav_names[i]
            sf.write(f'{args.savedir}/{wav_name}', out_wav[i,0:wav_len], 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisydir', default='./data',  type=str, help='Path to root data directory')
    parser.add_argument('--enhanceddir', default='./enhanced', type=str, help='Path to enhanced data directory')
    parser.add_argument('--txtfiledir', default='./txtfile',  type=str, help='Path to training txt directory')
    parser.add_argument('--list', default='msp1_11-test2-snr4.txt', type=str, help='File name of the training list txtfile')
    parser.add_argument('--savedir', type=str, required=False, default='./reconstructed_wavs', help='Output directory for your trained checkpoints')
    args = parser.parse_args()

    main(args)