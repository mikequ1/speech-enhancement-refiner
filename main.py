import argparse
import sys
import os
import yaml
from tqdm import tqdm

import torch
import torchaudio
import soundfile as sf

sys.path.append('./src')
from aligner import Aligner
from mixer import TMixer, TFMixer
from utils import batchify, load_batch_audio

CONFIG_PATH = './config.yml'

def main(args):
    with open(CONFIG_PATH, 'r') as config_file:
        config_data = yaml.safe_load(config_file)

    if args.use_config:
        noisy_dir = config_data['data_params']['noisy_path']
        enhanced_dir = config_data['data_params']['enhanced_path']
        list_file = config_data['data_params']['list_file']
        out_dir = config_data['data_params']['out_path']
    else:
        if args.noisydir is None or args.enhanceddir is None or args.listpath is None or args.savedir is None:
            print('args.noisydir, enhanceddir, listpath, savedir must be supplied. (use_config is set to False)')
            exit()
        noisy_dir = args.noisydir
        enhanced_dir = args.enhanceddir
        list_file = args.listpath
        out_dir = args.savedir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_name = 'whisper-timestamped'
    wt_aligner = Aligner(model=model_name, device=device)

    config_file = open(list_file, 'r').read().splitlines()

    # preprocessing metadata
    metadata = []
    for line in tqdm(config_file, desc='Preprocessing...'):
        config_parts = line.split(' ')
        noisy_path = os.path.join(noisy_dir, config_parts[0])
        enhanced_path = os.path.join(enhanced_dir, config_parts[1])
        wav_name = os.path.basename(noisy_path)
        if not os.path.exists(noisy_path):
            print(f'Skipping... | noisy path [{noisy_path}] not found')
            continue
        if not os.path.exists(enhanced_path):
            print(f'Skipping... | enhanced path [{noisy_path}] not found')
            continue

        try:
            noisy_alignment_dict = wt_aligner.align_whisper_timestamped(noisy_path)
            enhanced_alignment_dict = wt_aligner.align_whisper_timestamped(enhanced_path)
            noisy_segments = wt_aligner.get_segments(noisy_alignment_dict)
            enhanced_segments = wt_aligner.get_segments(enhanced_alignment_dict)

            metadata.append([noisy_path, enhanced_path, noisy_segments, enhanced_segments, wav_name])
        except:
            print(f'Error parsing metadata or aligning for {wav_name}, skipping...')

    if (args.use_config and config_data['model_params']['model'] == 'TF') or (args.tf):
        mixer = TFMixer(config_data, device)
    else:
        mixer = TMixer(config_data, device)
    
    batch_size = 4
    # batch inference
    for batch_meta in tqdm(batchify(metadata, batch_size), desc='Processing batch...'):
        noisy_wavs, enhanced_wavs, noisy_intervals, enhanced_intervals, wav_lengths, wav_names = load_batch_audio(batch_meta, device)
        mix_on_noi = mixer.mix_and_repair(noisy_wavs, enhanced_wavs, noisy_intervals)
        out_wav = mix_on_noi.detach().cpu().numpy()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i in range(len(wav_lengths)):
            wav_len = wav_lengths[i]
            wav_name = wav_names[i]
            sf.write(f'{out_dir}/{wav_name}', out_wav[i,0:wav_len], 16000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-config', action='store_true', help='Whether to use config.yaml for directory inputs')
    parser.add_argument('--tf', action='store_true', help='Perform T-F or T domain mixing and optimization')
    parser.add_argument('--noisydir', default=None, type=str, help='Path to root data directory')
    parser.add_argument('--enhanceddir', default=None, type=str, help='Path to enhanced data directory')
    parser.add_argument('--listpath', default=None,  type=str, help='Path to training txt directory')
    parser.add_argument('--savedir', default=None, type=str, help='Path to the output data directory')
    args = parser.parse_args()

    main(args)