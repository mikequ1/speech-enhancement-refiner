from scipy.io import wavfile
import soundfile as sf
import os
import numpy as np
import csv
import torch
from tqdm import tqdm
import pandas as pd
from collections import Counter
import torchaudio
import warnings
warnings.filterwarnings('ignore')

import argparse
import re
from pesq import pesq
from pystoi import stoi
import soxr
import json


TARGET_FS = 16000

# PROCESSING SEMamba
def init_json(args):
    f = open(os.path.join(args.txtfiledir, args.list), 'r').read().splitlines()
    wavfiles = {}
    with tqdm(total=len(f), desc="Processing Audio Files") as pbar:
        for line in f:
            parts = line.split(';')
            wavfile = parts[0]
            filename = wavfile.split('/')[-1]
            # MSP _snr_ file suffix cleaning for noisy
            snr_parts = re.split(r'_snr\d_', filename)
            if (len(snr_parts) == 1):
                filename_clean = filename
            else:
                filename_clean = snr_parts[0] + '.wav'
            # MSP txtfile clean
            wavfile = filename
            
            wav_path = os.path.join(args.datadir, wavfile)
            clean_path = os.path.join(args.cleandir, filename_clean)
            if args.model == 'clean':
                wav_path = clean_path
            if (not os.path.exists(wav_path)):
                print(f"wav path does not exist: {wav_path}")
                continue
            elif (not os.path.exists(clean_path)):
                print(f"corresponding clean path does not exist: {clean_path}")
                continue
            cur_dict = {
                'wav_path': wav_path,
                'clean_path': clean_path
            }
            wavfiles[filename] = cur_dict
            pbar.update(1)


    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'w') as f:
        json.dump(wavfiles, f, indent=4)



def extract_csv(args):
    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'r') as f:
        data = json.load(f)

    # Flattened output rows
    rows = []
    columns = set()

    for filename, info in data.items():
        row = {'filename': filename}
        for key, value in info.items():
            if isinstance(value, list) and key == 'audiobox':
                for subkey, subval in value[0].items():
                    flat_key = f'{key}.{subkey}'
                    row[flat_key] = subval
                    columns.add(flat_key)
            else:
                row[key] = value
                columns.add(key)
        rows.append(row)

    columns = ['filename'] + sorted([col for col in columns if col != 'filename'])

    with open(f'speech_metrics/csvs/{args.model}__{args.list.split(".txt")[0]}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)



# PESQ/STOI score calculation
def compute_pesq_stoi(args):
    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'r') as f:
        data = json.load(f)

    with tqdm(total=len(data), desc="Processing Audio Files") as pbar:
        for file in data.keys():
            try:
                wav, rate_wav = torchaudio.load(data[file]['wav_path'])
                if rate_wav!=16000:
                    transform = torchaudio.transforms.Resample(rate_wav, 16000)
                    wav = transform(wav)
                    rate_wav = 16000
                clean, rate_clean = torchaudio.load(data[file]['clean_path'])
                if rate_clean!=16000:
                    transform = torchaudio.transforms.Resample(rate_clean, 16000)
                    clean = transform(clean)
                    rate_clean = 16000

                # rate_wav, wav = wavfile.read(data[file]['wav_path'])
                # rate_clean, clean = wavfile.read(data[file]['clean_path'])

                if rate_wav != rate_clean:
                    raise ValueError(f"Sample rates do not match: {rate_wav} vs {rate_clean}")

                min_len = min(wav.shape[1], clean.shape[1])
                wav = wav[0, :min_len].numpy().astype(np.float32)
                clean = clean[0, :min_len].numpy().astype(np.float32)


                pesq_score = pesq(rate_wav, clean, wav, 'wb')
                stoi_score = stoi(clean, wav, rate_wav, extended=False)

                data[file]['pesq'] = pesq_score
                data[file]['stoi'] = stoi_score
                pbar.update(1)
            
            except Exception as e:
                print(f"Error processing {file}: {e}")
                pbar.update(1)
                continue

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'w') as f:
        json.dump(data, f, indent=4)


# Speaker Similarity Score Calculation
def compute_spsim(args):
    from espnet2.bin.spk_inference import Speech2Embedding

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'r') as f:
        data = json.load(f)

    with tqdm(total=len(data), desc="Processing Audio Files") as pbar:
        for file in data.keys():
            try:
                model = Speech2Embedding.from_pretrained(
                    model_tag="espnet/voxcelebs12_rawnet3", device="cuda"
                )
                model.spk_model.eval()
                score = process_one_pair(data[file]['clean_path'], data[file]['wav_path'], model=model)
                data[file]['spsim'] = score
                pbar.update(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing {f}: {e}")
                pbar.update(1)
                continue

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'w') as f:
        json.dump(data, f, indent=4)


# SPEAKER SIMILARITY
def speaker_similarity_metric(model, ref, inf, fs=16000):
    """Calculate the cosine similarity between ref and inf speaker embeddings.

    Args:
        model (torch.nn.Module): speaker model
            Please use the model in https://huggingface.co/espnet/voxcelebs12_rawnet3
            to get comparable results.
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        similarity (float): cosine similarity value between [0, 1]
    """
    if fs != TARGET_FS:
        ref = soxr.resample(ref, fs, TARGET_FS)
        inf = soxr.resample(inf, fs, TARGET_FS)
    with torch.no_grad():
        ref_emb = model(ref)
        inf_emb = model(inf)
        similarity = torch.cosine_similarity(ref_emb, inf_emb, dim=-1).item()
    return similarity

def process_one_pair(ref_path, inf_path, model=None):
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")

    min_len = min(len(ref), len(inf))
    ref = ref[:min_len].astype(np.float32)
    inf = inf[:min_len].astype(np.float32)

    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    assert ref.ndim == 1, ref.shape
    score = speaker_similarity_metric(model, ref, inf, fs=fs)

    return score


def compute_audiobox(args):
    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor()

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'r') as f:
        data = json.load(f)

    with tqdm(total=len(data), desc="Processing Audio Files") as pbar:
        for file in data.keys():
            try:
                wav, sr = torchaudio.load(data[file]['wav_path'])
                if sr!=16000:
                    transform = torchaudio.transforms.Resample(sr, 16000)
                    wav = transform(wav)
                    sr = 16000
                output = predictor.forward([{"path":wav, "sample_rate": sr}])
                data[file]['audiobox'] = output
                pbar.update(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing {file}: {e}")
                pbar.update(1)
                continue

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'w') as f:
        json.dump(data, f, indent=4)

def compute_scoreq(args):
    import scoreq

    nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr')

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'r') as f:
        data = json.load(f)

    with tqdm(total=len(data), desc="Processing Audio Files") as pbar:
        for file in data.keys():
            try:
                pred_mos = nr_scoreq.predict(test_path=data[file]['wav_path'], ref_path=None)
                data[file]['scoreq'] = pred_mos
                pbar.update(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing {file}: {e}")
                pbar.update(1)
                continue

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'w') as f:
        json.dump(data, f, indent=4)

def compute_wer(args):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from jiwer import wer
    
    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'r') as f:
        data = json.load(f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    with tqdm(total=len(data), desc="Processing Audio Files") as pbar:
        for file in data.keys():
            try:
                enh = pipe(data[file]['wav_path'])
                ref = pipe(data[file]['clean_path'])

                enh_asr = enh['text']
                ref_asr = ref['text']

                err = wer(ref_asr, enh_asr)
                data[file]['wer'] = err
                # print(enh)
                # print(ref)
                # print(err)
                pbar.update(1)
            
            except Exception as e:
                print(f"Error processing {file}: {e}")
                pbar.update(1)
                continue

    with open(f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json', 'w') as f:
        json.dump(data, f, indent=4)

def _load_mono_16k(path):
    wav, sr = torchaudio.load(path)               # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)       # mono
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    return wav.squeeze(0).numpy(), sr             # (T,), 16000

def compute_pitch_energy(args):
    """
    Adds per-file:
      data[file]['prosody'] = {
          'time_s': [...],
          'f0_hz': [...],          # np.nan for unvoiced
          'rms_energy': [...]      # framewise RMS
      }
    """

    import json
    import numpy as np
    from tqdm import tqdm
    import librosa
    import pyworld as pw

    json_path = f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    frame_ms = 5   # hop for WORLD + RMS
    win_ms = 25    # RMS window

    with tqdm(total=len(data), desc="Pitch+Energy (means only)") as pbar:
        for file, meta in data.items():
            try:
                y, sr = _load_mono_16k(meta['wav_path'])

                # ----- Pitch (WORLD) -----
                y64 = y.astype(np.float64)
                _f0, t = pw.dio(y64, sr, frame_period=frame_ms)
                f0 = pw.stonemask(y64, _f0, t, sr)
                f0 = np.where(f0 > 0, f0, np.nan)  # unvoiced -> NaN
                pitch_mean = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else None

                # ----- Energy (RMS) -----
                hop_length = int(sr * frame_ms / 1000.0)
                win_length = int(sr * win_ms / 1000.0)
                rms = librosa.feature.rms(
                    y=y.astype(np.float32),
                    frame_length=win_length,
                    hop_length=hop_length,
                    center=False
                )[0]
                energy_mean = float(np.nanmean(rms)) if rms.size > 0 else None

                # Write ONLY scalars
                data[file]['pitch'] = pitch_mean
                data[file]['energy'] = energy_mean

                pbar.update(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing {file}: {e}")
                # Ensure keys exist even on failure
                data.setdefault(file, {})
                data[file]['pitch'] = data[file].get('pitch', None)
                data[file]['energy'] = data[file].get('energy', None)
                pbar.update(1)
                continue

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def compute_verification(args):
    import nemo.collections.asr as nemo_asr
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

    json_path = f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_correct = 0
    num_total = 0
    with tqdm(total=len(data), desc="Processing Audio Files") as pbar:
        for file in data.keys():
            try:
                emb_clean = speaker_model.get_embedding(data[file]['clean_path'])
                emb_noisy = speaker_model.get_embedding(data[file]['wav_path'])
                emb_dist = torch.dist(emb_clean, emb_noisy, p=2)
                data[file]['speaker-verification'] = emb_dist.item()
                pbar.update(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing {f}: {e}")
                pbar.update(1)
                continue

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def compute_jitter_shimmer(args):
    import parselmouth
    from parselmouth.praat import call

    json_path = f'speech_metrics/{args.model}__{args.list.split(".txt")[0]}.json'
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Praat default pitch search range (adult speech); tune if needed
    fmin = 75
    fmax = 500

    with tqdm(total=len(data), desc="Jitter+Shimmer") as pbar:
        for file, meta in data.items():
            try:
                snd = parselmouth.Sound(meta['wav_path'])
                intensity = snd.to_intensity()
                
                pitch = snd.to_pitch()
                pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
                intensity_mean = intensity.get_average()

                pp = call(snd, "To PointProcess (periodic, cc)", fmin, fmax)
                jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

                data[file]['jitter'] = float(jitter_local)
                data[file]['shimmer'] = float(shimmer_local)
                data[file]['pitch'] = float(pitch_mean)
                data[file]['intensity'] = float(intensity_mean)
                pbar.update(1)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error processing {file}: {e}")
                pbar.update(1)
                continue

    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='temp', help='model used to enhance/reconstruct wavs')
    parser.add_argument('--datadir', type=str, default='/proj/speech/projects/noise_robustness/MSP-PODCAST-Publish-1.11', help='Path of your model DATA/ directory')
    parser.add_argument('--cleandir', type=str, default='/proj/speech/projects/noise_robustness/MSP-PODCAST-Publish-1.11/Audios', help='Path of your clean reference DATA/ directory')
    parser.add_argument('--list', type=str, default='msp1_11-test2-clean-noisy.txt', help='Path of your DATA/ list')
    parser.add_argument('--txtfiledir', default='./txtfile',  type=str, help='Path to testing txt directory')
    
    

    args = parser.parse_args()

    init_json(args)
    compute_pesq_stoi(args)
    # compute_spsim(args)
    compute_audiobox(args)
    compute_scoreq(args)
    # compute_wer(args)
    compute_jitter_shimmer(args)
    # compute_verification(args)
    # extract_csv(args)

    

if __name__ == "__main__":
    main()