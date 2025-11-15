import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def batchify(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]

def load_batch_audio(batch_meta, device):
    noisy_wavs = []
    enhanced_wavs = []
    noisy_intervals = []
    enhanced_intervals = []
    lengths = []
    wav_names = []

    for entry in batch_meta:
        noisy_path = entry[0]
        enhanced_path = entry[1]
        noisy_interval = entry[2]
        enhanced_interval = entry[3]
        wav_name = entry[4]

        noisy_wav, _ = torchaudio.load(noisy_path)
        enhanced_wav, _ = torchaudio.load(enhanced_path)
        noisy_wav = noisy_wav.squeeze(0).to(device)
        enhanced_wav = enhanced_wav.squeeze(0).to(device)

        wav_len = max(noisy_wav.shape[-1], enhanced_wav.shape[-1])
        if noisy_wav.shape[-1] < wav_len:
            noisy_wav = F.pad(noisy_wav, (0, wav_len - noisy_wav.shape[-1]))
        if enhanced_wav.shape[-1] < wav_len:
            enhanced_wav = F.pad(enhanced_wav, (0, wav_len - enhanced_wav.shape[-1]))

        noisy_wavs.append(noisy_wav)
        enhanced_wavs.append(enhanced_wav)
        noisy_intervals.append(noisy_interval)
        enhanced_intervals.append(enhanced_interval)
        lengths.append(wav_len)
        wav_names.append(wav_name)

    batched_noisy_wavs = pad_sequence(noisy_wavs, batch_first=True, padding_value=0.0)
    batched_enhanced_wavs = pad_sequence(enhanced_wavs, batch_first=True, padding_value=0.0)

    return batched_noisy_wavs, batched_enhanced_wavs, noisy_intervals, enhanced_intervals, lengths, wav_names