import torch
import torchaudio
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor

SR = 16000
N_MELS = 80
N_FRAMES = 3000

class WhisperEmbedder:
    """
    The Embedder class provides functionality for encoding a speech signal using Whisper's pretrained encoder
    This is used as a robustness metric to prevent the corrected signal from deviating too far from the original signal.

    Attributes:
        model_name (str): Whisper model to use (e.g. 'openai/whisper-base')
        device (str)
    """
    def __init__(self, model_name, device):
        self.model = WhisperModel.from_pretrained(model_name).encoder
        self.device = device
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=N_MELS,
            center=True,
            power=2.0,
            norm=None,
            mel_scale="htk",
        ).to(self.device)

    def transform_melspec(self, wavs) -> torch.Tensor:
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)  # [T] => [1, T]
            
        wavs = wavs.to(self.device)
        mel = self.mel_spec(wavs)
        logmel = torch.log(mel + 1e-10)

        frames = logmel.size(-1)
        if frames < N_FRAMES:
            pad = N_FRAMES - frames
            logmel = F.pad(logmel, (0, pad), mode="constant", value=0.0)
        elif frames > N_FRAMES:
            logmel = logmel[..., :N_FRAMES]

        model_param = next(self.model.parameters())
        return logmel.to(device=model_param.device, dtype=model_param.dtype)

    def embed(self, features: torch.Tensor):
        enc = self.model(input_features=features).last_hidden_state  # [B, S, H]
        return enc.mean(dim=1).detach()  # [B, H]
