import sys
sys.path.append('./src')
import torch
import torch.nn.functional as F

SR = 16000
class TMixer:
    def __init__(self, mos_model, embedding_model, device):
        self.device = device
        self.mos_model = mos_model
        self.embedding_model = embedding_model
        self.steps = 20
        self.lr = 0.05

    def _prepare_intervals(self, intervals, B, T):
        batch_ids_list = []
        starts_list = []
        ends_list = []

        for b in range(B):
            for seg in intervals[b]:
                i0 = max(0, int(seg[1] * SR))
                i1 = min(T, int(seg[2] * SR))
                if i1 <= i0:
                    continue

                batch_ids_list.append(b)
                starts_list.append(i0)
                ends_list.append(i1)

        batch_ids = torch.tensor(batch_ids_list, dtype=torch.long, device=self.device)
        starts = torch.tensor(starts_list, dtype=torch.long, device=self.device)
        ends = torch.tensor(ends_list, dtype=torch.long, device=self.device)

        return batch_ids, starts, ends
    
    def _build_alpha_mask(self, alphas, batch_ids, starts, ends, B, T):
        flat_stride = T + 1
        delta = torch.zeros(B * flat_stride, device=self.device)
        alpha = alphas.clamp(0.0, 1.0)

        valid = ends > starts
        if valid.any():
            a = alpha[valid]
            b = batch_ids[valid]
            s = starts[valid]
            e = ends[valid]

            flat_start = b * flat_stride + s
            flat_end = b * flat_stride + e

            delta.index_add_(0, flat_start, a - 1.0)
            delta.index_add_(0, flat_end, 1.0 - a)

        delta = delta.view(B, flat_stride) # [B, T+1]
        alpha_mask = 1.0 + delta[:, :-1].cumsum(1) # [B, T]
        return alpha_mask

    def mix_and_repair(self, noi, enh, intervals):
        # Initialization
        B,T = noi.shape
        batch_ids, starts, ends = self._prepare_intervals(intervals, B, T)
        # param initialization
        alphas = torch.full((len(batch_ids),), 0.5, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([alphas], lr=self.lr)
        kernel_size = 201  # 200/16k = 1/80 s convolution steps
        kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size
        # reference embedding
        ref_melspec = self.embedding_model.transform_melspec(noi)
        ref_emb = self.embedding_model.embed(ref_melspec).detach()

        # Optimization Loop
        for _ in range(self.steps):
            opt.zero_grad()
            # Synthesize mixed output per-interval for each item in the batch
            alpha_mask = self._build_alpha_mask(
                alphas, batch_ids, starts, ends, B, T
            )
            alpha_mask = F.conv1d(
                alpha_mask.unsqueeze(1),
                kernel,
                padding=kernel_size // 2,
                groups=1
            ).squeeze(1)
            out = alpha_mask * enh + (1.0 - alpha_mask) * noi
            
            # loss calculation
            mos_scores_t = self.mos_model.evaluate(out)
            out_melspec = self.embedding_model.transform_melspec(out)
            out_emb = self.embedding_model.embed(out_melspec)

            cost_mos = mos_scores_t.mean() # quality component
            cost_embsim = F.cosine_similarity(out_emb, ref_emb, dim=1).mean() # robustness component

            loss = 0
            loss -= cost_mos
            loss += cost_embsim

            loss.backward()
            opt.step()
            with torch.no_grad():
                alphas.data.clamp_(0., 1.)

        with torch.no_grad():
            alpha_mask = self._build_alpha_mask(
                alphas, batch_ids, starts, ends, B, T
            )
            alpha_mask = F.conv1d(
                alpha_mask.unsqueeze(1),
                kernel,
                padding=kernel_size // 2,
                groups=1
            ).squeeze(1)
            out = alpha_mask * enh + (1.0 - alpha_mask) * noi
        return out


class TFMixer:
    def __init__(self, mos_model, embedding_model, device):
        self.device = device
        self.mos_model = mos_model
        self.embedding_model = embedding_model
        self.steps = 20
        self.lr = 0.05

        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.window = torch.hann_window(self.win_length).to(device)

    def _stft(self, wav):
        # input: [B, T]
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )  # [B, F, T_frames]

    def _istft(self, spec, length):
        # spectrogram: [B, F, T_frames]
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
        )  # [B, T]

    def _prepare_interval_frames(self, intervals, B, T_frames):
        batch_ids_list = []
        starts_list = []
        ends_list = []

        for b in range(B):
            for seg in intervals[b]:
                f0 = int(seg[1] * SR / self.hop_length)
                f1 = int(seg[2] * SR / self.hop_length)
                f0 = max(0, f0)
                f1 = min(T_frames, f1)
                if f1 <= f0:
                    continue

                batch_ids_list.append(b)
                starts_list.append(f0)
                ends_list.append(f1)

        if len(batch_ids_list) == 0:
            return None, None, None

        batch_ids = torch.tensor(batch_ids_list, dtype=torch.long, device=self.device)
        starts = torch.tensor(starts_list, dtype=torch.long, device=self.device)
        ends = torch.tensor(ends_list, dtype=torch.long, device=self.device)

        return batch_ids, starts, ends

    def _build_alpha_mask_tf(self, alphas, batch_ids, starts, ends, B, F, T_frames):
        alpha_mask = torch.ones(B, F, T_frames, device=self.device)
        alpha_clamped = alphas.clamp(0.0, 1.0)

        num_seg = batch_ids.shape[0]
        for i in range(num_seg):
            b = batch_ids[i].item()
            s = starts[i].item()
            e = ends[i].item()
            alpha_mask[b, :, s:e] = alpha_clamped[i].unsqueeze(-1)
        return alpha_mask


    def mix_and_repair(self, noi, enh, intervals):
        # Initialization
        spec_noi = self._stft(noi)  # [B, F, T_frames]
        spec_enh = self._stft(enh)  # [B, F, T_frames]
        B, T = noi.shape
        B, FQ, T_frames = spec_noi.shape
        batch_ids, starts, ends = self._prepare_interval_frames(intervals, B, T_frames)
        N_seg = batch_ids.shape[0]
        # param initialization
        alphas = torch.full(
            (N_seg, FQ),
            0.5,
            device=self.device,
            requires_grad=True,
        )
        opt = torch.optim.Adam([alphas], lr=self.lr)
        kernel_t = 5
        kernel_f = 3
        tf_kernel = torch.ones(1, 1, kernel_f, kernel_t, device=self.device)
        tf_kernel = tf_kernel / tf_kernel.numel()
        # reference embedding
        ref_melspec = self.embedding_model.transform_melspec(noi)
        ref_emb = self.embedding_model.embed(ref_melspec).detach()

        for _ in range(self.steps):
            opt.zero_grad()
            alpha_mask = self._build_alpha_mask_tf(
                alphas, batch_ids, starts, ends, B, FQ, T_frames
            )  # [B, F, T_frames]
            alpha_mask_sm = F.conv2d(
                alpha_mask.unsqueeze(1),  # [B, 1, F, T_frames]
                tf_kernel,
                padding=(kernel_f // 2, kernel_t // 2),
            ).squeeze(1)  # [B, F, T_frames]

            mix_spec = alpha_mask_sm * spec_enh + (1.0 - alpha_mask_sm) * spec_noi
            out = self._istft(mix_spec, length=T)  # [B, T]

            # loss calculation
            mos_scores_t = self.mos_model.evaluate(out)
            out_melspec = self.embedding_model.transform_melspec(out)
            out_emb = self.embedding_model.embed(out_melspec)

            cost_mos = mos_scores_t.mean()
            cost_embsim = F.cosine_similarity(out_emb, ref_emb, dim=1).mean()

            loss = 0.0
            loss -= cost_mos
            loss += cost_embsim

            loss.backward()
            opt.step()
            with torch.no_grad():
                alphas.clamp_(0.0, 1.0)

        with torch.no_grad():
            alpha_mask = self._build_alpha_mask_tf(
                alphas, batch_ids, starts, ends, B, FQ, T_frames
            )
            alpha_mask_sm = F.conv2d(
                alpha_mask.unsqueeze(1),
                tf_kernel,
                padding=(kernel_f // 2, kernel_t // 2),
            ).squeeze(1)
            mix_spec = alpha_mask_sm * spec_enh + (1.0 - alpha_mask_sm) * spec_noi
            out = self._istft(mix_spec, length=T)
        return out