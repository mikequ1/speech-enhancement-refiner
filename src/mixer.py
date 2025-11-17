import sys
sys.path.append('./src')
import torch
import torch.nn.functional as F

SR = 16000
class Mixer:
    def __init__(self, mos_model, embedding_model, device):
        self.device = device
        self.mos_model = mos_model
        self.embedding_model = embedding_model
        self.steps = 20
        self.lr = 0.1

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
        # get the number of alpha mixing coefficients
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
            alphas.data.clamp_(0., 1.)

        with torch.no_grad():
            alpha_mask = self._build_alpha_mask(
                alphas, batch_ids, starts, ends, B, T
            )
            out = alpha_mask * enh + (1.0 - alpha_mask) * noi

        return out
        