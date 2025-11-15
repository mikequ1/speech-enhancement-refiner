import sys
sys.path.append('./src')
from evaluator import ScoreqEvaluator
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

    def mix_and_repair(self, noi, enh, intervals):
        # Initialization
        B,T = noi.shape
        # get the number of alpha mixing coefficients
        num_alphas_per_interval = [len(iv) for iv in intervals]
        num_alphas_total = sum(num_alphas_per_interval)
        alphas = torch.full((num_alphas_total,), 0.5, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([alphas], lr=self.lr)
        # alpha index offset mapping
        offsets = [0]
        for i in range(B):
            offsets.append(offsets[-1] + num_alphas_per_interval[i])
        # reference embedding
        ref_melspec = self.embedding_model.transform_melspec(noi)
        ref_emb = self.embedding_model.embed(ref_melspec)

        # Optimization Loop
        for _ in range(self.steps):
            opt.zero_grad()
            # Synthesize mixed output per-interval for each item in the batch
            out = enh.clone()
            for wav_idx in range(B):
                for i, seg in enumerate(intervals[wav_idx]):
                    i0 = max(0, int(seg[1] * SR))
                    i1 = min(T, int(seg[2] * SR))
                    if i1 <= i0: continue # if interval is invalid, default to using enhanced signal
                    alpha = alphas[offsets[wav_idx] + i].clamp(0., 1.)
                    # Reconstruction
                    out[wav_idx, i0:i1] = alpha * enh[wav_idx, i0:i1] + (1. - alpha) * noi[wav_idx, i0:i1]
            
            # loss calculation
            mos_scores_t = self.mos_model.evaluate(out)
            out_melspec = self.embedding_model.transform_melspec(out)
            out_emb = self.embedding_model.embed(out_melspec)

            cost_mos = mos_scores_t.sum() # quality component
            cost_embsim = F.cosine_similarity(out_emb, ref_emb, dim=1).sum() # robustness component

            loss = 0
            loss -= cost_mos
            loss -= cost_embsim

            loss.backward()
            opt.step()
            alphas.data.clamp_(0., 1.)

        # Final synthesis with optimized alphas
        with torch.no_grad():
            out = enh.clone()
            for wav_idx in range(B):
                for i, seg in enumerate(intervals[wav_idx]):
                    i0 = max(0, int(seg[1] * SR))
                    i1 = min(T, int(seg[2] * SR))
                    if i1 <= i0:
                        continue
                    alpha = alphas[offsets[wav_idx] + i]
                    out[wav_idx, i0:i1] = (alpha * enh[wav_idx, i0:i1] + (1. - alpha) * noi[wav_idx, i0:i1])

        return out
        