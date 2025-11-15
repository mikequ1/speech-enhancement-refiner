import scoreq
from audiobox_aesthetics.infer import initialize_predictor

class ScoreqEvaluator:
    def __init__(self, device):
        self.sq = scoreq.Scoreq(data_domain='natural', mode='nr')

    def evaluate(self, wav_tensor):
        return self.sq.model(wav_tensor)
