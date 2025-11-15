import sys
import os
import whisper_timestamped as wt

class Aligner:
    def __init__(self, model:str, device):
        self.model = None
        self.model_name = model
        self.device = device
        if model == 'whisper-timestamped':
            self.model = wt.load_model('base', device=device)
        elif model == 'clarsiu':
            raise NotImplementedError
        else:
            print('Unspecified mode: [whisper-timestamped, charsiu]')
            exit()

    def align_whisper_timestamped(self, path):
        wav = wt.load_audio(path)
        wt_dict = wt.transcribe(self.model, wav, verbose=None)
        return wt_dict
    
    def get_segments(self, wt_dict):
        intervals = []
        for segment in wt_dict['segments']:
            segment_words = segment['words']
            for segment_word in segment_words:
                intervals.append((segment_word['text'], segment_word['start'], segment_word['end']))
        return intervals
