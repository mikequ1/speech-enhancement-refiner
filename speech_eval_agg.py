import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='noisy__msp1_11-test2-snr4.json', help='file path of the json to run evaluation')
    args = parser.parse_args()

    with open(f'speech_metrics/{args.file}', 'r') as f:
        data = json.load(f)

    print(f'Running file: {args.file}')
    agg = {}
    for i in data:
        for metric, value in data[i].items():
            if metric in ('wav_path', 'clean_path'):
                continue

            if metric == 'audiobox':
                for submetric, subvalue in value[0].items():  # assuming value is a list of dicts
                    key = f'{metric}.{submetric}'
                    agg.setdefault(key, []).append(subvalue)
            else:
                agg.setdefault(metric, []).append(value)

    for metric in agg.keys():
        cur_arr = np.array(agg[metric])
        cur_arr = [x if x is not None else np.nan for x in cur_arr]
        print(f'{metric} | Mean: {np.round(np.nanmean(cur_arr),5)}, St.Dev: {np.round(np.nanstd(cur_arr),5)}')


if __name__ == "__main__":
    main()