import json
import os
import pandas as pd
import re


def load_spider(data_dir, split):
    if os.path.exists(os.path.join(data_dir, f'{split}.tsv')):
        df = pd.read_csv(os.path.join(data_dir, f'{split}.tsv'), sep='\t', header=None, names=['query', 'parse'])
    else:
        df = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
    return df


def load_snli(data_dir, split):
    raw_data = []
    with open(os.path.join(data_dir, f"snli_1.0_{split}.jsonl")) as fin:
        for line in fin:
            raw_data.append(json.loads(line))
    key_set = list(raw_data[0].keys())
    raw_data_dict = {}
    for k_ in key_set:
        raw_data_dict[k_] = [row[k_] for row in raw_data]
    for k_ in key_set:
        if k_.startswith('sentence1'):
            new_k = k_.replace('sentence1', 'premise')
            raw_data_dict[new_k] = raw_data_dict[k_]
            del raw_data_dict[k_]
        if k_.startswith('sentence2'):
            new_k = k_.replace('sentence2', 'hypothesis')
            raw_data_dict[new_k] = raw_data_dict[k_]
            del raw_data_dict[k_]
    df = pd.DataFrame(raw_data_dict)
    # print(f"DEBUG: len(df) = {len(df)}, columns = {df.columns}")
    return df


def load_boolq(data_dir, split):
    raw_data = []
    with open(os.path.join(data_dir, f"{split}.jsonl")) as fin:
        for line in fin:
            raw_data.append(json.loads(line))
    key_set = list(raw_data[0].keys())
    raw_data_dict = {}
    for k_ in key_set:
        raw_data_dict[k_] = [row[k_] for row in raw_data]
    df = pd.DataFrame(raw_data_dict)
    return df
