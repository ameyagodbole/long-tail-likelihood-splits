import argparse
import os
from nltk import word_tokenize
import numpy as np
import pandas as pd
import json

from lm_analysis.data_utils import load_snli


def main(args):
    rng = np.random.default_rng(42)
    train_dset = load_snli(args.data_dir, 'train')
    train_dset['split'] = 'train'
    train_dset = train_dset.rename(index=dict(zip(train_dset.index, [f'train_{i_}' for i_ in range(len(train_dset))])))

    dev_dset = load_snli(args.data_dir, 'dev')
    dev_dset['split'] = 'dev'
    dev_dset = dev_dset.rename(index=dict(zip(dev_dset.index, [f'dev_{i_}' for i_ in range(len(dev_dset))])))

    test_dset = load_snli(args.data_dir, 'test')
    test_dset['split'] = 'test'
    test_dset = test_dset.rename(index=dict(zip(test_dset.index, [f'test_{i_}' for i_ in range(len(test_dset))])))

    base_size = len(train_dset)
    train_dset.drop(train_dset[train_dset['gold_label'] == '-'].index, inplace=True)
    train_dset.drop(train_dset[train_dset['premise'] == "Cannot see picture to describe."].index, inplace=True)
    train_dset.drop(train_dset[train_dset['hypothesis'].apply(lambda h_str: h_str.lower()) == 'n/a'].index, inplace=True)
    drop_size = len(train_dset)
    print(f"TRAIN: Dropped {base_size - drop_size} of {base_size} examples. "
          f"Total number of samples retained = {drop_size}")

    base_size = len(dev_dset)
    dev_dset.drop(dev_dset[dev_dset['gold_label'] == '-'].index, inplace=True)
    dev_dset.drop(dev_dset[dev_dset['premise'] == "Cannot see picture to describe."].index, inplace=True)
    dev_dset.drop(dev_dset[dev_dset['hypothesis'].apply(lambda h_str: h_str.lower()) == 'n/a'].index,
                  inplace=True)
    drop_size = len(dev_dset)
    print(f"DEV: Dropped {base_size - drop_size} of {base_size} examples. "
          f"Total number of samples retained = {drop_size}")

    base_size = len(test_dset)
    test_dset.drop(test_dset[test_dset['gold_label'] == '-'].index, inplace=True)
    test_dset.drop(test_dset[test_dset['premise'] == "Cannot see picture to describe."].index, inplace=True)
    test_dset.drop(test_dset[test_dset['hypothesis'].apply(lambda h_str: h_str.lower()) == 'n/a'].index,
                   inplace=True)
    drop_size = len(test_dset)
    print(f"TEST: Dropped {base_size - drop_size} of {base_size} examples. "
          f"Total number of samples retained = {drop_size}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("--- --- ---")
    print(f"Final sizes: train ({len(train_dset)}), dev ({len(dev_dset)}), test ({len(test_dset)})")

    with open(os.path.join(args.output_dir, "train_idx.json"), 'w') as fout:
        json.dump(list(train_dset.index), fout, indent=4)
    with open(os.path.join(args.output_dir, "dev_idx.json"), 'w') as fout:
        json.dump(list(dev_dset.index), fout, indent=4)
    with open(os.path.join(args.output_dir, "test_idx.json"), 'w') as fout:
        json.dump(list(test_dset.index), fout, indent=4)
    train_dset.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    dev_dset.to_csv(os.path.join(args.output_dir, "dev.csv"), index=False)
    test_dset.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--output_dir", type=str)
    args_ = parser.parse_args()

    print(vars(args_))
    main(args_)
