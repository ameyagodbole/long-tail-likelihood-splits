import argparse
import os
from nltk import word_tokenize
import numpy as np
import pandas as pd
import json

DEV_SPLIT_SZ = {'snli': 10000, 'boolq': 2540}
TEST_SPLIT_SZ = {'snli': 10000, 'boolq': 2540}
TASK_COL_NAMES = {'snli': ('premise', 'hypothesis', 'gold_label'),
                  'boolq': ('passage', 'question', 'label')}


def main(args):
    rng = np.random.default_rng(args.seed)
    df_all = pd.read_csv(args.src_data_file)

    base_size = len(df_all)
    print(f"Loaded {base_size} examples")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sentence1_key, sentence2_key, label_key = TASK_COL_NAMES[args.dataset_name]
    df_all['length'] = df_all.apply(lambda x: len(word_tokenize(x[sentence2_key])), axis=1)

    df_all = df_all.sort_values('length', axis=0, ascending=True, kind='mergesort', ignore_index=False)

    examples_train, examples_dev, examples_test = [], [], []
    label_classes = df_all[label_key].unique().tolist()
    for outer_ctr, label in enumerate(label_classes):
        print(f"Class: {label}")
        df_class = df_all[df_all[label_key] == label]
        size_target_te = TEST_SPLIT_SZ[args.dataset_name] // len(label_classes)
        size_target_de = DEV_SPLIT_SZ[args.dataset_name] // len(label_classes)
        if outer_ctr == 0:
            size_target_te += int(TEST_SPLIT_SZ[args.dataset_name] % len(label_classes))
            size_target_de += int(DEV_SPLIT_SZ[args.dataset_name] % len(label_classes))
        size_target_eval = size_target_te + size_target_de
        # size_target_tr = len(df_class) - size_target_te - size_target_de

        # LONGER examples in eval set
        examples_eval = df_class[-size_target_eval:]
        examples_eval = examples_eval.sample(frac=1, random_state=rng.bit_generator)

        examples_test.append(examples_eval[:size_target_te])
        examples_dev.append(examples_eval[size_target_te:])

        # SHORTER examples in train
        examples_train.append(df_class[:-size_target_eval])

    examples_train = pd.concat(examples_train)
    examples_dev = pd.concat(examples_dev)
    examples_test = pd.concat(examples_test)

    print("--- --- ---")
    print(f"Final sizes: train ({len(examples_train)}), dev ({len(examples_dev)}), test ({len(examples_test)})")

    with open(os.path.join(args.output_dir, "train_idx.json"), 'w') as fout:
        json.dump(list(examples_train.index), fout, indent=4)
    with open(os.path.join(args.output_dir, "dev_idx.json"), 'w') as fout:
        json.dump(list(examples_dev.index), fout, indent=4)
    with open(os.path.join(args.output_dir, "test_idx.json"), 'w') as fout:
        json.dump(list(examples_test.index), fout, indent=4)
    examples_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    examples_dev.to_csv(os.path.join(args.output_dir, "dev.csv"), index=False)
    examples_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data_file", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    args_ = parser.parse_args()

    print(vars(args_))
    main(args_)
