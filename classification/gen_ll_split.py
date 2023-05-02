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

    all_lls = np.load(os.path.join(args.lls_dir, 'all_examples__False_loglikelihoods.npy'))
    df_all['ll_score'] = all_lls

    all_ppls = np.load(os.path.join(args.lls_dir, 'all_examples__False_perplexity.npy'))
    df_all['ppl_score'] = all_ppls

    base_size = len(df_all)
    print(f"Loaded {base_size} examples")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sentence1_key, sentence2_key, label_key = TASK_COL_NAMES[args.dataset_name]
    df_all['length'] = df_all.apply(lambda x: len(word_tokenize(x[sentence2_key])), axis=1)
    if args.use_custom_ppl:
        print("Computing PPL from likelihood using NLTK sequence length")
        df_all["ppl_score"] = np.exp(-df_all["ll_score"] / df_all["length"])

    # sorted_order is in increasing order of likelihood
    if args.use_ppl:
        sort_key = "ppl_score"
        greater_in_test = True
    else:
        sort_key = "ll_score"
        greater_in_test = False
    df_all = df_all.sort_values(sort_key, axis=0, ascending=greater_in_test, kind='mergesort', ignore_index=False)

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

        if args.length_buckets > 1 or args.extreme_length_buckets:
            if args.extreme_length_buckets:
                uniq_lens, df_class["length_bucket"], uniq_len_counts = np.unique(df_class["length"],
                                                                                  return_inverse=True,
                                                                                  return_counts=True)
                print(f"Uniq lengths: {uniq_lens}")
                print(f"Lengths counts: {uniq_len_counts}")
            else:
                quantiles = [q_ / args.length_buckets for q_ in range(args.length_buckets + 1)]
                df_class["length_bucket"] = pd.qcut(df_class["length"], quantiles, labels=False)
            bucket_counts = df_class['length_bucket'].value_counts()
            train_dfs, eval_dfs = [], []

            for q_ in range(len(bucket_counts)):
                q_size_target_eval = (size_target_eval * bucket_counts[q_]) // bucket_counts.sum()
                q_size_target_eval += rng.binomial(1, ((size_target_eval * bucket_counts[q_]) / bucket_counts.sum()) % 1.0)
                print(q_size_target_eval, (q_size_target_eval * bucket_counts[q_]) / bucket_counts.sum())
                eval_dfs.append(df_class[df_class['length_bucket'] == q_][-q_size_target_eval:])
                train_dfs.append(df_class[df_class['length_bucket'] == q_][:-q_size_target_eval])

            df_tr = pd.concat(train_dfs)
            df_eval = pd.concat(eval_dfs)

            df_tr = df_tr.sort_values(sort_key, axis=0, ascending=(not greater_in_test), kind='mergesort',
                                      ignore_index=False)
            df_eval = df_eval.sort_values(sort_key, axis=0, ascending=greater_in_test, kind='mergesort',
                                      ignore_index=False)

            if len(df_eval) < size_target_eval:
                needed_samples = size_target_eval - len(df_eval)
                print(f"Moving {needed_samples} from train ({len(df_tr)}) to eval ({len(df_eval)})")
                df_temp, df_tr = df_tr[:needed_samples], df_tr[needed_samples:]
                df_eval = pd.concat([df_temp, df_eval])
            elif len(df_eval) > size_target_eval:
                needed_samples = len(df_eval) - size_target_eval
                print(f"Moving {needed_samples} from eval ({len(df_eval)}) to train ({len(df_tr)})")
                df_temp, df_eval = df_eval[:needed_samples], df_eval[needed_samples:]
                df_tr = pd.concat([df_tr, df_temp])

            df_eval = df_eval.sample(frac=1, random_state=rng.bit_generator)

            df_te = df_eval[:size_target_te]
            df_de = df_eval[size_target_te:]

            print(f"Final sizes for class {label}: train ({len(df_tr)}), dev ({len(df_de)}), test ({len(df_te)})")
            examples_train.append(df_tr)
            examples_dev.append(df_de)
            examples_test.append(df_te)
        else:
            # least LIKELY examples in eval set
            examples_eval = df_class[-size_target_eval:]
            examples_eval = examples_eval.sample(frac=1, random_state=rng.bit_generator)

            examples_test.append(examples_eval[:size_target_te])
            examples_dev.append(examples_eval[size_target_te:])

            # most LIKELY examples in train
            examples_train.append(df_class[:-size_target_eval])

    examples_train = pd.concat(examples_train)
    examples_dev = pd.concat(examples_dev)
    examples_test = pd.concat(examples_test)

    examples_train = examples_train.sort_values(sort_key, axis=0, ascending=(not greater_in_test), kind='mergesort',
                                                ignore_index=False)
    examples_dev = examples_dev.sort_values(sort_key, axis=0, ascending=greater_in_test, kind='mergesort',
                                            ignore_index=False)
    examples_test = examples_test.sort_values(sort_key, axis=0, ascending=greater_in_test, kind='mergesort',
                                              ignore_index=False)
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
    parser.add_argument("--lls_dir", type=str, required=True,
                        help="Path to directory containing precomputed likelihood values")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--use_ppl", action='store_true')
    parser.add_argument("--use_custom_ppl", action='store_true', help="Compute perplexity using NLTK sequence length")
    parser.add_argument("--extreme_length_buckets", action='store_true',
                        help="Set each unique length value as it's own bucket")
    parser.add_argument("--length_buckets", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args_ = parser.parse_args()

    print(vars(args_))
    main(args_)
