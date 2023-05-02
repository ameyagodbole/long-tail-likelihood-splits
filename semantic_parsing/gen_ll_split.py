import argparse
import os
from nltk import word_tokenize
import numpy as np
import pandas as pd
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                             'google-research-language')))
from language.nqg.tasks import mcd_utils
from language.nqg.tasks.spider.gen_tmcd_split import AtomAndCompoundCache as SpiderAtomAndCompoundCache
from language.nqg.tasks.geoquery import tmcd_utils as GeoQueryTMCDUtils
from language.nqg.tasks import tsv_utils

TRAIN_SPLIT_SZ = {'spider': 5966, 'geoquery': 440}


def main(args):
    """
    Reference: https://huggingface.co/transformers/perplexity.html
    """
    rng = np.random.default_rng(args.seed)
    df_all = pd.read_csv(os.path.join(args.raw_data), sep='\t', header=None, names=['query', 'parse'])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_lls = np.load(args.precomputed_lls)
    all_ppls = np.load(args.precomputed_ppls)
    df_all["ll_score"] = all_lls
    df_all["ppl_score"] = all_ppls
    df_all['length'] = df_all['query'].apply(lambda x: len(word_tokenize(
        x.split(':', 1)[1].split(' | ', 1)[0].strip())))
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

    if args.length_buckets > 1 or args.extreme_length_buckets:
        if args.extreme_length_buckets:
            uniq_lens, df_all["length_bucket"], uniq_len_counts = np.unique(df_all["length"], return_inverse=True,
                                                                            return_counts=True)
            print(f"Uniq lengths: {uniq_lens}")
            print(f"Lengths counts: {uniq_len_counts}")
        else:
            quantiles = [q_ / args.length_buckets for q_ in range(args.length_buckets + 1)]
            df_all["length_bucket"] = pd.qcut(df_all["length"], quantiles, labels=False)
        bucket_counts = df_all['length_bucket'].value_counts()
        train_dfs, test_dfs = [], []
        size_target_tr = TRAIN_SPLIT_SZ[args.dataset_name]

        for q_ in range(len(bucket_counts)):
            q_size_target_tr = (size_target_tr * bucket_counts[q_]) // bucket_counts.sum()
            q_size_target_tr += rng.binomial(1, ((size_target_tr * bucket_counts[q_]) / bucket_counts.sum()) % 1.0)
            print(q_size_target_tr, (size_target_tr * bucket_counts[q_]) / bucket_counts.sum())
            train_dfs.append(df_all[df_all['length_bucket'] == q_][:q_size_target_tr])
            test_dfs.append(df_all[df_all['length_bucket'] == q_][q_size_target_tr:])

        df_tr = pd.concat(train_dfs)
        df_te = pd.concat(test_dfs)

        df_tr = df_tr.sort_values(sort_key, axis=0, ascending=(not greater_in_test), kind='mergesort',
                                  ignore_index=False)
        df_te = df_te.sort_values(sort_key, axis=0, ascending=greater_in_test, kind='mergesort',
                                  ignore_index=False)
        if len(df_tr) < TRAIN_SPLIT_SZ[args.dataset_name]:
            needed_samples = TRAIN_SPLIT_SZ[args.dataset_name] - len(df_tr)
            print(f"Moving {needed_samples} from test ({len(df_te)}) to train ({len(df_tr)})")
            df_temp, df_te = df_te[:needed_samples], df_te[needed_samples:]
            df_tr = pd.concat([df_temp, df_tr])
            print(f"Final sizes: train ({len(df_tr)}), test ({len(df_te)})")
        elif len(df_tr) > TRAIN_SPLIT_SZ[args.dataset_name]:
            needed_samples = len(df_tr) - TRAIN_SPLIT_SZ[args.dataset_name]
            print(f"Moving {needed_samples} from train ({len(df_tr)}) to test ({len(df_te)})")
            df_temp, df_tr = df_tr[:needed_samples], df_tr[needed_samples:]
            df_te = pd.concat([df_temp, df_te])
            print(f"Final sizes: train ({len(df_tr)}), test ({len(df_te)})")

        examples_train = [(row["query"], row["parse"], r_ctr) for r_ctr, row in df_tr.iterrows()]
        examples_test = [(row["query"], row["parse"], r_ctr) for r_ctr, row in df_te.iterrows()]
    else:
        examples = [(row["query"], row["parse"], r_ctr) for r_ctr, row in df_all.iterrows()]

        # most LIKELY test examples are at the top and will be swapped to train preferrentially
        examples_test = examples[TRAIN_SPLIT_SZ[args.dataset_name]:]
        # most UNLIKELY train examples are at the top and will be swapped to test preferrentially
        examples_train = examples[:TRAIN_SPLIT_SZ[args.dataset_name]][::-1]

    if args.dataset_name == 'spider':
        cache = SpiderAtomAndCompoundCache()
        # Swap examples to ensure train has at least one example for every atom in test
        examples_train, examples_test = mcd_utils.balance_atoms(examples_train, examples_test, cache.get_atoms,
                                                                max_iterations=10000)
    elif args.dataset_name == 'geoquery':
        # Swap examples to ensure train has at least one example for every atom in test
        examples_train, examples_test = mcd_utils.balance_atoms(examples_train, examples_test,
                                                                GeoQueryTMCDUtils.get_example_atoms,
                                                                max_iterations=10000)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset_name}")

    n_dev = len(examples_test) // 2
    shuffled_examples_test = [examples_test[rand_i] for rand_i in rng.permutation(len(examples_test))]
    examples_dev, examples_test = shuffled_examples_test[:n_dev], shuffled_examples_test[n_dev:]

    ex_train = [tr_idx[2] for tr_idx in examples_train]
    ex_dev = [de_idx[2] for de_idx in examples_dev]
    ex_test = [te_idx[2] for te_idx in examples_test]
    with open(os.path.join(args.output_dir, "train_idx.json"), 'w') as fout:
        json.dump(ex_train, fout, indent=4)
    with open(os.path.join(args.output_dir, "dev_idx.json"), 'w') as fout:
        json.dump(ex_dev, fout, indent=4)
    with open(os.path.join(args.output_dir, "test_idx.json"), 'w') as fout:
        json.dump(ex_test, fout, indent=4)
    tsv_utils.write_tsv(examples_train, os.path.join(args.output_dir, "train.tsv"))
    tsv_utils.write_tsv(examples_dev, os.path.join(args.output_dir, "dev.tsv"))
    tsv_utils.write_tsv(examples_test, os.path.join(args.output_dir, "test.tsv"))
    pd.read_csv(os.path.join(args.output_dir, "train.tsv"), sep='\t', header=None, names=["query", "parse"]).to_csv(
        os.path.join(args.output_dir, "train.csv"), index=False)
    pd.read_csv(os.path.join(args.output_dir, "dev.tsv"), sep='\t', header=None, names=["query", "parse"]).to_csv(
        os.path.join(args.output_dir, "dev.csv"), index=False)
    pd.read_csv(os.path.join(args.output_dir, "test.tsv"), sep='\t', header=None, names=["query", "parse"]).to_csv(
        os.path.join(args.output_dir, "test.csv"), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--precomputed_lls", type=str, required=True, help="Path to precomputed likelihood values")
    parser.add_argument("--precomputed_ppls", type=str, required=True, help="Path to precomputed perplexity values")
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
