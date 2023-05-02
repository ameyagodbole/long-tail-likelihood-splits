import argparse
import os
from collections import Counter
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                             'google-research-language')))
from language.nqg.tasks import mcd_utils
from language.nqg.tasks import tsv_utils
from language.nqg.tasks.spider import tmcd_utils as SpiderTMCDUtils
from language.nqg.tasks.geoquery import tmcd_utils as GeoQueryTMCDUtils


def main(args):
    examples_1 = tsv_utils.read_tsv(args.input_1)
    examples_2 = tsv_utils.read_tsv(args.input_2)
    TMCDUtils = SpiderTMCDUtils if args.dataset_name == 'spider' else GeoQueryTMCDUtils
    atom_counts_1 = Counter()
    for ex in examples_1:
        atom_counts_1.update(TMCDUtils.get_atom_counts(ex[1]))
    atom_counts_2 = Counter()
    for ex in examples_2:
        atom_counts_2.update(TMCDUtils.get_atom_counts(ex[1]))
    divergence = mcd_utils._compute_divergence(atom_counts_1, atom_counts_2, coef=0.5)
    print("Atom divergence: %s" % divergence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_1", type=str)
    parser.add_argument("--input_2", type=str)
    parser.add_argument("--dataset_name", type=str)
    args_ = parser.parse_args()

    main(args_)
