import datasets
from datasets import load_dataset, load_metric, Metric, MetricInfo
from semantic_parsing.spider.evaluation import eval_internal, build_foreign_key_map_from_json
import numpy as np


class SeqAcc(Metric):
    def _info(self):
        return MetricInfo(
            description="SeqAcc",
            citation=".",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[],
        )

    def _compute(self, *, predictions=None, references=None, **kwargs):
        acc = []
        for (pred, ref) in zip(predictions, references):
            pred_seq = pred.strip().split()
            ref_seq = ref.strip().split()
            if pred_seq == ref_seq:
                acc.append(1)
            else:
                acc.append(0)
        acc = np.mean(acc)
        return {"seq_accuracy": acc}


class SpiderExactMatch(Metric):
    def _info(self):
        return MetricInfo(
            description="SpiderExactMatch",
            citation=".",
            inputs_description="",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[""],
            reference_urls=[],
        )

    def _compute(self, *, predictions=None, references=None, reference_tables=None, db_dir=None, tables_map=None,
                 **kwargs):
        scores, _ = eval_internal(list(zip(references, reference_tables)), predictions, db_dir, "match", tables_map)
        return {"spider_exact_match": scores['all']['exact']}
