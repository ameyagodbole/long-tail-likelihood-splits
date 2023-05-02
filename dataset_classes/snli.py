import json
import os
import pandas as pd


def snli_dataset_preprocessor(example):
    result = example.copy()
    key_set = list(result.keys())
    for k_ in key_set:
        if k_.startswith('sentence1'):
            new_k = k_.replace('sentence1', 'premise')
            result[new_k] = result[k_]
            del result[k_]
        if k_.startswith('sentence2'):
            new_k = k_.replace('sentence2', 'hypothesis')
            result[new_k] = result[k_]
            del result[k_]
    if 'gold_label' in key_set:
        result['label'] = result['gold_label']
    return result


class SNLI_Instance:
    def __init__(self, premise, hypothesis, label, unused_fields=None):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label
        self.unused_fields = unused_fields

    @staticmethod
    def _get_ll_query(premise, hypothesis, label, with_label=False, **kwargs):
        if with_label:
            if label == 'contradiction':
                return f"Premise: {premise} This hypothesis is a contradiction:", ' ' + hypothesis
            elif label == 'entailment':
                return f"Premise: {premise} This hypothesis is entailed:", ' ' + hypothesis
            else:
                return f"Premise: {premise} This hypothesis is neutral:", ' ' + hypothesis
        else:
            return f"Premise: {premise} Hypothesis:", ' ' + hypothesis

    def get_ll_query(self, with_label=False, **kwargs):
        return self._get_ll_query(self.premise, self.hypothesis, self.label, with_label=with_label)

    @classmethod
    def df_to_instance_list(cls, df):
        instance_list = []
        for _, row in df.iterrows():
            inst_dict = row.to_dict()
            premise_ = inst_dict.pop('premise')
            hypothesis_ = inst_dict.pop('hypothesis')
            label_ = inst_dict.pop('gold_label')
            instance_list.append(SNLI_Instance(premise_, hypothesis_, label_, inst_dict))
        return instance_list


class SNLI_Partition:
    def __init__(self, data_dir, partition, mode='processed'):
        self.data_dir = data_dir
        self.partition = partition
        self.mode = mode
        self.instances: list[SNLI_Instance] = []
        self.load_snli(data_dir, partition)

    def load_snli(self, data_dir, partition):
        if self.mode == 'original':
            raw_data = []
            with open(os.path.join(data_dir, f"snli_1.0_{partition}.jsonl")) as fin:
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
        elif self.mode == 'processed':
            df = pd.read_csv(os.path.join(data_dir, f"{partition}.csv"))
        else:
            raise NotImplementedError(f"Unsupported mode {self.mode}")
        self.instances = SNLI_Instance.df_to_instance_list(df)
        return df

    def get_ll_query(self, with_label=False, **kwargs):
        return [inst.get_ll_query(with_label) for inst in self.instances]

    def __len__(self):
        return len(self.instances)
