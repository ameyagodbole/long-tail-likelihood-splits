import json
import os
import pandas as pd


def boolq_dataset_preprocessor(example):
    return example


class BoolQ_Instance:
    def __init__(self, passage, question, answer, unused_fields=None):
        self.passage = passage
        self.question = question
        self.answer = answer
        self.unused_fields = unused_fields

    @staticmethod
    def _get_ll_query(passage, question, **kwargs):
        return f"Passage: {passage} Ask a question about the passage:", ' ' + question

    def get_ll_query(self, **kwargs):
        return self._get_ll_query(self.passage, self.question)

    @classmethod
    def df_to_instance_list(cls, df):
        instance_list = []
        for _, row in df.iterrows():
            inst_dict = row.to_dict()
            passage_ = inst_dict.pop('passage')
            question_ = inst_dict.pop('question')
            answer_ = inst_dict.pop('label')
            instance_list.append(BoolQ_Instance(passage_, question_, answer_, inst_dict))
        return instance_list


class BoolQ_Partition:
    def __init__(self, data_dir, partition, mode='processed'):
        self.data_dir = data_dir
        self.partition = partition
        self.mode = mode
        self.instances: list[BoolQ_Instance] = []
        self.load_boolq(data_dir, partition)

    def load_boolq(self, data_dir, partition):
        if self.mode == 'original':
            raw_data = []
            with open(os.path.join(data_dir, f"{partition}.jsonl")) as fin:
                for line in fin:
                    raw_data.append(json.loads(line))
            key_set = list(raw_data[0].keys())
            raw_data_dict = {}
            for k_ in key_set:
                raw_data_dict[k_] = [row[k_] for row in raw_data]
            df = pd.DataFrame(raw_data_dict)
        elif self.mode == 'processed':
            df = pd.read_csv(os.path.join(data_dir, f"{partition}.csv"))
        else:
            raise NotImplementedError(f"Unsupported mode {self.mode}")
        self.instances = BoolQ_Instance.df_to_instance_list(df)
        return df

    def get_ll_query(self, **kwargs):
        return [inst.get_ll_query() for inst in self.instances]

    def __len__(self):
        return len(self.instances)
