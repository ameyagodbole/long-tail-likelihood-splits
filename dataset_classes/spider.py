import os
import pandas as pd


def spider_dataset_preprocessor(example):
    result = example.copy()
    query_has_table_name = ':' in result["query"]
    query_has_table_schema = ' | ' in result["query"]
    if query_has_table_name:
        result["table_name"] = result["query"].split(':', 1)[0]
        result["query"] = result["query"].split(':', 1)[1]
    else:
        result["table_name"] = ''
    if query_has_table_schema:
        result["table_schema_str"] = ' | ' + result["query"].split(' | ', 1)[1]
        result["query"] = result["query"].split(' | ', 1)[0]
    else:
        result["table_schema_str"] = ''
    return result


class Spider_Instance:
    def __init__(self, query, parse, table_name, table_schema_str, unused_fields=None):
        self.query = query
        self.parse = parse
        self.table_name = table_name
        self.table_schema_str = table_schema_str
        self.unused_fields = unused_fields

    @staticmethod
    def _get_ll_query(prefix, query, table_name, table_schema_str,
                      space_after_prefix=False, db_id_as_prefix=False, **kwargs):
        if db_id_as_prefix:
            prefix_str = f"{prefix}{table_name}:"
        else:
            prefix_str = f"{prefix}"
        if space_after_prefix:
            query_str = ' ' + query.strip()
        else:
            query_str = query.strip()

        return prefix_str, query_str

    def get_ll_query(self, prefix='', space_after_prefix=False, db_id_as_prefix=False, **kwargs):
        return self._get_ll_query(prefix, self.query, self.table_name, self.table_schema_str,
                                  space_after_prefix, db_id_as_prefix)

    @classmethod
    def df_to_instance_list(cls, df):
        instance_list = []
        for _, row in df.iterrows():
            inst_dict = row.to_dict()
            query_ = inst_dict.pop('query')
            parse_ = inst_dict.pop('parse')
            table_name_ = inst_dict.pop('table_name')
            table_schema_str_ = inst_dict.pop('table_schema_str')
            instance_list.append(Spider_Instance(query_, parse_, table_name_, table_schema_str_, inst_dict))
        return instance_list


class Spider_Partition:
    def __init__(self, data_dir, partition):
        self.data_dir = data_dir
        self.partition = partition
        self.instances: list[Spider_Instance] = []
        self.load_spider(data_dir, partition)

    def load_spider(self, data_dir, partition):
        if os.path.exists(os.path.join(data_dir, f'{partition}.tsv')):
            df = pd.read_csv(os.path.join(data_dir, f'{partition}.tsv'), sep='\t', header=None, names=['query', 'parse'])
        else:
            df = pd.read_csv(os.path.join(data_dir, f'{partition}.csv'))
        query_has_table_name = df["query"].apply(lambda q_str: ':' in q_str)
        query_has_table_schema = df["query"].apply(lambda q_str: ' | ' in q_str)
        if query_has_table_name.all():
            df["table_name"] = df["query"].apply(lambda q_str: q_str.split(':', 1)[0])
            df["query"] = df["query"].apply(lambda q_str: q_str.split(':', 1)[1])
        else:
            df["table_name"] = ''
        if query_has_table_schema.all():
            df["table_schema_str"] = df["query"].apply(lambda q_str: ' | ' + q_str.split(' | ', 1)[1])
            df["query"] = df["query"].apply(lambda q_str: q_str.split(' | ', 1)[0])
        else:
            df["table_schema_str"] = ''

        self.instances = Spider_Instance.df_to_instance_list(df)
        return df

    def get_ll_query(self, prefix='', space_after_prefix=False, db_id_as_prefix=False, **kwargs):
        return [inst.get_ll_query(prefix, space_after_prefix, db_id_as_prefix) for inst in self.instances]

    def __len__(self):
        return len(self.instances)
