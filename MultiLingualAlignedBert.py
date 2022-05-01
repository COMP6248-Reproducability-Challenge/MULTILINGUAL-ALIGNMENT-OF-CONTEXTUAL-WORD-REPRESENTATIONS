from transformers import BertTokenizer, BertModel, BertConfig
from BaseBertWrapper import BaseBertWrapper


class MultiLingualAlignedBertWrapper(BaseBertWrapper):
    def __init__(self, bert_model, do_lower_case):
        super(MultiLingualAlignedBertWrapper, self).__init__(bert_model, do_lower_case)

    def forward(self, corpus):
        return self.get_bert_data(corpus)
