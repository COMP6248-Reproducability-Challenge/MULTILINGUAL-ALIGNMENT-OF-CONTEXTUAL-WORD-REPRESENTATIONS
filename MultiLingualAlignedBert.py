import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig


class BaseBertWrapper(nn.Module):
    def __init__(self, bert_model, do_lower_case, output_hidden_states=True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.bert = BertModel.from_pretrained(bert_model, output_hidden_states=output_hidden_states)

    def create_parallel_sentences(self):
        pass

    def forward(self):
        pass
