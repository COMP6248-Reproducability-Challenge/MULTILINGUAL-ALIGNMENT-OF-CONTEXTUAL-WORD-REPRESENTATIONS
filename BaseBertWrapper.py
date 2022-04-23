from transformers import BertTokenizer, BertModel, BertConfig


class BaseBertWrapper():
    def __init__(self, bert_model, do_lower_case, output_hidden_states=True):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        self.bert = BertModel.from_pretrained(bert_model, output_hidden_states=output_hidden_states)
        self.bert.init_weights()

