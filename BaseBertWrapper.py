import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertTokenizerFast
from pytorch_pretrained_bert import BertTokenizer as torchBertTokenizer, BertModel as torchBertModel


class BaseBertWrapper(nn.Module):
    def __init__(self, bert_model, do_lower_case, output_hidden_states=False, init_w=False, cuda=False, pytorch_pretrained=False):
        super().__init__()

        self.fast_tokenizer = BertTokenizerFast.from_pretrained(bert_model, do_lower_case=do_lower_case)

        if pytorch_pretrained:
            self.bert = torchBertModel.from_pretrained(bert_model)
            self.tokenizer = torchBertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        else:
            self.bert = BertModel.from_pretrained(bert_model, output_hidden_states=output_hidden_states)
            self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and cuda else "cpu")
        self.uses_pytorch_v = pytorch_pretrained

        if init_w:
            self.bert.init_weights()

        if cuda:
            self.cuda()

    def split_sentence_into_tokens(self, sentence, includes_sep=False):
        tokens = []

        if not includes_sep:
            tokens.append("[CLS]")

        for word in sentence:
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)

        if not includes_sep:
            tokens.append("[SEP]")

        return tokens

    def parse_sentences_to_bert(self, sentences):
        tokenized_data = self.fast_tokenizer(sentences, padding=True, is_split_into_words=True, return_offsets_mapping=True)
        input_ids = tokenized_data.data["input_ids"]
        attention_mask = tokenized_data.data["attention_mask"]

        word_end_mask = []
        for ir, row in enumerate(tokenized_data.data["offset_mapping"]):
            sent_mask = [1] # first is always the CLS sep
            # for each token pair
            for ipair in range(1, len(row)-1):
                # if the next token is the start of a new word
                # it means this one is the last token of this word
                # as long as this is indeed a word and not a padding
                sent_mask.append(int(
                    (row[ipair+1][0] == 0) and attention_mask[ir][ipair] == 1
                ))

            # last entry is either the last token of the last word, or a pad
            sent_mask.append(int(attention_mask[ir][-1] == 1))
            word_end_mask.append(sent_mask)

        return torch.IntTensor(input_ids).to(self.device), torch.IntTensor(attention_mask).to(self.device), torch.IntTensor(word_end_mask).to(self.device)

    def get_bert_data(self, corpus, batch_size=128):
        total_features = torch.Tensor([]).to(self.device)
        for i in tqdm(range(0, len(corpus), batch_size)):
            input_ids, att_mask, word_end_mask = self.parse_sentences_to_bert(corpus[i:i+batch_size])

            if self.uses_pytorch_v:
                features, _ = self.bert(input_ids, attention_mask=att_mask, output_all_encoded_layers=False)
                del _
            else:
                features = self.bert(input_ids, attention_mask=att_mask)["last_hidden_state"]

            features_packed = features.masked_select(word_end_mask.to(torch.bool).unsqueeze(-1)).reshape(-1, features.shape[-1])
            total_features = torch.cat((total_features, features_packed), dim=0)

        if total_features.size()[0] != sum([(len(s) + 2) for s in corpus]):
            raise Exception("Number of feature vectors does not match number of total tokens in the corpus")

        return total_features

    def forward(self, x):
        return self.get_bert_data(x)