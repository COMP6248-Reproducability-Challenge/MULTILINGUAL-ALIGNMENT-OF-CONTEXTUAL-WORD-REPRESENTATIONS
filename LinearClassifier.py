import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.fc1 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(0.1)
        self.bert = bert_model

    def forward(self, x):
        out = self.bert(x)[1]
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.softmax(out, dim=1)
        return out
