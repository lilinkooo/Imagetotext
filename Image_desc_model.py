import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class ImageCaptioningModel(nn.Module):
    def __init__(self, bert_model, hidden_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.bert = bert_model
        self.rnn = nn.LSTM(bert_model.config.hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image_features, input_ids):
        bert_output = self.bert(input_ids)[0]
        rnn_output, _ = self.rnn(bert_output)
        output = self.fc(rnn_output)
        return output