import torch.nn as nn

class MentalBERTClassifier(nn.Module):
    def __init__(self, bert_model, dropout_rate=0.3):
        super(MentalBERTClassifier, self).__init__()
        self.bert = bert_model

        for param in list(self.bert.parameters())[:6]:
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        return self.classifier(self.dropout(pooled_output))