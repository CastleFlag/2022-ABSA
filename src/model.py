from re import L
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer

class SimpleClassifier(nn.Module):
    def __init__(self, args, num_label):
        super().__init__()
        self.dense = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)
        self.output = nn.Linear(args.classifier_hidden_size, num_label)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class MyClassifier(nn.Module):
    def __init__(self, args, num_label, len_tokenizer):
        super(MyClassifier, self).__init__()

        self.num_label = num_label
        self.xlm_roberta = AutoModel.from_pretrained(args.base_model)
        self.xlm_roberta.resize_token_embeddings(len_tokenizer)

        self.labels_classifier = SimpleClassifier(args, self.num_label)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = outputs[0]
        logits = self.labels_classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_label),
                                                labels.view(-1))

        return loss, logits

class MyTokenizer(AutoTokenizer):
    entity_property_pair =[
    '제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', 
    '본품#다양성', '제품 전체#디자인', '패키지/ 구성품#가격', '본품#품질', '브랜드#인지도', 
    '제품 전체#일반', '브랜드#일반', '패키지/구성품#다양성', 
    '패키지/구성품#일반', '본품#인지도', '제품 전체#가격', '본품#편의성', '패키지/구성품#편의성', 
    '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', '제품 전체#인지도', '패키지/구성품#품질', 
    '제품 전체#다양성', '브랜드#가격']
    special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
    }


    def __init__(self, basemodel):
        super().from_pretrained(basemodel)
        super().add_special_tokens(MyTokenizer.special_tokens_dict)
