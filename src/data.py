import pandas as pd
import torch
from torch import tensor
import torch.nn.functional as F
from model import MyTokenizer
from utils import jsonlload
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import simple_major
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, opt):
        self.tokenizer = tokenizer
        data = [
            ('<CLS>'+input_text+'<SEP>', label, tokenizer, opt)
            for input_text, label in zip(
                data['input_text'], data['label']
            )
        ]
        preprocess_fn = (
            self.preprocess_data_bert
        )

        self.examples = [
            preprocess_fn(d) for d in tqdm(data, disable=True)
        ]
    def preprocess_data_bert(self, data):
        input_text, label, tokenizer, opt = data

        input_ids = tokenizer.batch_encode_plus(
            [input_text],
            max_length=opt.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "labels" : torch.tensor(label)
        }
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]
def get_inputs_dict(batch, tokenizer, device):
    source_ids, source_mask, label = batch["source_ids"], batch["source_mask"], batch['labels']
    inputs = {
        "input_ids": source_ids.to(device),
        "attention_mask": source_mask.to(device),
        "labels": label.to(device),
    }
    return inputs
polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}
entity_property_pair= [   
        '제품 전체#일반', '제품 전체#디자인','제품 전체#가격','제품 전체#품질','제품 전체#인지도', '제품 전체#편의성','제품 전체#다양성',
        '패키지/구성품#일반', '패키지/구성품#디자인','패키지/구성품#가격','패키지/구성품#품질''패키지/구성품#다양성', '패키지/구성품#편의성',
        '본품#일반', '본품#디자인','본품#가격', '본품#품질','본품#다양성','본품#인지도','본품#편의성',  
        '브랜드#일반', '브랜드#디자인', '브랜드#가격', '브랜드#품질', '브랜드#인지도']

polarity_en_to_ko ={
    'positive' : '긍정적',
    'negative' : '부정적',
    'neutral' : '중립적'
}
major_id_to_name = ['제품', '패키지', '본품', '브랜드']
major_name_to_id = { major_id_to_name[i]: i for i in range(len(major_id_to_name)) }

def create_dataloader(path, tokenizer, opt, big=False):
    raw_data = jsonlload(path)
    data = []
    for utterance in raw_data:
        sentence = utterance['sentence_form']
        sentence = re.compile('[^ 0-9A-Za-z가-힣]').sub('',sentence).strip()
        annotations = utterance['annotation']
        for annotation in annotations:
            entity = annotation[0]
            major, minor = entity.split('#')
            major = simple_major(major)
            polarity = polarity_en_to_ko[annotation[2]]
            data.append([sentence, major_name_to_id[major]])
    df = pd.DataFrame(data, columns=['input_text', 'label'])
    if not big:
        train_dataset = MyDataset(df, tokenizer, opt)
        return DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    if big:
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_ds = MyDataset(train_df, tokenizer, opt)
        test_ds = MyDataset(test_df, tokenizer, opt)
        return DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0)
