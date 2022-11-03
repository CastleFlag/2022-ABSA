import pandas as pd
import torch
from torch import tensor
import torch.nn.functional as F
from utils import jsonlload
from torch.utils.data import DataLoader, Dataset
from utils import simple_major
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from dictionaries import polarity_en_to_ko, major_name_to_id
class BertDataset(Dataset):
    def __init__(self, data, tokenizer, opt):
        self.tokenizer = tokenizer
        data = [
            (input_text, label, tokenizer, opt)
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
            "labels" : torch.tensor(label).float()
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

def create_dataloader(path, tokenizer, opt, big=False):
    raw_data = jsonlload(path)
    data = []
    for utterance in raw_data:
        sentence = utterance['sentence_form']
        # sentence = re.compile('[^ 0-9A-Za-z가-힣]').sub('',sentence).strip()
        annotations = utterance['annotation']
        labels = [0]*opt.num_labels
        for idx, annotation in enumerate(annotations):
            entity = annotation[0]
            major, minor = entity.split('#')
            major = simple_major(major)
            labels[major_name_to_id[major]] = 1
            polarity = polarity_en_to_ko[annotation[2]]
        data.append([sentence, labels])
    df = pd.DataFrame(data, columns=['input_text', 'label'])
    if not big:
        train_dataset = BertDataset(df, tokenizer, opt)
        return DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    if big:
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_ds = BertDataset(train_df, tokenizer, opt)
        test_ds = BertDataset(test_df, tokenizer, opt)
        return DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0)
