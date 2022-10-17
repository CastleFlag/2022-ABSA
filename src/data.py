import pandas as pd
import torch
from torch import tensor
from model import MyTokenizer
from utils import jsonlload
from torch.utils.data import DataLoader, Dataset, TensorDataset

def create_dataloader(path, tokenizer, args):
    json_data = jsonlload(path)
    entity_dataset, polarity_dataset = get_dataset(json_data, tokenizer, args.max_len, args.mode)
    return DataLoader(entity_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0), DataLoader(polarity_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

class MyDataset(Dataset):
    def __init__(self, id_list, attention_mask_list, token_label_list):
        self.ids = id_list
        self.masks = attention_mask_list
        self.labels = token_label_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        return id, mask, label


polarity_count = 0
entity_property_count = 0
label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

entity_property_pair = [
    '제품 전체#일반', '제품 전체#가격', '제품 전체#디자인', '제품 전체#품질', '제품 전체#편의성', '제품 전체#인지도',
    '본품#일반', '본품#디자인', '본품#품질', '본품#편의성', '본품#다양성',
    '패키지/구성품#일반', '패키지/구성품#디자인', '패키지/구성품#품질', '패키지/구성품#편의성', '패키지/구성품#다양성',
    '브랜드#일반', '브랜드#가격', '브랜드#디자인', '브랜드#품질', '브랜드#인지도']
entity_name_to_id = {entity_property_pair[i]: i for i in range(len(entity_property_pair))}
 

def tokenize_and_align_labels(tokenizer, form, annotations, max_len, trainortest):
    global polarity_count
    global entity_property_count

    entity_property_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }
    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }

    if trainortest == 'train':
        for pair in entity_property_pair:
            isPairInOpinion = False
            if pd.isna(form):
                break
            tokenized_data = tokenizer(form, padding='max_length', max_length=max_len, truncation=True)
            for annotation in annotations:
                entity_property = annotation[0]
                polarity = annotation[2]

                # # 데이터가 =로 시작하여 수식으로 인정된경우
                # if pd.isna(entity) or pd.isna(property):
                #     continue

                if polarity == '------------':
                    continue

                if entity_property == pair:
                    polarity_count += 1
                    entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                    entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                    entity_property_data_dict['label'].append(entity_name_to_id[entity_property])

                    polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                    polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                    polarity_data_dict['label'].append(polarity_name_to_id[polarity])

                    isPairInOpinion = True
                    break
    else:
        tokenized_data = tokenizer(form, padding='max_length', max_length=max_len, truncation=True)
        polarity_count += 1
        entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
        entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
        entity_property_data_dict['label'].append(0)

        polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
        polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
        polarity_data_dict['label'].append(0)

    return entity_property_data_dict, polarity_data_dict


def get_dataset(raw_data, tokenizer, max_len, trainortest):
    entity_input_ids_list = []
    entity_attention_mask_list = []
    entity_token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []
    
    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict  = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len, trainortest)

        entity_input_ids_list.extend(entity_property_data_dict['input_ids'])
        entity_attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        entity_token_labels_list.extend(entity_property_data_dict['label'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])
    print('entity_property_data_count: ', entity_property_count)
    print('polarity_data_count: ', polarity_count)

    return MyDataset(tensor(entity_input_ids_list), tensor(entity_attention_mask_list), tensor(entity_token_labels_list)), MyDataset(tensor(polarity_input_ids_list), tensor(polarity_attention_mask_list), tensor(polarity_token_labels_list))
            
