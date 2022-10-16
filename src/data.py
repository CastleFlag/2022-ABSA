import json 
import pandas as pd
import torch
from model import MyTokenizer
from torch.utils.data import DataLoader, Dataset, TensorDataset

def create_dataloader(path, tokenizer, args):
    json_data = jsonlload(path)
    entity_dataset = get_entity_dataset(json_data, tokenizer, args.max_len)
    return DataLoader(entity_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

def create_dataset():
    pass

class EntityDataset(Dataset):
    def __init__(self, id_list, attention_mask_list, token_label_list):
        self.ids = id_list
        self.masks = attention_mask_list
        self.labels = token_label_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = torch.tensor(self.ids[idx])
        mask = torch.tensor(self.masks[idx])
        label = torch.tensor(self.labels[idx])
        return id, mask, label
class PolalityDataset(Dataset):
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



def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)

# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

polarity_count = 0
entity_property_count = 0
label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

entity_property_pair = ['제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성', 
'제품 전체#디자인', '패키지/ 구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반',
 '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', '본품#인지도', '제품 전체#가격',
 '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', 
'제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격']
entity_name_to_id = {entity_property_pair[i]: i for i in range(len(entity_property_pair))}
 

def tokenize_and_align_labels(tokenizer, form, annotations, max_len):

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
    print(form)
    print(len(tokenizer))
    tokenized_data = tokenizer(form, padding='max_length', max_length=max_len, truncation=True)
    print(tokenized_data)
    for annotation in annotations:
        entity_property = annotation[0]
        polarity = annotation[2]

        entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
        entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
        entity_property_data_dict['label'].append(entity_name_to_id[entity_property])
    # for pair in entity_property_pair:
    #     isPairInOpinion = False
    #     if pd.isna(form):
    #         break
    #     tokenized_data = tokenizer(form, pair, padding='max_length', max_length=max_len, truncation=True)
    #     for annotation in annotations:
    #         entity_property = annotation[0]
    #         polarity = annotation[2]

    #         # # 데이터가 =로 시작하여 수식으로 인정된경우
    #         # if pd.isna(entity) or pd.isna(property):
    #         #     continue

    #         if polarity == '------------':
    #             continue

    #         if entity_property == pair:
    #             polarity_count += 1
    #             entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
    #             entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    #             entity_property_data_dict['label'].append(label_name_to_id['True'])

    #             polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
    #             polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    #             polarity_data_dict['label'].append(polarity_name_to_id[polarity])

    #             isPairInOpinion = True
    #             break

    #     if isPairInOpinion is False:
    #         entity_property_count += 1
    #         entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
    #         entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    #         entity_property_data_dict['label'].append(label_name_to_id['False'])

    return entity_property_data_dict, polarity_data_dict


def get_entity_dataset(raw_data, tokenizer, max_len):
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []
    cnt=0
    for utterance in raw_data:
        if cnt > 100:
            break
        cnt += 1
        entity_property_data_dict, _ = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_labels_list.extend(entity_property_data_dict['label'])
    print(input_ids_list)
    print('entity_property_data_count: ', entity_property_count)
    return EntityDataset(input_ids_list, attention_mask_list, token_labels_list)

def get_polarity_dataset(raw_data, tokenizer, max_len):
    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        _, polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    print('polarity_data_count: ', polarity_count)
    return PolalityDataset(polarity_input_ids_list, polarity_attention_mask_list, polarity_token_labels_list)