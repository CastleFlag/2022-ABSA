import json

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
def entity_to_id4(annotation):
    if annotation.startswith('제품'):
        return 0
    elif annotation.startswith('패키지'):
        return 1
    elif annotation.startswith('본품'):
        return 2
    elif annotation.startswith('브랜드'):
        return 3
def id4_to_entity(pred):
    if pred == 0:
        return '제품 전체#'
    elif pred == 1:
        return '패키지/구성품#'
    elif pred == 2:
        return '본품#'
    elif pred == 3:
        return '브랜드#'
def entity_to_id7(annotation):
    if annotation.endswith('일반'):
        return 0
    elif annotation.endswith('디자인'):
        return 1
    elif annotation.endswith('가격'):
        return 2
    elif annotation.endswith('품질'):
        return 3
    elif annotation.endswith('인지도'):
        return 4
    elif annotation.endswith('편의성'):
        return 5
    elif annotation.endswith('다양성'):
        return 6
def id7_to_entity(pred):
    if pred == 0:
        return '일반'
    elif pred == 1:
        return '디자인'
    elif pred == 2:
        return '가격'
    elif pred == 3:
        return '품질'
    elif pred == 4:
        return '인지도'
    elif pred == 5:
        return '편의성'
    elif pred == 6:
        return '다양성'
def simple_major(major):
    if major== '제품 전체':
        return '제품'
    elif major== '패키지/구성품':
        return '패키지'
    return major 
def get_inputs_dict(batch, tokenizer, device):
    source_ids, source_mask, label = batch["source_ids"], batch["source_mask"], batch['labels']
    inputs = {
        "input_ids": source_ids.to(device),
        "attention_mask": source_mask.to(device),
        "labels": label.to(device),
    }
    return inputs
