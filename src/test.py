import argparse
from transformers import AutoModel, AutoTokenizer
from data import *
from evalutation import evaluation_f1, evaluation
from model import MyClassifier
import copy
from tqdm import tqdm
from utils import jsondump, id4_to_entity, id7_to_entity, entity_to_id4, entity_to_id7
from datetime import datetime

special_tokens_dict = {
'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}
entity_property_pair= [   
        '제품 전체#일반', '제품 전체#디자인','제품 전체#가격','제품 전체#품질','제품 전체#인지도', '제품 전체#편의성','제품 전체#다양성',
        '패키지/구성품#일반', '패키지/구성품#디자인','패키지/구성품#가격','패키지/구성품#품질''패키지/구성품#다양성', '패키지/구성품#편의성',
        '본품#일반', '본품#디자인','본품#가격', '본품#품질','본품#다양성','본품#인지도','본품#편의성',  
        '브랜드#일반', '브랜드#디자인', '브랜드#가격', '브랜드#품질', '브랜드#인지도']

def test(opt, device):
    print(opt.base_model)
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = jsonlload(opt.test_data)

    
    model4 = MyClassifier(opt, 4, len(tokenizer))
    model4.load_state_dict(torch.load(opt.entity4_model_path, map_location=device))
    model4.to(device)
    model4.eval()

    model7 = MyClassifier(opt, 7, len(tokenizer))
    model7.load_state_dict(torch.load(opt.entity7_model_path, map_location=device))
    model7.to(device)
    model7.eval()

    polarity_model = MyClassifier(opt, 3, len(tokenizer))
    polarity_model.load_state_dict(torch.load(opt.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form(tokenizer, model4, model7, polarity_model, copy.deepcopy(test_data))

    now = datetime.now()
    current_day = now.strftime('%m%d')

    jsondump(pred_data, opt.output_dir + opt.base_model +'_'+ current_day + '.json')
    print(opt.output_dir + opt.base_model +'_'+ current_day + '.json')
    # pred_data = jsonload('./pred_data.json')
    # print('F1 result: ', evaluation_f1(test_data, pred_data))

    # entity_test_dataloader, polarity_test_dataloader = create_dataloader(opt.test_data, tokenizer, opt)
    # pred_list = []
    # label_list = []
    # print('polarity classification result')
    # for batch in polarity_test_dataloader:
    #     batch = tuple(t.to(device) for t in batch)
    #     b_input_ids, b_input_mask, b_labels = batch

    #     with torch.no_grad():
    #         loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)

    #     predictions = torch.argmax(logits, dim=-1)
    #     pred_list.extend(predictions)
    #     label_list.extend(b_labels)

    # evaluation(label_list, pred_list, len(polarity_id_to_name))

def predict_from_korean_form(tokenizer, ce4_model, ce7_model, pc_model, data):
    ce4_model.to(device)
    ce4_model.to(device)
    ce7_model.eval()
    ce7_model.eval()
    count = 0
    for sentence in data:
        form = sentence['sentence_form']
        sentence['annotation'] = []
        count += 1
        if type(form) != str:
            print("form type is arong: ", form)
            continue
        tokenized_data = tokenizer(form, padding='max_length', max_length=256, truncation=True)
        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        with torch.no_grad():
            _, ce4_logits = ce4_model(input_ids, attention_mask)
            _, ce7_logits = ce7_model(input_ids, attention_mask)
        ce4_predictions = torch.argmax(ce4_logits, dim = -1)
        ce4_result = id4_to_entity(ce4_predictions[0])
        ce7_predictions = torch.argmax(ce7_logits, dim = -1)
        ce7_result = id7_to_entity(ce7_predictions[0])

        # if ce_result == 'True':
        with torch.no_grad():
            _, pc_logits = pc_model(input_ids, attention_mask)

        pc_predictions = torch.argmax(pc_logits, dim=-1)
        pc_result = polarity_id_to_name[pc_predictions[0]]

        sentence['annotation'].append([ce4_result+ce7_result, pc_result])
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--test_data", type=str, default="../data/nikluge-sa-2022-dev.jsonl", help="test file")
    parser.add_argument( "--base_model", type=str, default="bert-base-multilingual-uncased")
    parser.add_argument( "--batch_size", type=int, default=8) 
    parser.add_argument( "--num_labels", type=int, default=1)
    parser.add_argument( "--entity4_model_path", type=str, default="../saved_model/best_model/4.pt")
    parser.add_argument( "--entity7_model_path", type=str, default="../saved_model/best_model/7.pt")
    parser.add_argument( "--polarity_model_path", type=str, default="../saved_model/best_model/polarity.pt")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--classifier_hidden_size", type=int, default=768)
    parser.add_argument( "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(opt, device)

