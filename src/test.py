import argparse
from transformers import AutoModel, AutoTokenizer
from data import *
from evalutation import evaluation_f1, evaluation
from model import MyClassifier
import copy
from tqdm import tqdm
from utils import jsondump

special_tokens_dict = {
'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}

entity_property_pair =[
    '제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', 
    '본품#다양성', '제품 전체#디자인', '패키지/ 구성품#가격', '본품#품질', '브랜드#인지도', 
    '제품 전체#일반', '브랜드#일반', '패키지/구성품#다양성', 
    '패키지/구성품#일반', '본품#인지도', '제품 전체#가격', '본품#편의성', '패키지/구성품#편의성', 
    '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', '제품 전체#인지도', '패키지/구성품#품질', 
    '제품 전체#다양성', '브랜드#가격']
entity_name_to_id = {entity_property_pair[i]: i for i in range(len(entity_property_pair))}

def test(opt, device):
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = jsonlload(opt.test_data)

    entity_test_dataloader, polarity_test_dataloader = create_dataloader(opt.test_data, tokenizer, opt)
    
    model = MyClassifier(opt, 25, len(tokenizer))
    model.load_state_dict(torch.load(opt.entity_model_path, map_location=device))
    model.to(device)
    model.eval()

    polarity_model = MyClassifier(opt, 3, len(tokenizer))
    polarity_model.load_state_dict(torch.load(opt.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form(tokenizer, model, polarity_model, copy.deepcopy(test_data))

    jsondump(pred_data, './pred_data.json')
    # pred_data = jsonload('./pred_data.json')
    print(pred_data)
    print(test_data)
    print('F1 result: ', evaluation_f1(test_data, pred_data))

    pred_list = []
    label_list = []
    print('polarity classification result')
    for batch in polarity_test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)

        predictions = torch.argmax(logits, dim=-1)
        pred_list.extend(predictions)
        label_list.extend(b_labels)

    evaluation(label_list, pred_list, len(polarity_id_to_name))

def predict_from_korean_form(tokenizer, ce_model, pc_model, data):

    ce_model.to(device)
    ce_model.eval()
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
            _, ce_logits = ce_model(input_ids, attention_mask)
        ce_predictions = torch.argmax(ce_logits, dim = -1)
        ce_result = entity_property_pair[ce_predictions[0]]

        # if ce_result == 'True':
        with torch.no_grad():
            _, pc_logits = pc_model(input_ids, attention_mask)

        pc_predictions = torch.argmax(pc_logits, dim=-1)
        pc_result = polarity_id_to_name[pc_predictions[0]]

        sentence['annotation'].append([ce_result, pc_result])
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--train_target", type=str, default="Entity", help="train entity or polarity")
    parser.add_argument( "--train_data", type=str, default="../data/nikluge-sa-2022-train.jsonl", help="train file")
    parser.add_argument( "--test_data", type=str, default="../data/nikluge-sa-2022-test.jsonl", help="test file")
    parser.add_argument( "--dev_data", type=str, default="../data/inikluge-sa-2022-dev.jsonl", help="dev file")
    parser.add_argument( "--batch_size", type=int, default=8) 
    parser.add_argument( "--learning_rate", type=float, default=3e-5) 
    parser.add_argument( "--eps", type=float, default=1e-8)
    # parser.add_argument( "--do_train", action="store_true")
    # parser.add_argument( "--do_eval", action="store_true")
    parser.add_argument( "--do_test", action="store_true")
    parser.add_argument( "--num_train_epochs", type=int, default=10)
    parser.add_argument( "--base_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument( "--entity_model_path", type=str, default="./saved_models/entity_model/")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/polarity_model/")
    parser.add_argument( "--output_dir", type=str, default="./output/default_path/")
    parser.add_argument( "--do_demo", action="store_true")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--classifier_hidden_size", type=int, default=768)
    parser.add_argument( "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier")
    parser.add_argument( "--mode", type=str, default='test', help="train or test")
    opt = parser.parse_args()
    print(opt.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if opt.do_train:
    #     train_sentiment_analysis(args)
    # elif opt.do_demo:
    #     demo_sentiment_analysis(args)
    # elif opt.do_test:
    #     test_sentiment_analysis(args)
    test(opt, device)

