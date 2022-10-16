import argparse
from transformers import AutoModel, AutoTokenizer
from data import *
from evalutation import evaluation_f1, evaluation
from model import MyClassifier
from tqdm import tqdm
import os

def test(opt, device):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    test_data = jsonlload(args.test_data)

    entity_property_test_data, polarity_test_data = get_dataset(test_data, tokenizer, args.max_len)
    entity_property_test_dataloader = DataLoader(entity_property_test_data, shuffle=True,
                                batch_size=args.batch_size)

    polarity_test_dataloader = DataLoader(polarity_test_data, shuffle=True,
                                                  batch_size=args.batch_size)
    
    model = RoBertaBaseClassifier(args, len(label_id_to_name), len(tokenizer))
    model.load_state_dict(torch.load(args.entity_property_model_path, map_location=device))
    model.to(device)
    model.eval()
            
    polarity_model = RoBertaBaseClassifier(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.load_state_dict(torch.load(args.polarity_model_path, map_location=device))
    polarity_model.to(device)
    polarity_model.eval()

    pred_data = predict_from_korean_form(tokenizer, model, polarity_model, copy.deepcopy(test_data))

    # jsondump(pred_data, './pred_data.json')
    # pred_data = jsonload('./pred_data.json')

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
        for pair in entity_property_pair:
            tokenized_data = tokenizer(form, pair, padding='max_length', max_length=256, truncation=True)

            input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
            attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
            with torch.no_grad():
                _, ce_logits = ce_model(input_ids, attention_mask)

            ce_predictions = torch.argmax(ce_logits, dim = -1)

            ce_result = label_id_to_name[ce_predictions[0]]

            if ce_result == 'True':
                with torch.no_grad():
                    _, pc_logits = pc_model(input_ids, attention_mask)

                pc_predictions = torch.argmax(pc_logits, dim=-1)
                pc_result = polarity_id_to_name[pc_predictions[0]]

                sentence['annotation'].append([pair, pc_result])
    return data



def train(opt, device):
    if not os.path.exists(opt.entity_property_model_path):
        os.makedirs(opt.entity_property_model_path)
    if not os.path.exists(opt.polarity_model_path):
        os.makedirs(opt.polarity_model_path)

    tokenizer = MyTokenizer.from_pretrained(opt.base_model)
    dataloader = create_dataloader(opt.train_data, tokenizer, opt)
    print('loading model')
    if opt.train_target == 'Entity':
        # model = MyClassifier(opt, len(label_id_to_name), len(tokenizer))
        model = MyClassifier(opt, 2, len(tokenizer))
    else:
        # model = MyClassifier(opt, len(polarity_id_to_name), len(tokenizer))
        model = MyClassifier(opt, 3, len(tokenizer))
    model.to(device)
    print('end loading')
    # entity_property_model_optimizer_setting
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, eps=opt.eps)
    lambda1 = lambda epoch: 0.65 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    epochs = opt.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(dataloader)

    for epoch in tqdm(range(epochs)):
        model.train()

        # entity_property train
        total_loss = 0

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()

            loss, _ = model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            total_loss += loss.item()
            # print('batch_loss: ', loss.item())

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(dataloader)
        print(f'{opt.train_target} Property_Epoch: {epoch+1}')
        print(f'Average train loss: {avg_train_loss}')

        if opt.train_target == 'Entity':
            model_saved_path = opt.entity_property_model_path + 'saved_model_epoch_' + str(epoch+1) + '.pt'
        else:
            model_saved_path = opt.polarity_property_model_path + 'saved_model_epoch_' + str(epoch+1) + '.pt'

        torch.save(model.state_dict(), model_saved_path)

        # if opt.do_eval:
        #     model.eval()

        #     pred_list = []
        #     label_list = []

        #     for batch in entity_dev_dataloader:
        #         batch = tuple(t.to(device) for t in batch)
        #         b_input_ids, b_input_mask, b_labels = batch

        #         with torch.no_grad():
        #             loss, logits = model(b_input_ids, b_input_mask, b_labels)

        #         predictions = torch.argmax(logits, dim=-1)
        #         pred_list.extend(predictions)
        #         label_list.extend(b_labels)

        #     evaluation(label_list, pred_list, len(label_id_to_name))
    print("training is done")

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
    # parser.add_argument( "--do_test", action="store_true")
    parser.add_argument( "--num_train_epochs", type=int, default=10)
    parser.add_argument( "--base_model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument( "--entity_property_model_path", type=str, default="./saved_models/entity_model/")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/polarity_model/")
    parser.add_argument( "--output_dir", type=str, default="./output/default_path/")
    parser.add_argument( "--do_demo", action="store_true")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--classifier_hidden_size", type=int, default=768)
    parser.add_argument( "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if opt.do_train:
    #     train_sentiment_analysis(args)
    # elif opt.do_demo:
    #     demo_sentiment_analysis(args)
    # elif opt.do_test:
    #     test_sentiment_analysis(args)
    test(opt, device)

