import argparse
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification, BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data import *
from model import MyClassifier
from evalutation import evaluation, evaluation_f1 
from tqdm import tqdm
import os
special_tokens_dict = {
'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}

def train(opt, device):
    if not os.path.exists(opt.entity_model_path):
        os.makedirs(opt.entity_model_path)
    if not os.path.exists(opt.polarity_model_path):
        os.makedirs(opt.polarity_model_path)

    # tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert', do_lower_case=False)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    dataloader = create_dataloader(opt.train_data, tokenizer, opt)
    # dev_dataloader = create_dataloader(opt.dev_data, tokenizer, opt)
    for b in dataloader:
        i,m,l = b
        print(i[:10])
        print(l)

    print('loading model')
    model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=25)
    # if opt.train_target == 'Entity':
    #     # model = MyClassifier(opt, len(label_id_to_name), len(tokenizer))
    #     model = MyClassifier(opt, 2, len(tokenizer))
    # else:
    #     # model = MyClassifier(opt, len(polarity_id_to_name), len(tokenizer))
    #     model = MyClassifier(opt, 3, len(tokenizer))
    model.to(device)
    print('end loading')
    # entity_property_model_optimizer_setting

    
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'gamma', 'beta']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #         'weight_decay_rate': 0.0}
    # ]
    # optimizer = AdamW(
    #     optimizer_grouped_parameters,
    #     lr=opt.learning_rate,
    #     eps=opt.eps
    # )
    optimizer = AdamW(model.parameters(), lr = opt.learning_rate, eps=opt.eps)

    epochs = opt.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(dataloader)

    # lambda1 = lambda epoch: 0.65 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    for step, batch in enumerate(dataloader):
        b_input_ids, b_input_mask, b_labels = batch
        print(b_input_ids)
        print(b_labels)


    for epoch in tqdm(range(epochs)):
        model.train()

        # entity_property train
        total_loss = 0

        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # for step, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            print(b_input_ids)
            model.zero_grad()

            loss, _ = model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            total_loss += loss.item()
            # print('batch_loss: ', loss.item())

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            # scheduler.step()

        avg_train_loss = total_loss / len(dataloader)
        print(f'{opt.train_target} Property_Epoch: {epoch+1}')
        print(f'Average train loss: {avg_train_loss}')

        if opt.train_target == 'Entity':
            model_saved_path = opt.entity_model_path + 'saved_model_epoch_' + str(epoch+1) + '.pt'
        else:
            model_saved_path = opt.polarity_model_path + 'saved_model_epoch_' + str(epoch+1) + '.pt'

        torch.save(model.state_dict(), model_saved_path)

        # if opt.do_eval:
        #     model.eval()

        #     pred_list = []
        #     label_list = []

        #     for batch in dev_dataloader:
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
    parser.add_argument( "--dev_data", type=str, default="../data/ninikluge-sa-2022-dev.jsonl", help="dev file")
    parser.add_argument( "--batch_size", type=int, default=8) 
    parser.add_argument( "--learning_rate", type=float, default=3e-5) 
    parser.add_argument( "--eps", type=float, default=1e-8)
    # parser.add_argument( "--do_train", action="store_true")
    parser.add_argument( "--do_eval", type=bool, default=True)
    # parser.add_argument( "--do_test", action="store_true")
    parser.add_argument( "--num_train_epochs", type=int, default=10)
    parser.add_argument( "--base_model", type=str, default="monologg/kobert")
    parser.add_argument( "--entity_model_path", type=str, default="./saved_models/entity_model/")
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
    train(opt, device)

