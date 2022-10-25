import argparse
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification, BertTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data import *
from model import MyClassifier
from evalutation import evaluation, evaluation_f1 
from tqdm import tqdm
from shutil import copyfile
import os

special_tokens_dict = {
'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}

def train(opt, device):
    model_path = opt.entity_model_path + opt.base_model.split('/')[0] + '/' 
    best_model_path = '../saved_model/best_model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    print(opt.base_model)
    print(opt.train_target + ' ' + str(opt.num_labels))
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    dataloader = create_dataloader(opt.train_data, tokenizer, opt)
    dev_dataloader = create_dataloader(opt.dev_data, tokenizer, opt)
    # entity_dev_dataloader, polarity_dev_dataloader = create_dataloader(opt.dev_data, tokenizer, opt, True)

    print('loading model')
    model = BertForSequenceClassification.from_pretrained(opt.base_model, num_labels=opt.num_labels)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print('end loading')

    optimizer = AdamW(model.parameters(), lr=opt.learning_rate, eps=opt.eps)
    # optimizer = AdamW(
    #     optimizer_grouped_parameters,
    #     lr=opt.learning_rate,
    #     eps=opt.eps
    # )

    epochs = opt.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(dataloader)

    # lambda1 = lambda epoch: 0.65 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps
    # )
    min_f1 = 99
    optim_model_path = ""
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = get_inputs_dict(batch, tokenizer, device)
            model.zero_grad()
            outputs = model(**inputs)
            
            loss=outputs[0]
            loss.backward()

            total_loss += loss.item()
            # print('batch_loss: ', loss.item())

            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            # scheduler.step()

        avg_train_loss = total_loss / len(dataloader)
        print(f'{opt.train_target} Property_Epoch: {epoch+1}')
        print(f'Average train loss: {avg_train_loss}')

        model_saved_path = model_path + 'saved_model_epoch_' + str(epoch+1) + '.pt'
        torch.save(model.state_dict(), model_saved_path)

        if opt.do_eval:
            model.eval()
            pred_list = []
            label_list = []

            for batch in dev_dataloader:
                inputs = get_inputs_dict(batch, tokenizer, device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions.cpu())
                label_list.extend(inputs['labels'].cpu())
            f1score = evaluation(label_list, pred_list, opt.num_labels)
    # # save best model 
    # if opt.num_labels==3:
    #     copyfile(optim_model_path, best_model_path + opt.base_model + '_P.pt')
    # else:
    #     copyfile(optim_model_path, best_model_path + opt.base_model + '_' + str(opt.num_labels) + '.pt')
    print("training is done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--train_target", type=str, default="Entity", help="train entity or polarity")
    parser.add_argument( "--train_data", type=str, default="../data/nikluge-sa-2022-train.jsonl", help="train file")
    parser.add_argument( "--test_data", type=str, default="../data/nikluge-sa-2022-test.jsonl", help="test file")
    parser.add_argument( "--dev_data", type=str, default="../data/nikluge-sa-2022-dev.jsonl", help="dev file")
    parser.add_argument( "--batch_size", type=int, default=8) 
    parser.add_argument( "--learning_rate", type=float, default=3e-5) 
    parser.add_argument( "--eps", type=float, default=1e-8)
    parser.add_argument( "--do_eval", type=bool, default=True)
    parser.add_argument( "--num_train_epochs", type=int, default=10)
    parser.add_argument( "--base_model", type=str, default="skt/kobert-base-v1")
    parser.add_argument( "--num_labels", type=int, default=25)
    parser.add_argument( "--entity_model_path", type=str, default="./saved_models/entity_model/")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/polarity_model/")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--classifier_hidden_size", type=int, default=768)
    parser.add_argument( "--classifier_dropout_prob", type=float, default=0.1, help="dropout in classifier")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(opt, device)

