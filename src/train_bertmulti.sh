# # train entity7
# python train.py \
#   --train_target Entity \
#   --train_data ../data/nikluge-sa-2022-train.jsonl \
#   --dev_data ../data/nikluge-sa-2022-dev.jsonl \
#   --base_model bert-base-multilingual-uncased \
#   --num_labels 7 \
#   --do_eval True \
#   --learning_rate 1e-5 \
#   --eps 1e-8 \
#   --num_train_epochs 10 \
#   --entity_model_path ../saved_model/entity/ \
#   --polarity_model_path ../saved_model/polarity/ \
#   --batch_size 8 \
#   --max_len 256 
# # train entity 4
# python train.py \
#   --train_target Entity \
#   --train_data ../data/nikluge-sa-2022-train.jsonl \
#   --dev_data ../data/nikluge-sa-2022-dev.jsonl \
#   --base_model bert-base-multilingual-uncased \
#   --num_labels 4 \
#   --do_eval True \
#   --learning_rate 1e-5 \
#   --eps 1e-8 \
#   --num_train_epochs 10 \
#   --entity_model_path ../saved_model/entity/ \
#   --polarity_model_path ../saved_model/polarity/ \
#   --batch_size 8 \
#   --max_len 256 
# train polarity 
python train.py \
  --train_target Polarity \
  --train_data ../data/nikluge-sa-2022-train.jsonl \
  --dev_data ../data/nikluge-sa-2022-dev.jsonl \
  --base_model bert-base-multilingual-uncased \
  --num_labels 3 \
  --do_eval True \
  --learning_rate 1e-5 \
  --eps 1e-8 \
  --num_train_epochs 10 \
  --entity_model_path ../saved_model/entity/ \
  --polarity_model_path ../saved_model/polarity/ \
  --batch_size 8 \
  --max_len 256 