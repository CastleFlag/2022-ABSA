# train entity
python train.py \
  --train_target Entity \
  --train_data ../data/nikluge-sa-2022-train.jsonl \
  --dev_data ../data/nikluge-sa-2022-dev.jsonl \
  --base_model bert-base-multilingual-cased \
  --num_labels 25
  --do_eval True\
  --learning_rate 1e-5 \
  --eps 1e-8 \
  --num_train_epochs 10 \
  --entity_property_model_path ../saved_model/category_extraction/ \
  --polarity_model_path ../saved_model/polarity_classification/ \
  --batch_size 8 \
  --max_len 256 

# train polarity 
python train.py \
  --train_target Polarity \
  --train_data ../data/nikluge-sa-2022-train.jsonl \
  --dev_data ../data/nikluge-sa-2022-dev.jsonl \
  --base_model bert-base-multilingual-cased \
  --num_labels 3 
  --do_eval True\
  --learning_rate 3e-6 \
  --eps 1e-8 \
  --num_train_epochs 10 \
  --entity_property_model_path ../saved_model/category_extraction/ \
  --polarity_model_path ../saved_model/polarity_classification/ \
  --batch_size 8 \
  --max_len 256 
  # --base_model  monologg/kobert \