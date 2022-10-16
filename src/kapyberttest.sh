
python kapybert.py \
  --test_data ../data/sample.jsonl \
  --base_model bert-base-multilingual-cased\
  --do_test \
  --entity_property_model_path ../saved_model/category_extraction/saved_model_epoch_10.pt \
  --polarity_model_path ../saved_model/polarity_classification/saved_model_epoch_10.pt \
  --batch_size 1 \
  --max_len 16 