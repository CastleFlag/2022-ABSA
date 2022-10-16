python sentiment_analysis.py \
  --test_data ../data/nikluge-sa-2022-test.jsonl \
  --base_model xlm-roberta-base \
  --do_test \
  --entity_property_model_path ../saved_model/category_extraction/saved_model_epoch_9.pt \
  --polarity_model_path ../saved_model/polarity_classification/saved_model_epoch_0.pt \
  --batch_size 8 \
  --max_len 256
  # --test_data ../data/nikluge-sa-2022-test.jsonl \