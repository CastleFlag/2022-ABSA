python test.py \
  --test_data ../data/nikluge-sa-2022-test.jsonl \
  --base_model bert-base-multilingual-uncased \
  --entity4_model_path ../saved_model/best_model/entity_4.pt \
  --entity7_model_path ../saved_model/best_model/entity_7.pt \
  --polarity_model_path ../saved_model/best_model/polarity.pt \
  --batch_size 8 \
  --max_len 256 \
  --mode test
  # --test_data ../data/nikluge-sa-2022-test.jsonl \