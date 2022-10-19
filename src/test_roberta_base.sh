python test.py \
--test_data ../data/nikluge-sa-2022-test.jsonl \
--base_model xlm-roberta-base \
--entity4_model_path ../saved_model/best_model/4.pt \
--entity7_model_path ../saved_model/best_model/7.pt \
--polarity_model_path ../saved_model/best_model/pola.pt \
--batch_size 16 \
--max_len 256 \
--mode test