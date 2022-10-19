python test.py \
--test_data ../data/nikluge-sa-2022-test.jsonl \
--base_model $1 \
--entity4_model_path $2 \
--entity7_model_path $3 \
--polarity_model_path $4 \
--batch_size 16 \
--max_len 256 