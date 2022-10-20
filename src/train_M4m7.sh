TARGETS=(Entity Entity Polarity)
LABLES=(7 4 3)
TRAIN_DATA=../data/nikluge-sa-2022-train.jsonl 
DEV_DATA=../data/nikluge-sa-2022-dev.jsonl
BASE_MODEL=$1
DO_EVAL=True
LEARNING_RATE=1e-5
EPS=1e-8
EPOCHS=10
ENTITY_PATH=../saved_model/entity/
PORALITY_PATH=../saved_model/polarity/
OUTPUT_PATH=../output/
BATCH_SIZE=16
MAX_LEN=256
HIDDEN_SIZE=$2
DROPOUT=0.1
for ITER in 0 1 2
do
python3 train.py \
  --train_target ${TARGETS[$ITER]} \
  --train_data $TRAIN_DATA \
  --dev_data $DEV_DATA \
  --base_model $BASE_MODEL \
  --num_labels ${LABLES[$ITER]} \
  --do_eval $DO_EVAL \
  --learning_rate $LEARNING_RATE \
  --eps $EPS \
  --num_train_epochs $EPOCHS \
  --entity_model_path $ENTITY_PATH \
  --polarity_model_path $PORALITY_PATH \
  --output_dir $OUTPUT_PATH \
  --batch_size $BATCH_SIZE \
  --max_len $MAX_LEN \
  --classifier_hidden_size $HIDDEN_SIZE \
  --classifier_dropout_prob $DROPOUT 
done