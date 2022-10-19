MODELS=(bert-base-multilingual-uncased koelectra-base-v3-discriminator xlm-roberta-large)
ENTITY4_MODEL_PATH=()
ENTITY7_MODEL_PATH=()
POLARITY_MODEL_PATH=()

for MODEL in ${MODELS[@]}
do
ENTITY4_BEST_PATH="../saved_model/best_model/${MODEL}_4.pt"
ENTITY7_BEST_PATH="../saved_model/best_model/${MODEL}_7.pt"
POLARITY_BEST_PATH="../saved_model/best_model/${MODEL}_P.pt"
./test_M4m7.sh $MODEL $ENTITY4_BEST_PATH $ENTITY7_BEST_PATH $POLARITY_BEST_PATH
done