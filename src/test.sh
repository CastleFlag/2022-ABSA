MODELS=(bert-base-multilingual-uncased monologg/koelectra-base-v3-discriminator xlm-roberta-large)
HIDDEN_SIZE=(768 768 1024)
ENTITY4_MODEL_PATH=()
ENTITY7_MODEL_PATH=()
POLARITY_MODEL_PATH=()

for ITER in 0 1 2
do
ENTITY4_BEST_PATH="../saved_model/best_model/${MODELS[$ITER]}_4.pt"
ENTITY7_BEST_PATH="../saved_model/best_model/${MODELS[$ITER]}_7.pt"
POLARITY_BEST_PATH="../saved_model/best_model/${MODELS[$ITER]}_P.pt"
./test_M4m7.sh ${MODELS[$ITER]} $ENTITY4_BEST_PATH $ENTITY7_BEST_PATH $POLARITY_BEST_PATH ${HIDDEN_SIZE[$ITER]}
done