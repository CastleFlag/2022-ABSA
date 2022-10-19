# MODELS=(bert-base-multilingual-uncased koelectra-base-v3-discriminator xlm-roberta-large)
# HIDDEN_SIZE=(768 768 1024)
MODELS=(bert-base-multilingual-uncased)
HIDDEN_SIZE=(768)

# for ITER in 0 1 2;
for ITER in 0;
do
./train_M4m7.sh ${MODELS[$ITER]} ${HIDDEN_SIZE[$ITER]}
done