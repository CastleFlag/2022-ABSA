MODELS=(xlm-roberta-base monologg/koelectra-base-v3-discriminator xlm-roberta-large)
# HIDDEN_SIZE=(768 768 1024)
HIDDEN_SIZE=(768)

for ITER in 0 1 2;
do
./train_M4m7.sh ${MODELS[$ITER]}
done