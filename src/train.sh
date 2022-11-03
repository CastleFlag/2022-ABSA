# MODELS=(bert-base-multilingual-cased)
MODELS=(xlm-roberta-base)
# MODELS=(monologg/koelectra-base-v3-discriminator xlm-roberta-large)
# HIDDEN_SIZE=(768 768 1024)
# MODELS=(digit82/kobart-summarization)
# MODELS=(hyunwoongko/kobart)
HIDDEN_SIZE=(768)

for ITER in 0;
do
./train_M4m7.sh ${MODELS[$ITER]} ${HIDDEN_SIZE[$ITER]}
done