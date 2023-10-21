REPO=$PWD
TASK=${1:-conll}
GPU=${2:-0}
SEED=${3:-1}
DATA_DIR=${4:-"$REPO/download/"}
OUT_DIR=${5:-"$REPO/outputs/"}

export CUDA_VISIBLE_DEVICES=$GPU
MODEL="roberta-base"
MODEL_TYPE="roberta"
GRAD_ACC=1
MLP_DIM=256
MLP_DROPOUT=0.2
SAMPLE_RATE=0.35

SP="--select_positive"
# SP=""
SN="--select_negative"
# SN=""

LR=1e-5
BEGIN_EPOCH=1
if [ ${TASK:0:5} == "conll" ]; then
    WARMUP_STEPS=800
    SAVE_STEPS=200
    BATCH_SIZE=32
    NUM_EPOCHS=40
elif [ ${TASK:0:7} == "webpage" ]; then
    WARMUP_STEPS=200
    SAVE_STEPS=100
    BATCH_SIZE=16
    NUM_EPOCHS=100
elif [ ${TASK:0:7} == "mitrest" ]; then
    WARMUP_STEPS=400
    SAVE_STEPS=100
    BATCH_SIZE=32
    NUM_EPOCHS=50
elif [ ${TASK:0:8} == "mitmovie" ]; then
    WARMUP_STEPS=400
    SAVE_STEPS=100
    BATCH_SIZE=32
    NUM_EPOCHS=50
elif [ ${TASK:0:6} == "bc5cdr" ]; then
    WARMUP_STEPS=200
    SAVE_STEPS=100
    BATCH_SIZE=32
    LR=2e-5
    NUM_EPOCHS=50
fi

DATA_DIR=$DATA_DIR/$TASK
OUTPUT_DIR="$OUT_DIR/${TASK}/${TASK}-UES-NPE-${SEED}"
mkdir -p $OUTPUT_DIR
python $REPO/main.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL \
    --output_dir $OUTPUT_DIR \
    --log_file $OUTPUT_DIR/train.log \
    --labels $DATA_DIR/labels.txt \
    --eval_test_set \
    --do_train --do_eval \
    --batch_size $BATCH_SIZE \
    --eval_batch_size 32 \
    --gradient_accumulation_steps $GRAD_ACC \
    --mlp_hidden_size $MLP_DIM \
    --mlp_dropout_rate $MLP_DROPOUT \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --overwrite_output_dir \
    --seed $SEED \
    --select_begin_epoch $BEGIN_EPOCH \
    --neg_sample_rate $SAMPLE_RATE \
    $SP $SN \
