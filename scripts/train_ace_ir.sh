#!/usr/bin/env bash
set -e 
set -x 
DATA_DIR=data/ace/json
CKPT_NAME=gen_ir
MODEL=gen

echo "******************** train ********************"

rm -rf checkpoints/${CKPT_NAME}
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME} \
    --dataset=ACE \
    --train_file=${DATA_DIR}/train.oneie.json \
    --val_file=${DATA_DIR}/dev.oneie.json \
    --test_file=${DATA_DIR}/test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --mark_trigger \
    --gpus=-1 \
    --ir

printf "\n"


echo "******************** test epoch=4  ********************"
rm -rf checkpoints/${CKPT_NAME}-pred

python train.py --model=$MODEL --ckpt_name="${CKPT_NAME}-pred" \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=4.ckpt \
    --dataset=ACE \
    --eval_only \
    --train_file="${DATA_DIR}/train.oneie.json" \
    --val_file=${DATA_DIR}/dev.oneie.json \
    --test_file=${DATA_DIR}/test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --gpus=-1 \
    --ir

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/test.oneie.json \
--coref 

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/test.oneie.json \
--coref --head-only 


echo "******************** test epoch=5  ********************"
rm -rf checkpoints/${CKPT_NAME}-pred

python train.py --model=$MODEL --ckpt_name="${CKPT_NAME}-pred" \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=5.ckpt \
    --dataset=ACE \
    --eval_only \
    --train_file="${DATA_DIR}/train.oneie.json" \
    --val_file=${DATA_DIR}/dev.oneie.json \
    --test_file=${DATA_DIR}/test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --gpus=-1 \
    --ir

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/test.oneie.json \
--coref 

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/test.oneie.json \
--coref --head-only 