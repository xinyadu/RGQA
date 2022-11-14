#!/usr/bin/env bash
# set -e 
# set -x 
DATA_DIR=data_toy/ace/json
CKPT_NAME=gen_ir_yn
MODEL=gen

rm -rf checkpoints/${CKPT_NAME}
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME} \
    --dataset=ACE \
    --train_file=${DATA_DIR}/toy.train.oneie.json \
    --val_file=${DATA_DIR}/toy.dev.oneie.json \
    --test_file=${DATA_DIR}/toy.test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=2 \
    --mark_trigger \
    --ir
