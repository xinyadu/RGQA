#!/usr/bin/env bash
set -e 
set -x 
DATA_DIR=data/ace
CKPT_NAME=gen-ACE
MODEL=constrained-gen

rm -rf checkpoints/${CKPT_NAME}-pred
python train.py --model=$MODEL --ckpt_name=$CKPT_NAME-pred \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=0.ckpt \
    --dataset=ACE \
    --eval_only \
    --train_file=${DATA_DIR}/train.json \
    --val_file=${DATA_DIR}/dev.json \
    --test_file=${DATA_DIR}/test.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=3 \
    --ir


python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/json/toy.test.oneie.json \
--coref 

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/json/toy.test.oneie.json \
--coref --head-only 