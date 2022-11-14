#!/usr/bin/env bash
# set -e 
# set -x 
DATA_DIR=data_toy/ace/json
CKPT_NAME=gen_ir_yn
MODEL=gen


echo "******************** test epoch=5  ********************"
rm -rf checkpoints/${CKPT_NAME}-pred

python train.py --model=$MODEL --ckpt_name="${CKPT_NAME}-pred" \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=5.ckpt \
    --dataset=ACE \
    --eval_only \
    --train_file="${DATA_DIR}/toy.train.oneie.json" \
    --val_file=${DATA_DIR}/toy.dev.oneie.json \
    --test_file=${DATA_DIR}/toy.test.oneie.json \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --ir \
    --add_yn


echo "******************** postprocess and eval (final output under output-file)  ********************"

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/toy.test.oneie.json \
--output-file=${DATA_DIR}/predict.toy.test.oneie.json \

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACE \
--test-file=${DATA_DIR}/toy.test.oneie.json \
--output-file=${DATA_DIR}/predict.toy.test.oneie.json \
--head-only 