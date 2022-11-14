#!/usr/bin/env bash
# set -e 
# set -x 
DATA_DIR_ACE=data/ace/json
DATA_DIR_WIKI=data/wikievents
CKPT_NAME=acewiki_gen_ir
MODEL=gen

echo "******************************************************************************** gen_ir ****************************************************************************************************"

echo "******************** train ********************"

rm -rf checkpoints/${CKPT_NAME}
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME} \
    --dataset=ACEWIKI \
    --train_file=${DATA_DIR_ACE}/train.oneie.json \
    --val_file=${DATA_DIR_WIKI}/dev.jsonl \
    --test_file=${DATA_DIR_WIKI}/test.jsonl \
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
    --dataset=ACEWIKI \
    --eval_only \
    --train_file="${DATA_DIR_ACE}/train.oneie.json" \
    --val_file=${DATA_DIR_WIKI}/dev.jsonl \
    --test_file=${DATA_DIR_WIKI}/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --gpus=-1 \
    --ir

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref 

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref --head-only 


echo "******************** test epoch=5  ********************"
rm -rf checkpoints/${CKPT_NAME}-pred

python train.py --model=$MODEL --ckpt_name="${CKPT_NAME}-pred" \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=5.ckpt \
    --dataset=ACEWIKI \
    --eval_only \
    --train_file="${DATA_DIR_ACE}/train.oneie.json" \
    --val_file=${DATA_DIR_WIKI}/dev.jsonl \
    --test_file=${DATA_DIR_WIKI}/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --gpus=-1 \
    --ir

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref 

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref --head-only 


echo "******************************************************************************** gen ****************************************************************************************************"

DATA_DIR_ACE=data/ace/json
DATA_DIR_WIKI=data/wikievents
CKPT_NAME=acewiki_gen
MODEL=gen


echo "******************** train ********************"

rm -rf checkpoints/${CKPT_NAME}
python train.py --model=$MODEL --ckpt_name=${CKPT_NAME} \
    --dataset=ACEWIKI \
    --train_file=${DATA_DIR_ACE}/train.oneie.json \
    --val_file=${DATA_DIR_WIKI}/dev.jsonl \
    --test_file=${DATA_DIR_WIKI}/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --mark_trigger \
    --gpus=-1 \

printf "\n"


echo "******************** test epoch=4  ********************"
rm -rf checkpoints/${CKPT_NAME}-pred

python train.py --model=$MODEL --ckpt_name="${CKPT_NAME}-pred" \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=4.ckpt \
    --dataset=ACEWIKI \
    --eval_only \
    --train_file="${DATA_DIR_ACE}/train.oneie.json" \
    --val_file=${DATA_DIR_WIKI}/dev.jsonl \
    --test_file=${DATA_DIR_WIKI}/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --gpus=-1 \

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref 

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref --head-only 


echo "******************** test epoch=5  ********************"
rm -rf checkpoints/${CKPT_NAME}-pred

python train.py --model=$MODEL --ckpt_name="${CKPT_NAME}-pred" \
    --load_ckpt=checkpoints/$CKPT_NAME/epoch=5.ckpt \
    --dataset=ACEWIKI \
    --eval_only \
    --train_file="${DATA_DIR_ACE}/train.oneie.json" \
    --val_file=${DATA_DIR_WIKI}/dev.jsonl \
    --test_file=${DATA_DIR_WIKI}/test.jsonl \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=4 \
    --num_train_epochs=6 \
    --gpus=-1 \

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref 

python src/genie/scorer_acewiki.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl --dataset=ACEWIKI \
--test-file=${DATA_DIR_WIKI}/test.jsonl \
--coref --head-only 