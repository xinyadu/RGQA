#!/usr/bin/env bash
# set -e 
# set -x 
TRAIN_DATA_DIR=data/fewshot
DATA_DIR=data/ace/json
CKPT_NAME=fewshot_gen_ir_yn
MODEL=gen

samplings=('random' 'ppl' 'c_context' 'c_context_trg')
# samplings=('ppl')

# shots=(50 100 200 300 400 500 1000)
# shots=(100 200 300 400 500 1000)
# shots=(400 600)
shots=(100 200 300 400 500 600 700 800 900 1000)
# shots=(200 400 600 800 1000)
# shots=(100 300 500 700 900)
# shots=(150 200 250 300 350 400 450 500)


for shot in "${shots[@]}"
do
    for sampling in "${samplings[@]}"
    do
        CKPT_NAME_final="${CKPT_NAME}_${sampling}_${shot}"

        echo "------------------------------------------------------------ $sampling, shot: $shot ------------------------------------------------------------"

        # train
        rm -rf checkpoints/${CKPT_NAME_final}

        python train.py --model=$MODEL --ckpt_name=$CKPT_NAME_final \
        --fewshot \
        --sampling $sampling \
        --shot $shot \
        --dataset=ACE \
        --train_file="${TRAIN_DATA_DIR}/${sampling}_${shot}.json" \
        --val_file=${DATA_DIR}/dev.oneie.json \
        --test_file=${DATA_DIR}/test.oneie.json \
        --train_batch_size=4 \
        --eval_batch_size=4 \
        --learning_rate=2e-5 \
        --accumulate_grad_batches=4 \
        --num_train_epochs=8 \
        --mark_trigger \
        --gpus=-1 \
        --ir \
        --add_yn

        printf "\n"

        # # test
        # echo "******************** test epoch=5 ********************"
        # rm -rf checkpoints/${CKPT_NAME_final}-pred

        # python train.py --model=$MODEL --ckpt_name="${CKPT_NAME_final}-pred" \
        # --fewshot \
        # --sampling $sampling \
        # --shot $shot \
        # --load_ckpt=checkpoints/$CKPT_NAME_final/epoch=5.ckpt \
        # --dataset=ACE \
        # --eval_only \
        # --train_file="${TRAIN_DATA_DIR}/${sampling}_${shot}.json" \
        # --val_file=${DATA_DIR}/dev.oneie.json \
        # --test_file=${DATA_DIR}/test.oneie.json \
        # --train_batch_size=4 \
        # --eval_batch_size=4 \
        # --learning_rate=3e-5 \
        # --accumulate_grad_batches=4 \
        # --num_train_epochs=6 \
        # --gpus=-1 \
        # --ir \
        # --add_yn


        # python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME_final-pred/predictions.jsonl --dataset=ACE \
        # --test-file=${DATA_DIR}/test.oneie.json \
        # --coref 

        # printf "\n"

        # # python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME_final-pred/predictions.jsonl --dataset=ACE \
        # # --test-file=${DATA_DIR}/test.oneie.json \
        # # --coref --head-only 

        # echo "******************** test epoch=7 ********************"
        # rm -rf checkpoints/${CKPT_NAME_final}-pred

        # python train.py --model=$MODEL --ckpt_name="${CKPT_NAME_final}-pred" \
        # --fewshot \
        # --sampling $sampling \
        # --shot $shot \
        # --load_ckpt=checkpoints/$CKPT_NAME_final/epoch=7.ckpt \
        # --dataset=ACE \
        # --eval_only \
        # --train_file="${TRAIN_DATA_DIR}/${sampling}_${shot}.json" \
        # --val_file=${DATA_DIR}/dev.oneie.json \
        # --test_file=${DATA_DIR}/test.oneie.json \
        # --train_batch_size=4 \
        # --eval_batch_size=4 \
        # --learning_rate=3e-5 \
        # --accumulate_grad_batches=4 \
        # --num_train_epochs=6 \
        # --gpus=-1 \
        # --ir \
        # --add_yn

        # python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME_final-pred/predictions.jsonl --dataset=ACE \
        # --test-file=${DATA_DIR}/test.oneie.json \
        # --coref 

        # # python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME_final-pred/predictions.jsonl --dataset=ACE \
        # # --test-file=${DATA_DIR}/test.oneie.json \
        # # --coref --head-only 

        echo "******************** test epoch=last ********************"
        rm -rf checkpoints/${CKPT_NAME_final}-pred

        python train.py --model=$MODEL --ckpt_name="${CKPT_NAME_final}-pred" \
        --fewshot \
        --sampling $sampling \
        --shot $shot \
        --load_ckpt=checkpoints/$CKPT_NAME_final/last.ckpt \
        --dataset=ACE \
        --eval_only \
        --train_file="${TRAIN_DATA_DIR}/${sampling}_${shot}.json" \
        --val_file=${DATA_DIR}/dev.oneie.json \
        --test_file=${DATA_DIR}/test.oneie.json \
        --train_batch_size=4 \
        --eval_batch_size=4 \
        --learning_rate=3e-5 \
        --accumulate_grad_batches=4 \
        --num_train_epochs=6 \
        --gpus=-1 \
        --ir \
        --add_yn

        python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME_final-pred/predictions.jsonl --dataset=ACE \
        --test-file=${DATA_DIR}/test.oneie.json \
        --coref 

        printf "\n"

        # python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME_final-pred/predictions.jsonl --dataset=ACE \
        # --test-file=${DATA_DIR}/test.oneie.json \
        # --coref --head-only 
    done
done