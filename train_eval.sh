#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
for i in {1..1}
do
    DATA=./data/bench/$i
    echo $DATA
    
    mkdir -p ${DATA}/processed
    
    pushd ${DATA}/processed &> /dev/null
    rm -rf test_*.table
    rm -rf train_*.pair
    popd &> /dev/null
    
    pushd output/${i} &> /dev/null
    rm -rf checkpoint-epoch*
    popd &> /dev/null
    
    python trainer.py --data_dir ${DATA} \
                      --tabert_path ./model/tabert_base_k3/model.bin \
                      --config_file ./model/tabert_base_k3/tb_config.json \
                      --gpus 1 \
                      --precision 16 \
                      --max_epochs 10 \
                      --weight_decay 0.0 \
                      --min_row 20 \
                      --lr 5e-5 \
                      --do_train \
                      --stocahstic_weight_avg \
                      --train_batch_size 2 \
                      --valid_batch_size 1 \
                      --output_dir output/${i} \
                      --accumulate_grad_batches 8 \
                      --seed 20200401 
            
    CKPT=`ls output/${i} | sort -k3 -t'=' | head -1`
    
    echo ${CKPT}
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 100 \
                              --ckpt_file ./output/${i}/${CKPT} \
                              --hnsw_index | tee ./result/${i}_hnsw.result
    
    python dense_retrieval.py --data_dir ${DATA} \
                              --gpus 1 \
			      --topk 100 \
                              --ckpt_file ./output/${i}/${CKPT} | tee ./result/${i}.result
done

python predict.py
