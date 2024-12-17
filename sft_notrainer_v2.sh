

cd /data/huangyinkui/code/peft

export CUDA_VISIBLE_DEVICES=0,1

MODEL_NAME=Qwen2.5-7B-Instruct
TASK_NAME=v4
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${MODEL_NAME} model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file config/stage3_no_offloading_accelerate.conf \
    sft_notrainer.py \
    --model_name_or_path /data/huangyinkui/models/${MODEL_NAME} \
    --tokenizer_name /data/huangyinkui/models/${MODEL_NAME} \
    --train_file /tmp/huangyinkui/sft_data/title_generation_train_${TASK_NAME}.json \
    --validation_file /tmp/huangyinkui/sft_data/title_generation_valid_${TASK_NAME}.json \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /data/huangyinkui/output/${TASK_NAME}_${MODEL_NAME}/ \
    --with_tracking \
    --report_to tensorboard \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --trainable "q_proj,v_proj,k_proj,o_proj,gate,w1,w2,w3" \
    --modules_to_save "embed_tokens,lm_head" \
    --max_train_steps 50 \
    # --logging_steps 1 \
    # --warmup_ratio 0.03 \
    # --max_seq_length 2048 \
    # --use_flash_attn debug时数据长度并不长，暂时不使用，但后续必须添加(RAG)
    # --use_special_tokens 需要扩充词表

