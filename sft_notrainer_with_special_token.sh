export CUDA_VISIBLE_DEVICES=1,2

MODEL_NAME=Llama-2-7b-Chat-hf
TRAIN_FP=train.json
VALIDATION_FP=dev.json
TASK_NAME=debug 

NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${MODEL_NAME} model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file /public/HYK/code/FAWKES/peft/config/stage3_no_offloading_accelerate.conf \
    run_clm_no_trainer.py \
    --model_name_or_path /public/HYK/models/${MODEL_NAME} \
    --tokenizer_name /public/HYK/models/${MODEL_NAME} \
    --use_slow_tokenizer \
    --train_file ${TRAIN_FP} \
    --validation_file ${VALIDATION_FP} \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir /public/HYK/checkpoints/${TASK_NAME}_${MODEL_NAME}/ \
    --with_tracking \
    --report_to tensorboard \
    --use_lora
    # --logging_steps 1 \
    # --warmup_ratio 0.03 \
    # --max_seq_length 2048 \
    # --use_flash_attn debug时数据长度并不长，暂时不使用，但后续必须添加(RAG)
    # --use_special_tokens 需要扩充词表
