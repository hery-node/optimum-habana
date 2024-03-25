python gaudi_spawn.py \
    --world_size 8 --use_deepspeed language-modeling/run_alpaca.py \
    --model_name_or_path "huggyllama/llama-7b" \
    --data_path alpaca.json \
    --bf16 True \
    --output_dir ./alpaca_checkpoint_gaudi2/ \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --do_train \
    --use_habana \
    --use_lazy_mode \
    --use_hpu_graphs_for_inference \
    --throughput_warmup_steps 10 \
    --overwrite_output_dir \
    --deepspeed "ds.json" \
    --gaudi_config_name Habana/gpt2 \
    --logging_step 1
