model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
dataset_name='yzhuang/xsum'
n_shot=1



TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" accelerate launch --num_processes 4 --mixed_precision bf16 --main_process_port 29502 ../run_few_shot_baseline.py \
    --model_name_or_path $model_name_or_path \
    --output_dir ./results/pretrained_summary_baseline/${shorthand} \
    --dataset_name $dataset_name \
    --low_cpu_mem_usage \
    --n_shot $n_shot \
    --max_seq_length 8096 \
    --max_gen_length 128 \
    --task_type summarization