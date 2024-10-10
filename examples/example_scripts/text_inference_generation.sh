model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
embed_model="nvidia/NV-Embed-v1"
dataset_name='yzhuang/xsum'
projector_type="linear"
n_shot=1

#get the name after "/"
model_name=$(echo $model_name_or_path | cut -d'/' -f 2)
embed_model_name=$(echo $embed_model | cut -d'/' -f 2)
projector_model_id=yzhuang/linear_icl_projector_summarization_${embed_model_name}_${model_name}


TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" accelerate launch --num_processes 2 --mixed_precision bf16 --main_process_port 29502 ../run_vector_icl_generation.py \
    --model_name_or_path $model_name_or_path \
    --embed_model_name_or_path $embed_model \
    --output_dir ./results/finetuned_${projector_type}_summary/${model_name}_${embed_model_name} \
    --dataset_name $dataset_name \
    --low_cpu_mem_usage \
    --n_shot $n_shot \
    --max_seq_length 8096 \
    --max_gen_length 128 \
    --projector_model_id $projector_model_id \
    --task_type summarization 