model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct"
embed_model="nvidia/NV-Embed-v1"
dataset_name='yzhuang/imdb'
projector_type="linear"
n_shot=1

#get the name after "/"
model_name=$(echo $model_name_or_path | cut -d'/' -f 2)
embed_model_name=$(echo $embed_model | cut -d'/' -f 2)
projector_model_id=yzhuang/linear_icl_projector_sentiment_analysis_${embed_model_name}_${model_name}


TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" accelerate launch --num_processes 2 --mixed_precision bf16 --main_process_port 29502 ../run_vector_icl_classification.py \
    --model_name_or_path $model_name_or_path \
    --embed_model_name_or_path $embed_model \
    --output_dir ./results/finetuned_${projector_type}_sentiment_analysis/${model_name}_${embed_model_name} \
    --dataset_name $dataset_name \
    --low_cpu_mem_usage \
    --n_shot $n_shot \
    --max_seq_length 2048 \
    --projector_model_id $projector_model_id 