base_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
embed_model="nvidia/NV-Embed-v1"

model_name=$(echo $base_model | cut -d'/' -f 2)
embed_model_name=$(echo $embed_model | cut -d'/' -f 2)


TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" python ../train_taskfinetune.py \
    --base_model_name_or_path $base_model \
    --embed_model_name_or_path $embed_model \
    --dataset_name "yzhuang/imdb yzhuang/sst2 yzhuang/rotten_tomatoes yzhuang/financial_phrasebank yzhuang/emotion"\
    --output_dir ../ckpts \
    --batch_size 128 \
    --max_length 512 \
    --lr 1e-3 \
    --projector_type linear \
    --num_train_epochs 1 \
    --task_type sentiment_analysis \
    --hub_model_id yzhuang/linear_projector_distclm2_wikitext_${embed_model_name}_${model_name}