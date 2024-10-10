base_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
embed_model="nvidia/NV-Embed-v1"

TOKENIZERS_PARALLELISM=false PYTHONPATH="../":"$PYTHONPATH" python ../train_textpretrain.py \
    --base_model_name_or_path $base_model \
    --embed_model_name_or_path $embed_model \
    --dataset_name wikitext \
    --output_dir ../ckpts \
    --batch_size 128 \
    --max_length 512 \
    --lr 1e-3 \
    --projector_type linear 