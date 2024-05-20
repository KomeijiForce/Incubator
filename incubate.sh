python incubate.py --n_epoch 16 \
    --batch_size 4 \
    --device 1 \
    --n_sample 16 \
    --max_new_tokens 64 \
    --instruction "Build a classifier that can categorize text messages by 'about food' and 'about movie'." \
    --incubator "KomeijiForce/Incubator-llama-2-7b" \
    --classifier "roberta-base" \
    --save_path "roberta-base-incubated"
