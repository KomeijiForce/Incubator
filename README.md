# Incubator

This repo is the official implementation for [Incubating Text Classifiers Following User Instruction with Nothing but LLM](https://arxiv.org/abs/2404.10877). We allow users to get a personalized classifier with only the instruction as input. The incubation is based on a [llama-2-7b](https://huggingface.co/KomeijiForce/Incubator-llama-2-7b) fine-tuned on Huggingface Meta Data and Self-Diversification.

![Incubator](https://github.com/KomeijiForce/Incubator/blob/main/overview.jpg)

## Incubating Classifiers

You can use the script ```incubate.sh``` to incubate your own classifiers.

```bash
python incubate.py --n_epoch 16 \
    --batch_size 4 \
    --device 1 \
    --n_sample 16 \
    --max_new_tokens 64 \
    --instruction "Build a classifier that can categorize text messages by 'about food' and 'about movie'." \
    --incubator "KomeijiForce/Incubator-llama-2-7b" \
    --classifier "roberta-base" \
    --save_path "roberta-base-incubated"
```
