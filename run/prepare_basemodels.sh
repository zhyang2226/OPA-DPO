# Download official vision-tower checkpoint
huggingface-cli download openai/clip-vit-large-patch14-336 --repo-type model --local-dir ./base_models/vision_tower-clip336/ --local-dir-use-symlinks False

# Download official LLaVA-Instruct-150K
huggingface-cli download liuhaotian/llava-v1.5-7b --repo-type model --local-dir ./base_models/llava-v1.5-7b/ --local-dir-use-symlinks False
huggingface-cli download liuhaotian/llava-v1.5-13b --repo-type model --local-dir ./base_models/llava-v1.5-13b/ --local-dir-use-symlinks False
# Modify the model config to use the vision-tower checkpoint
python base_operations/modify_base_model_config.py
