# Download official openbmb/RLAIF-V-Dataset
huggingface-cli download openbmb/RLAIF-V-Dataset --repo-type dataset --local-dir ./base_datasets/LLaVA-RLAIF-Data/ --local-dir-use-symlinks False
# Extract Partial Dataset and Split them with fixed proportion
python base_operations/make_online_generation_dataset.py