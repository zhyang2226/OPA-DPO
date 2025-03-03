# OPA-DPO (On-Policy Alignment Direct Preference Optimization)

Project Page: [https://opa-dpo.github.io/](https://opa-dpo.github.io/)

Authors: [Zhihe Yang](https://zhyang2226.github.io)$^{1,3}$, [Xufang Luo](https://www.microsoft.com/en-us/research/people/xufluo/)$^{2*}$, [Dongqi Han](https://www.microsoft.com/en-us/research/people/dongqihan/)$^2$, [Yunjian Xu](https://www4.mae.cuhk.edu.hk/peoples/xu-yunjian/)$^{1,3*}$, [Dongsheng Li](http://recmind.cn/)$^2$

($^*$ for corresponding authors)

Affiliations:

1. The Chinese University of Hong Kong, Hong Kong SAR, China
2. Microsoft Research Asia, Shanghai, China
3. The Chinese University of Hong Kong, Shenzhen Research Institute (SZRI), Guandong, China

## GOOD NEWS!!!

Our paper has been accepted by **CVPR 2025**! See you in Nashville this summer!

Find our full paper through arXiv Link: https://arxiv.org/abs/2501.09695.
We will update the experimental content and codebase in the next few weeks (including the experimental results on the basis of [LLaVA-Onevision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/)).

## Introduction

Hallucination remains a major challenge for Large Vision-Language Models (LVLMs). Direct Preference Optimization (DPO) has gained increasing attention as a simple solution to hallucination issues. Nonetheless, different data construction methods in existing works bring notable performance variations.

We identify a crucial factor: outcomes are largely contingent on whether the constructed data aligns on-policy w.r.t the initial (reference) policy of DPO. Due to the implicit KL-divergence constraint, off-policy data cannot be effectively learned (Fig.a left).

We propose On-Policy Alignment (OPA)-DPO framework (Fig.a right and Fig.d), which uniquely leverages expert feedback to correct hallucinated responses and aligns both the original and expert-revised responses in an on-policy manner.

Compared with DPO without OPA operations, OPA-DPO significantly enhances performance (Fig.c). It achieves SOTA performance with only 4.8k training data, while most DPO-based algorithms require over 10k data (Fig.b).

![intro_fig](assests/README/intro_fig.png)

## Example

Below is a qualitative example of OPA-DPO. OPA-DPO successfully resolves most of the hallucinations (marked in red), though at the expense of omitting some details present in the original description.

![example](assests/README/example.png)

## Environment Set up

Our codebase extensively utilizes [LLaVA](https://github.com/haotian-liu/LLaVA). However, due to copyright concerns, we are unable to include the LLaVA source code directly within our OPA-DPO codebase. To begin, please execute the following command to copy and adapt the LLaVA codebase into OPA-DPO directory:

```
git clone https://github.com/microsoft/OPA-DPO
cd OPA-DPO/llava_setup
git clone https://github.com/haotian-liu/LLaVA
cd LLaVA
git reset --hard 817a4af
cd ..
patch -p1 < llava_modifications.patch
cd ..
mv ./llava_setup/LLaVA/llava ./
```

To set up the environment for OPA-DPO, please execute the following command:

```
conda env create -f environment.yaml
conda activate OPA_DPO
pip install flash-attn==2.5.3
```

Please be aware that the existing version of OPA-DPO is exclusively compatible with the Linux environment. The default configuration is designed for a single node with 4x80GB A100.

## Training OPA-DPO

### Step 0: BASE Models and Datasets Preparation

Before initiating the training process, it is required to download the base models (LLaVA-1.7-7B/13B) and vision towers (CLIP-VIT) to the local directory `./base_models/` using the following command:

```
bash run/prepare_basemodels.sh
```

Following this, it's crucial to construct the training dataset. If you already have the required datasets for OPA-DPO training at hand, skip ahead to Step 3. However, if you're starting without these datasets, they can be compiled using the procedure outlined below.

By default, we utilize partial prompts and images from the [RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset). To download this dataset to local directory `./base_datasets/LLaVA-RLAIF-Data` and extract a subset of samples, simply execute the following command:

```
bash run/prepare_datasets.sh
```

### Step 1-2: Rollout base models & use GPT-4V for fine-grained hallucination correction

As shown in Fig.d, the first step is to collect responses from the original policy. The second step involves using GPT-4V to correct any hallucinations in the generated responses. These steps can be executed simultaneously using the following command:

```
bash run/online_generate.sh
```

Before proceeding, it is essential to **configure your personal API endpoint and key** within the `run/online_generate.sh` script. Please refer to lines 66 to 69 to complete this setup.

By default, each run can only be performed on **a single subset consisting of 2500 samples**. If you want to use more training data or employ a different base model, you'll need to modify the sections of the code that are currently commented out.

After collecting the dataset, OPA and OPA-DPO dataset can be built up through:

```
python base_operations/make_opadpo_dataset.py
```

Please make sure that you have corresponding `opa_training_data-7B` (or `opa_training_data-13B`) and `opadpo_training_data-7B` (or `opadpo_training_data-13B`) in directory `./base_datasets/` after this step.

### Step 3: OPA Training

As presented in Fig.d, the third step is to conduct LoRA-SFT on the GT responses and revised responses. It can be conducted through:

```
bash run/train_opa.sh
```

Your OPA model will be saved at `./output/llava7b_opa_model/checkpoint-final` (or `output/llava13b_opa_model/checkpoint-final`) after this step.

### Step 4: OPA-DPO Training

As presented in Fig.d, the last (fourth) step is to initiate OPA-DPO training from the policy obtained in step3. Please run the following command:

```
bash run/train_opa_dpo.sh
```

Your final OPA-DPO model will be saved at `./output/llava7b_opadpo_model/checkpoint-final` (or `output/llava13b_opadpo_model/checkpoint-final`). It is a LoRA adapter and should be combined with its base model for final usage.

## Acknowledgements

We would like to express our gratitude for the code snippets provided in [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF), [FastChat](https://github.com/lm-sys/FastChat) and [TRL](https://github.com/huggingface/trl), and datasets provided in [RLAIF-V](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset). These resources have significantly contributed to the development of our project.

## Bibtex

If you find OPA-DPO helpful for your work, please cite

```
@article{yang2025opadpo,
  title={Mitigating Hallucinations in Large Vision-Language Models via DPO: On-Policy Data Hold the Key},
  author={Yang, Zhihe and Luo, Xufang and Han, Dongqi and Xu, Yunjian and Li, Dongsheng},
  journal={arXiv preprint arXiv:2501.09695},
  year={2025}
}
```
