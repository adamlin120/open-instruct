# Taiwanese Instruction-following Language Models

<p align="center">
<img src="images/tulu_logo.png" width="200" />
</p>

## Setup

You can install the required packages by running the following command (after installing pytorch):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc


conda create -n open-instruct python=3.10
conda activate open-instruct
#pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 
pip3 install --upgrade --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 
pip install -r llama2_requirements.txt && pip install flash-attn>=2.0.0 --no-build-isolation
```


## Training

### Pretrain

```bash
bash pretrain_sft_flash.sh 13 1 0
```


### Instruction-tuning

```bash
./scripts/finetune_with_hf_trainer.sh
```


## Demo and Model Checkpoints

We provide a number of model checkpoints as diffs. You can find them on Hugging Face [here](https://huggingface.co/yentinglin). They are also all here:

### Licensing

The is licensed under Apache 2.0 as given in `LICENSE`.

# Citation

If you used this repository or our models, please cite our work:
```
@inproceedings{lin-chen-2023-llm,
    title = "{LLM}-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations with Large Language Models",
    author = "Lin, Yen-Ting  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 5th Workshop on NLP for Conversational AI (NLP4ConvAI 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.nlp4convai-1.5",
    pages = "47--58",
    abstract = "We propose LLM-Eval, a unified multi-dimensional automatic evaluation method for open-domain conversations with large language models (LLMs). Existing evaluation methods often rely on human annotations, ground-truth responses, or multiple LLM prompts, which can be expensive and time-consuming. To address these issues, we design a single prompt-based evaluation method that leverages a unified evaluation schema to cover multiple dimensions of conversation quality in a single model call. We extensively evaluate the performance of LLM-Eval on various benchmark datasets, demonstrating its effectiveness, efficiency, and adaptability compared to state-of-the-art evaluation methods. Our analysis also highlights the importance of choosing suitable LLMs and decoding strategies for accurate evaluation results. LLM-Eval offers a versatile and robust solution for evaluating open-domain conversation systems, streamlining the evaluation process and providing consistent performance across diverse scenarios.",
}
```

