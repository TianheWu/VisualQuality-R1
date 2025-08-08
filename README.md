<p align="center">
    <img src="images/logo.png" width="400">
</p>


<div align="center">

# Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank

This is the official code of VisualQuality-R1.

<p align="center">
    <img src="images/intro.png" width="700">
</p>


<a href="https://arxiv.org/abs/2505.14460" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-VisualQuality--R1-red?logo=arxiv" height="25" />
</a>
<a href="https://huggingface.co/TianheWu/VisualQuality-R1-7B" target="_blank">
    <img alt="HF Model: VisualQuality-R1" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-VisualQuality--R1--7B-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://huggingface.co/TianheWu/VisualQuality-R1-7B-preview" target="_blank">
    <img alt="HF Model: VisualQuality-R1" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-Preview--Version-ffc107?color=ffc107&logoColor=white" height="25" />
</a>


<div style="font-family: charter;">
    <a href="https://tianhewu.github.io/tianhe-page/" target="_blank">Tianhe Wu</a>,
    <a href="https://scholar.google.com/citations?user=-jj3Ub8AAAAJ&hl=zh-CN&authuser=2" target="_blank">Jian Zou</a>,
    <a href="https://scholar.google.com/citations?user=REWxLZsAAAAJ&hl=en" target="_blank">Jie Liang</a>,
    <a href="https://www4.comp.polyu.edu.hk/~cslzhang/" target="_blank">Lei Zhang*</a>,
    <a href="https://kedema.org/" target="_blank">Kede Ma*</a>,
    <br>
    * denotes the corresponding author
</div>

</div align="center">

<br>

> *The first NR-IQA model enhanced by RL2R, capable of both quality description and rating through reasoning.*

## Release
- [06/05/25] 🔥 We have released the training code of VisualQuality-R1.
- [05/25/25] 🤗 We have released the final version of [VisualQuality-R1-7B](https://huggingface.co/TianheWu/VisualQuality-R1-7B).
- [05/16/25] 🤗 We have released a preview version of [VisualQuality-R1-7B-preview](https://huggingface.co/TianheWu/VisualQuality-R1-7B-preview), trained on three datasets: KADID-10K, TID2013, and KonIQ-10k.


## Installation
We currently support training on A100 and A800 GPUs. To set up the training environment, use the following commands:
```
conda create -n visualquality python=3.11.10
conda activate visualquality

bash setup.sh
```

## 🚀 Updated: VisualQuality-R1 high efficiency inference script with vLLM

<details>
<summary>Example Code (VisualQuality-R1: Batch Images Quality Rating with thinking, using vLLM)</summary>

```python
# Please install vLLM first: https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html

from transformers import Qwen2_5_VLProcessor, AutoProcessor
from vllm import LLM, RequestOutput, SamplingParams
from qwen_vl_utils import process_vision_info

import torch
import random
import re
import os

IMAGE_PATH = "./images"
MODEL_PATH = "TianheWu/VisualQuality-R1-7B"

def get_image_paths(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def score_batch_image(image_paths, model: LLM, processor: Qwen2_5_VLProcessor):
    PROMPT = (
        "You are doing the image quality assessment task. Here is the question: "
        "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
        "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
    )

    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."

    messages = []
    for img_path in image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {'type': 'image', 'image': img_path},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=PROMPT)}
                ],
            }
        ]
        messages.append(message)

    all_outputs = []  # List to store all answers

    # Preparation for inference
    print("preprocessing ...")
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = [{
        "prompt": texts[i],
        "multi_modal_data": {
            "image": image_inputs[i]
        },
    } for i in range(len(messages))]
    
    output: list[RequestOutput] = model.generate(
        inputs,
        sampling_params=SamplingParams(
            max_tokens=512,
            temperature=0.1,
            top_k=50,
            top_p=1.0,
            stop_token_ids=[processor.tokenizer.eos_token_id],
        ),
    )

    batch_output_text = [o.outputs[0].text for o in output]

    all_outputs.extend(batch_output_text)
    
    path_score_dict = {}
    for img_path, model_output in zip(image_paths, all_outputs):
        print(f"{model_output = }")
        try:
            model_output_matches = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else model_output.strip()
            score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
        except:
            print(f"Meet error with {img_path}, please generate again.")
            score = random.randint(1, 5)

        path_score_dict[img_path] = score

    return path_score_dict


random.seed(1)
model = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    trust_remote_code=True,
    seed=1,
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

image_paths = get_image_paths(IMAGE_PATH) # It should be a list

path_score_dict = score_batch_image(
    image_paths, model, processor
)

file_name = "output.txt"
with open(file_name, "w") as file:
    for key, value in path_score_dict.items():
        file.write(f"{key} {value}\n") 

print("Done!")
```
</details>

## Training

### Preparation
1. To smoothly execute the training procedure, first download the IQA images and place them all in a **single folder**.
2. Given an original MOS file (e.g., KADID-10K_mos.txt), first execute `cd datasets`, then run `python make_data.py` (with moderate modifications) to generate a **JSON file** for model training.
3. Download the [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) into a folder.

### Training within a Single Node
Please modify three elements in `src/open-r1-multimodal/run_scripts/KADID-10K/one_node_run_kadid.sh`:
```
--model_name_or_path [Your Qwen2.5-VL-7B-Instruct path] \
--image_folders [Your dataset images path] \
--data_file_paths [Your JSON file path] \
```
Then, run:
```
bash src/open-r1-multimodal/run_scripts/KADID-10K/one_node_run_kadid.sh
```

### Training within Multiple Nodes
After making the necessary modifications, run the following command:
```
bash src/open-r1-multimodal/run_scripts/KADID-10K/multi_run_kadid.sh
```


## Quick Start for Inference
This section includes the usages of **VisualQuality-R1**.

<details>
<summary>Example Code (Single Image Quality Rating)</summary>
    
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import random
import re
import os


def score_image(image_path, model, processor):
    PROMPT = (
        "You are doing the image quality assessment task. Here is the question: "
        "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
        "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. "
        "First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."
    )
        
    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."
    # QUESTION_TEMPLATE = "Please describe the quality of this image."
    message = [
        {
            "role": "user",
            "content": [
                {'type': 'image', 'image': image_path},
                {"type": "text", "text": PROMPT}
            ],
        }
    ]

    batch_messages = [message]

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=True, top_k=50, top_p=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    reasoning = re.findall(r'<think>(.*?)</think>', batch_output_text[0], re.DOTALL)
    reasoning = reasoning[-1].strip()

    try:
        model_output_matches = re.findall(r'<answer>(.*?)</answer>', batch_output_text[0], re.DOTALL)
        model_answer = model_output_matches[-1].strip() if model_output_matches else batch_output_text[0].strip()
        score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
    except:
        print(f"================= Meet error with {img_path}, please generate again. =================")
        score = random.randint(1, 5)

    return reasoning, score


random.seed(1)
MODEL_PATH = ""
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
image_path = ""

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

reasoning, score = score_image(
    image_path, model, processor
)

print(reasoning)
print(score)
```
</details>


<details>
<summary>Example Code (Batch Images Quality Rating with thinking)</summary>

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

import torch
import random
import re
import os


def get_image_paths(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths

def score_batch_image(image_paths, model, processor):
    PROMPT = (
        "You are doing the image quality assessment task. Here is the question: "
        "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, "
        "rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
    )

    QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags."

    messages = []
    for img_path in image_paths:
        message = [
            {
                "role": "user",
                "content": [
                    {'type': 'image', 'image': img_path},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=PROMPT)}
                ],
            }
        ]
        messages.append(message)

    BSZ = 32
    all_outputs = []  # List to store all answers
    for i in tqdm(range(0, len(messages), BSZ)):
        batch_messages = messages[i:i + BSZ]
    
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=True, top_k=50, top_p=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        all_outputs.extend(batch_output_text)
    
    path_score_dict = {}
    for img_path, model_output in zip(image_paths, all_outputs):
        reasoning = re.findall(r'<think>(.*?)</think>', model_output, re.DOTALL)
        reasoning = reasoning[-1].strip()

        try:
            model_output_matches = re.findall(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            model_answer = model_output_matches[-1].strip() if model_output_matches else model_output.strip()
            score = float(re.search(r'\d+(\.\d+)?', model_answer).group())
        except:
            print(f"Meet error with {img_path}, please generate again.")
            score = random.randint(1, 5)

        path_score_dict[img_path] = score

    return path_score_dict


random.seed(1)
MODEL_PATH = ""
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

image_root = ""
image_paths = get_image_paths(image_root) # It should be a list

path_score_dict = score_batch_image(
    image_paths, model, processor
)

file_name = "output.txt"
with open(file_name, "w") as file:
    for key, value in path_score_dict.items():
        file.write(f"{key} {value}\n") 

print("Done!")
```
</details>


<details>
<summary>Example Code (Images Inference)</summary>

You can prompt anything what you like in the following commands (including multi-image as input)
```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import random
import re
import os


def generate(image_paths, model, prompt, processor):
    message = [
        {
            "role": "user",
            "content": [
                *({'type': 'image', 'image': img_path} for img_path in image_paths),
                {"type": "text", "text": prompt}
            ],
        }
    ]

    batch_messages = [message]

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True) for msg in batch_messages]
    image_inputs, video_inputs = process_vision_info(batch_messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=True, top_k=50, top_p=1)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return batch_output_text[0]


random.seed(1)
MODEL_PATH = ""
device = torch.device("cuda:5") if torch.cuda.is_available() else torch.device("cpu")
image_path = [
    "",
    ""
]

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=device,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.tokenizer.padding_side = "left"

prompt = "Please describe the quality of given two images."
answer = generate(
    image_path, model, prompt, processor
)

print(answer)
```
</details>

### Example
The image shown was captured using the OPPO Find X6 Pro.
<p align="center">
    <img src="images/realistic_scenario.png" width="600">
</p>

## Acknowledgement
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1): We start from codebase from the VLM-R1.

## Related Projects
- [ECCV 2024] [A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment](https://arxiv.org/abs/2403.10854v2)
- [CVPR 2025] [Toward Generalized Image Quality Assessment: Relaxing the Perfect Reference Quality Assumption](https://www.arxiv.org/abs/2503.11221)

## 📧 Contact
If you have any question, please email `sigstianhewu@gmail.com` or `tianhewu@cityu.edu.hk`.

## BibTeX
```
@article{wu2025visualquality,
  title={{VisualQuality-R1}: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank},
  author={Wu, Tianhe and Zou, Jian and Liang, Jie and Zhang, Lei and Ma, Kede},
  journal={arXiv preprint arXiv:2505.14460},
  year={2025}
}
```



