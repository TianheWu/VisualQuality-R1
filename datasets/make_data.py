import random
import json
import math
import numpy as np
import random
import os
import torch


random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




PROMPT_POOL = [
    "What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality."
]


def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def convert_to_rl_jsonl(data, dataset_name, output_file):
    converted_data = []
    for item in data:
        problem = random.choice(PROMPT_POOL)
        solution = ""
        images = []
        
        for message in item["messages"]:
            if message["role"] == "assistant":
                for content in message["content"]:
                    if content["type"] == "text":
                        solution = content["text"]
        
        images = item["images"]
        new_item = {
            "image": images,
            "problem": problem,
            "solution": solution
        }
        converted_data.append(new_item)
    
    out_data = []
    for index, item in enumerate(converted_data):
        converted_item = {
            "id": index + 1,
            "dataset_name": dataset_name,
            "image": item["image"],
            "conversations": [
                {
                    "from": "human",
                    "value": f"You are doing the image quality assessment task. Here is the question: {item['problem']}"
                },
                {
                    "from": "gpt",
                    "value": item["solution"]
                }
            ]
        }
        out_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in out_data:
            json.dump(item, f)
            f.write("\n") 
    f.close()


def scoring_images(dataset_name, file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    result = []
    for line in lines:
        columns = line.strip().split()
        if dataset_name in ["KADID-10K"]:
            ref, dis, mos, std = columns
        elif dataset_name in ["AGIQA-3K", "KONIQ-10K", "SPAQ", "LIVEC"]:
            dis, mos, std = columns
        elif dataset_name in ["BID"]:
            dis, mos = columns 
        elif dataset_name in ["PIPAL", "TID2013", "CSIQ", "Deblur", "CVIU", "SHRQ"]:
            ref, dis, mos = columns
        else:
            print(f"Warning: Unexpected number of columns in line: {line.strip()}")
        
        mos = float(mos)
        result.append([dis, mos])

    return result


def generate_temp_json_data(comparison_result):
    prompt = (
        "--"
    )
    json_data = []

    for item in comparison_result:
        if len(item) == 2:
            dis, mos = item
        else:
            print(f"Warning: Unexpected number of columns in line: {line.strip()}")
        
        user_message = {
            "content": [
                {
                    "index": None,
                    "text": prompt,
                    "type": "text"
                }
            ],
            "role": "user"
        }

        # Determine the answer based on the comparison result
        answer = mos
        assistant_message = {
            "content": [
                {
                    "index": None,
                    "text": answer,
                    "type": "text"
                }
            ],
            "role": "assistant"
        }

        # Add the images (dis1 and dis2)
        images = [dis]

        # Append the data to the JSON list
        json_data.append({
            "messages": [user_message, assistant_message],
            "images": images
        })

    return json_data

def read_data_from_file(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(line)
    return data

def split_data_by_reference(data, train_ratio=0.6, val_ratio=0.2):
    reference_images = list(set(line.split()[0] for line in data))
    random.shuffle(reference_images)

    split_index_train = round(len(reference_images) * train_ratio)
    split_index_val = round(len(reference_images) * val_ratio)
    train_references = set(reference_images[:split_index_train])
    val_references = set(reference_images[split_index_train:split_index_train + split_index_val])
    test_references = set(reference_images[split_index_train + split_index_val:])

    train_set = [line for line in data if line.split()[0] in train_references]
    val_set = [line for line in data if line.split()[0] in val_references]
    test_set = [line for line in data if line.split()[0] in test_references]

    return train_set, val_set, test_set

def save_to_file(data, filename):
    with open(filename, "w") as file:
        for line in data:
            file.write(line + "\n")

def normalize(input_file, output_file, dataset_name):
    data = []
    with open(input_file, "r") as file:
        for line in file:
            parts = line.strip().split()
            if dataset_name in ["AGIQA-3K", "KONIQ-10K", "SPAQ", "LIVEC"]:
                distorted, mos, std = parts
                data.append((distorted, float(mos), float(std)))
            elif dataset_name in ["BID"]:
                distorted, mos = parts
                data.append((distorted, float(mos)))
            elif dataset_name == "KADID-10K":
                reference, distorted, mos, std = parts
                data.append((reference, distorted, float(mos), float(std)))
            elif dataset_name in ["PIPAL", "TID2013", "CSIQ", "Deblur", "CVIU", "SHRQ"]:
                reference, distorted, mos = parts
                data.append((reference, distorted, float(mos)))
            else:
                raise ValueError(f"Choose dataset name in {DATASET_NAMES}")

    if dataset_name in ["AGIQA-3K", "KONIQ-10K", "SPAQ", "LIVEC", "BID"]:
        mos_scores = [item[1] for item in data]
    elif dataset_name in ["KADID-10K", "PIPAL", "TID2013", "CSIQ", "Deblur", "CVIU", "SHRQ"]:
        mos_scores = [item[2] for item in data]
    else:
        raise ValueError(f"Choose dataset name in {DATASET_NAMES}")
    
    min_mos = min(mos_scores)
    max_mos = max(mos_scores)

    print(f"dataset name {dataset_name}, min mos {min_mos}")
    print(f"dataset name {dataset_name}, max mos {max_mos}")

    normalized_data = []
    for item in data:
        if dataset_name in ["AGIQA-3K", "KONIQ-10K", "SPAQ", "LIVEC"]:
            distorted, mos, std = item
            normalized_mos = (mos - min_mos) / (max_mos - min_mos) * 4 + 1
            normalized_data.append((distorted, normalized_mos, std))
        elif dataset_name in ["BID"]:
            distorted, mos = item
            normalized_mos = (mos - min_mos) / (max_mos - min_mos) * 4 + 1
            normalized_data.append((distorted, normalized_mos))
        elif dataset_name == "KADID-10K":
            reference, distorted, mos, std = item
            normalized_mos = (mos - min_mos) / (max_mos - min_mos) * 4 + 1
            normalized_data.append((reference, distorted, normalized_mos, std))
        elif dataset_name in ["PIPAL", "TID2013", "CSIQ", "Deblur", "CVIU", "SHRQ"]:
            reference, distorted, mos = item
            normalized_mos = (mos - min_mos) / (max_mos - min_mos) * 4 + 1
            normalized_data.append((reference, distorted, normalized_mos))

    with open(output_file, "w") as file:
        for item in normalized_data:
            if dataset_name in ["AGIQA-3K", "KONIQ-10K", "SPAQ", "LIVEC"]:
                distorted, normalized_mos, std = item
                file.write(f"{distorted} {normalized_mos} {std}\n")
            elif dataset_name in ["BID"]:
                distorted, normalized_mos = item
                file.write(f"{distorted} {normalized_mos}\n")
            elif dataset_name == "KADID-10K":
                reference, distorted, normalized_mos, std = item
                file.write(f"{reference} {distorted} {normalized_mos} {std}\n")
            elif dataset_name in ["PIPAL", "TID2013", "CSIQ", "Deblur", "CVIU", "SHRQ"]:
                reference, distorted, normalized_mos = item
                file.write(f"{reference} {distorted} {normalized_mos}\n")

    print(f"normalized data are saved in {output_file}")


if __name__ == "__main__":
    dataset_name = "KADID-10K"

    # Norm
    input_file = f"./{dataset_name}/{dataset_name}_mos.txt"
    norm_file = f"./{dataset_name}/{dataset_name}_normalized_mos.txt" 
    normalize(input_file, norm_file, dataset_name)

    # Spliting dataset
    data = read_data_from_file(norm_file)

    train_set, val_set, test_set = split_data_by_reference(data, train_ratio=0.6, val_ratio=0.2)
    train_set_path = f"./{dataset_name}/{dataset_name}_train_set.txt"
    val_set_path = f"./{dataset_name}/{dataset_name}_val_set.txt"
    test_set_path = f"./{dataset_name}/{dataset_name}_test_set.txt"

    save_to_file(train_set, train_set_path)
    save_to_file(val_set, val_set_path)
    save_to_file(test_set, test_set_path)

    scoring_result = scoring_images(dataset_name, train_set_path)

    # Generate JSON data
    json_data = generate_temp_json_data(scoring_result)

    # Save to a JSON file
    output_file = f"./{dataset_name}/scoring/Temp-{dataset_name}_train_scoring.json"
    folder_path = os.path.dirname(output_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"JSON data saved to {output_file}")

    temp_data = read_json_file(output_file)
    rl_save_data_path = f"./{dataset_name}/scoring/RL-{dataset_name}_train_scoring.jsonl"
    rl_data = convert_to_rl_jsonl(temp_data, dataset_name, rl_save_data_path)
    
    print(f"JSON data saved to {rl_save_data_path}")

    if os.path.exists(output_file):
        os.remove(output_file)

