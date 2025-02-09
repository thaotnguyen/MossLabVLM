import os
import time
import base64
import re
from openai import OpenAI
import requests
import pandas as pd
from PIL import Image
import torch
from transformers import (
    AutoProcessor, 
    Blip2ForConditionalGeneration,
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)

# Set API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

PROMPT = (
    "Analyze this retinal fundus image of the optic disc. Is the optic disc swollen? "
    "Please answer with your answer enclosed in curly braces, e.g., {swollen} or {not swollen}."
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------

blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
blip2_model.to("cpu")

llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
llava_model.to("cpu")

# --------------------------------------------------

def parse_prediction(text):
    """
    Extract the answer within curly braces.
    Returns 1 if the answer is exactly "swollen", 0 if "not swollen".
    If the answer is ambiguous, additional keywords are used as a fallback.
    """
    match = re.search(r'\{([^}]+)\}', text)
    if match:
        answer = match.group(1).strip().lower()
        if answer == "swollen":
            return 1
        elif answer == "not swollen":
            return 0
        else:
            # Fallback: check for keywords
            if "swollen" in answer and "not" not in answer:
                return 1
            elif "not swollen" in answer or "normal" in answer or "healthy" in answer or "no swelling" in answer:
                return 0
            else:
                return 0
    else:
        # Fallback if no braces found
        text_lower = text.lower()
        if "not swollen" in text_lower or "normal" in text_lower or "healthy" in text_lower or "no swelling" in text_lower:
            return 0
        elif ("swollen" in text_lower and "not" not in text_lower) or "edematous" in text_lower:
            return 1
        else:
            return 0

def ask_gpt4v(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ],
    )
    answer = response.choices[0].message.content
    return answer

def ask_blip2_local(image_path):
    blip2_model.to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = blip2_processor(images=image, text='Question: ' + PROMPT + '\nAnswer: ', return_tensors="pt").to(device)
    output = blip2_model.generate(**inputs, max_new_tokens=100)
    return blip2_processor.decode(output[0], skip_special_tokens=True).replace(PROMPT, "").strip()

def ask_llava_local(image_path):
    llava_model.to(device)
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image"},
            ],
        },
    ]
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = llava_model.generate(**inputs, max_new_tokens=100)
    return llava_processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

def compute_metrics(predictions, true_labels):
    TP = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    TN = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    FP = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    FN = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else None
    specificity = TN / (TN + FP) if (TN + FP) > 0 else None
    return TP, TN, FP, FN, sensitivity, specificity

# --------------------------------------------------

def main():
    dataset_dir = "/home/ttn/Development/vlm/data/pseudopapilledema"

    image_paths = []
    classes = []
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            image_path = os.path.join(root, file)
            image_paths.append(image_path)
            classes.append(1 if root.split('/')[-1] == "Papilledema" else 0)
    
    labels_dict = dict(zip(image_paths, classes))
    
    models_funcs = {
        "GPT-4V": ask_gpt4v,
        "BLIP2": ask_blip2_local,
        "LLaVa": ask_llava_local,
    }
    
    predictions_results = {name: [] for name in models_funcs}
    true_labels_list = []
        
    for image_file in image_paths:
        image_path = os.path.join(dataset_dir, image_file)
        true_label = labels_dict[image_file]
        true_labels_list.append(true_label)
        for model_name, func in models_funcs.items():
            try:
                print(f"Processing {image_file} with {model_name}...")
                answer = func(image_path)
                print(f"Raw answer from {model_name}: {answer}")
                prediction = parse_prediction(answer)
                predictions_results[model_name].append(prediction)
            except Exception as e:
                print(f"Error processing {image_file} with {model_name}: {e}")
                predictions_results[model_name].append(None)
            time.sleep(0.1)  # brief pause to avoid rate limits
    
    for model_name, preds in predictions_results.items():
        valid = [(p, t) for p, t in zip(preds, true_labels_list) if p is not None]
        if not valid:
            print(f"No valid predictions for {model_name}.")
            continue
        preds_filtered, true_filtered = zip(*valid)
        TP, TN, FP, FN, sensitivity, specificity = compute_metrics(preds_filtered, true_filtered)
        print("\n------------------------------------------")
        print(f"Metrics for {model_name}:")
        print(f"  True Positives: {TP}")
        print(f"  True Negatives: {TN}")
        print(f"  False Positives: {FP}")
        print(f"  False Negatives: {FN}")
        print(f"  Sensitivity (TPR): {sensitivity}")
        print(f"  Specificity (TNR): {specificity}")
        print("------------------------------------------\n")

if __name__ == "__main__":
    main()