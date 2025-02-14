import os
import base64
from openai import OpenAI
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

PROMPTS = {
    'Description': None,
    'Anatomical structure': (
        "Question: What anatomical structure in the eye is this? Reply in less than 5 words Answer:"
    ),
    'Disc swelling decision': (
        "Question: This photo shows an optic nerve, does the optic nerve appear to be swollen? Reply either yes or no? Answer:"
    )
}

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

def ask_gpt4v(image_path, prompt):
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
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ] if prompt else [
            {
                "role": "user",
                "content": [
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

def ask_blip2_local(image_path, prompt):
    blip2_model.to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = blip2_processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = blip2_model.generate(**inputs, max_new_tokens=500)
    return blip2_processor.batch_decode(output, skip_special_tokens=True)[0]

def ask_llava_local(image_path, prompt):
    llava_model.to(device)
    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        },
    ] if prompt else [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            ],
        },
    ]
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")
    output = llava_model.generate(**inputs, max_new_tokens=500)
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
    
    predictions_results = {n: {k: [] for k in PROMPTS.keys()} for n in models_funcs.keys()}
        
    for image_file in labels_dict.keys():
        image_path = os.path.join(dataset_dir, image_file)
        for model_name, func in models_funcs.items():
            if predictions_results[model_name].get("Image Name") is None:
                predictions_results[model_name]["Image Name"] = []
            predictions_results[model_name]["Image Name"].append(image_path.split('/')[-1])
            for prompt in PROMPTS.keys():
                print(f"Processing {image_file} with {model_name}...")
                answer = func(image_path, PROMPTS[prompt])
                print(f"Raw answer from {model_name}: {answer}")
                predictions_results[model_name][prompt].append(answer)

    for preds in predictions_results.keys():
        df = pd.DataFrame(predictions_results[preds])
        df.to_csv(f"{preds}_results.csv", index=False)

if __name__ == "__main__":
    main()