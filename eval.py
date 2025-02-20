import os
import base64
from openai import OpenAI
import pandas as pd
import re
from tqdm import tqdm
from PIL import Image
import torch
import replicate
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

BASIC_PROMPT = "Does this fundus photo show optic disc swelling? Answer yes or no."

PROMPTS = {
    # 'Caption': ("Describe this photo."),
    # 'Basic': (BASIC_PROMPT),
    # 'Context': (
    #     f"""
    #     This fundus photo shows an optic disc.
        
    #     {BASIC_PROMPT}"""
    # ),
    # 'Chain of thought': (
    #     f"""
    #     This fundus photo shows an optic disc.

    #     Assess optic disc margin clarity (sharp/blurred/obscured), evaluate retinal vessel visibility through disc (clear/partially obscured/fully obscured), and check for peripapillary hemorrhage (present/absent), then synthesize findings to classify swelling.

    #     {BASIC_PROMPT}
    #     """
    # ),
    'Few-shot': (
        f"This fundus photo shows an optic disc. {BASIC_PROMPT}"
    ),
    'Role prompting': (
        f"""
        Your role is a Neuro-ophthalmologist attending. This fundus photo shows an optic disc.

        To assess whether an optic nerve is swollen, first assess optic disc margin clarity (sharp/blurred/obscured), then evaluate retinal vessel visibility through disc (clear/partially obscured/fully obscured), and check for peripapillary hemorrhage (present/absent), then synthesize findings to classify if the optic nerve appears to be swollen.

        {BASIC_PROMPT}
        """
    )
}

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

NORMAL_PATH = "/home/ttn/Development/vlm/normal.jpg"
PAPILLEDEMA_PATH = "/home/ttn/Development/vlm/papilledema.jpg"

NORMAL_IMG = Image.open(NORMAL_PATH).convert("RGB")
PAPILLEDEMA_IMG = Image.open(PAPILLEDEMA_PATH).convert("RGB")

FEW_SHOT_IMAGE_PATHS = [NORMAL_PATH, PAPILLEDEMA_PATH]
FEW_SHOT_IMG = [Image.open(img) for img in FEW_SHOT_IMAGE_PATHS]

NORMAL_IMG_BASE64 = base64.b64encode(open(NORMAL_PATH, "rb").read()).decode("utf-8")
PAPILLEDEMA_IMG_BASE64 = base64.b64encode(open(PAPILLEDEMA_PATH, "rb").read()).decode("utf-8")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------

blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
blip2_model.to("cpu")

llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
llava_model.to("cpu")

# --------------------------------------------------

def ask_gpt4v(image_path, prompt, few_shot=True):
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    if few_shot:
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
                                "url": f"data:image/jpeg;base64,{NORMAL_IMG_BASE64}",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text", 
                            "text": "no"
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": BASIC_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{PAPILLEDEMA_IMG_BASE64}",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text", 
                            "text": "yes"
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": BASIC_PROMPT
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            },
                        },
                    ],
                },
            ]
        )
    else:
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

def ask_deepseek_vl2(image_path, prompt, few_shot=True):
    image = open(image_path, "rb")
    if few_shot:
        images_list = [open(img, "rb") for img in FEW_SHOT_IMAGE_PATHS]
        images = [*images_list, image]
        input = {
            "images": images,
            "prompt": prompt + '<image>' + ''.join([BASIC_PROMPT + '<image>'] * (len(images) - 1)),
        }
    else:
        input = {
            "image": image, 
            "prompt": prompt + ' <image>' if prompt else '<image>'
        }
    output = replicate.run("deepseek-ai/deepseek-vl2:e5caf557dd9e5dcee46442e1315291ef1867f027991ede8ff95e304d4f734200", input)
    return output

def ask_blip2_local(image_path, prompt, few_shot=True):
    blip2_model.to(device)
    image = Image.open(image_path).convert("RGB")
    if few_shot:
        images = [*FEW_SHOT_IMG, image]
        inputs = blip2_processor(images=images, text=prompt + ' Answer:', return_tensors="pt").to(device)
    else:
        inputs = blip2_processor(images=image, text=prompt + ' Answer:', return_tensors="pt").to(device)
    output = blip2_model.generate(**inputs, max_new_tokens=500)
    return blip2_processor.batch_decode(output, skip_special_tokens=True)[0].replace(prompt + ' Answer:', '').strip()

def ask_llava_local(image_path, prompt, few_shot=True):
    llava_model.to(device)
    image = Image.open(image_path).convert("RGB")
    if few_shot:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": NORMAL_IMG},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "no"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASIC_PROMPT},
                    {"type": PAPILLEDEMA_IMG},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "yes"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASIC_PROMPT},
                    {"type": image},
                ],
            },
        ]
    else:
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

# --------------------------------------------------

def main():
    dataset_dir = "/home/ttn/Development/vlm/data/pseudopapilledema"

    image_paths = []
    classes = []
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if not file.startswith('Pseudopapilledema'):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
                classes.append(1 if root.split('/')[-1] == "Papilledema" else 0)
    
    labels_dict = dict(zip(image_paths, classes))
    
    models_funcs = {
        "GPT-4V": ask_gpt4v,
        # "BLIP2": ask_blip2_local,
        "LLaVa": ask_llava_local,
        "DeepSeek_VL2": ask_deepseek_vl2,
    }
    
    predictions_results = {n: {k: [] for k in PROMPTS.keys()} for n in models_funcs.keys()}
        
    for image_file in tqdm(labels_dict.keys()):
        image_path = os.path.join(dataset_dir, image_file)
        for model_name, func in models_funcs.items():
            if predictions_results[model_name].get("Image Name") is None:
                predictions_results[model_name]["Image Name"] = []
            predictions_results[model_name]["Image Name"].append(image_path.split('/')[-1])
            for prompt in PROMPTS.keys():
                try:
                    few_shot = PROMPTS[prompt] in ['Few-shot', 'Role prompting']
                    # print(f"Processing {image_file} with {model_name}...")
                    answer = func(image_path, PROMPTS[prompt], few_shot=few_shot)
                    # print(f"Raw answer from {model_name}: {answer}")
                    predictions_results[model_name][prompt].append(answer)
                except Exception as e:
                    # print(f"Error: {e}")
                    predictions_results[model_name][prompt].append('')

    for preds in predictions_results.keys():
        df = pd.DataFrame(predictions_results[preds])
        df.to_csv(f"{preds}_results.csv", index=False)

if __name__ == "__main__":
    main()