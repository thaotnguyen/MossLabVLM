import torch
from tqdm import tqdm
import os
import re
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

PROMPT = (
    "Analyze this retinal fundus image of the optic disc. Is the optic disc swollen? "
    "Please answer with your answer enclosed in curly braces, e.g., {swollen} or {not swollen}."
)

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

def compute_metrics(predictions, true_labels):
    TP = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    TN = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    FP = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    FN = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else None
    specificity = TN / (TN + FP) if (TN + FP) > 0 else None
    return TP, TN, FP, FN, sensitivity, specificity

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

dataset_dir = "/home/ttn/Development/vlm/data/pseudopapilledema"

image_paths = []
classes = []

for root, _, files in os.walk(dataset_dir):
    for file in files:
        image_path = os.path.join(root, file)
        image_paths.append(image_path)
        classes.append(1 if root.split('/')[-1] == "Papilledema" else 0)

labels_dict = dict(zip(image_paths, classes))

predictions_results = []
true_labels_list = []
    
for image_file in tqdm(image_paths, desc='Evaluating... '):
    image_path = os.path.join(dataset_dir, image_file)
    true_label = labels_dict[image_file]
    true_labels_list.append(true_label)

    # multiple images/interleaved image-text
    conversation = [
        {
            "role": "<|User|>",
            "content": PROMPT,
            "images": [image_path],
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(vl_gpt.device)

    with torch.no_grad():
        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # incremental_prefilling when using 40G GPU for vl2-small
        inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512 # prefilling size
        )

        # run the model to get the response
        outputs = vl_gpt.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,

            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,

            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(), skip_special_tokens=False)

    prediction = parse_prediction(answer)
    predictions_results.append(prediction)

valid = [(p, t) for p, t in zip(predictions_results, true_labels_list) if p is not None]

preds_filtered, true_filtered = zip(*valid)
TP, TN, FP, FN, sensitivity, specificity = compute_metrics(preds_filtered, true_filtered)
print("\n------------------------------------------")
print(f"Metrics for DeepSeek-VL2:")
print(f"  True Positives: {TP}")
print(f"  True Negatives: {TN}")
print(f"  False Positives: {FP}")
print(f"  False Negatives: {FN}")
print(f"  Sensitivity (TPR): {sensitivity}")
print(f"  Specificity (TNR): {specificity}")
print("------------------------------------------\n")