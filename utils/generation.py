import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

def build_llava_prompt(question):
    """
    Standard generation prompt for LLaVA-1.5 models.
    """
    return f"USER: <image>\n{question}\nASSISTANT:"

def generate_answers_for_pope(model, processor, dataset, output_file, max_new_tokens=10):
    """
    Generates answers for the POPE dataset.
    """
    predictions = []
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Starting generation for POPE ({len(dataset)} examples)...")
    for item in tqdm(dataset):
        image = item['image']
        question = item['question']
        
        # Depending on POPE schema, it might be question_id or id
        q_id = item.get('question_id', item.get('id', None))
        
        # Build prompt
        prompt = build_llava_prompt(question)
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        # Decode
        # We only want the generated part, so we slice the output ids
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_len:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        predictions.append({
            "question_id": q_id,
            "text": output_text
        })
        
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
        
    return predictions

def generate_answers_for_chair(model, processor, images_dataset, output_file, prompt="Please describe this image in detail.", max_new_tokens=100):
    """
    Generates captions for CHAIR evaluation.
    images_dataset: Can be a list of dicts with {'image_id': ..., 'image': PIL.Image} 
                    OR a HuggingFace dataset containing such fields.
    """
    predictions = []
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Starting CHAIR caption generation ({len(images_dataset)} examples)...")
    for item in tqdm(images_dataset):
        image = item['image']
        image_id = item['image_id']
        
        full_prompt = build_llava_prompt(prompt)
        
        inputs = processor(text=full_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_len:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        predictions.append({
            "image_id": image_id,
            "caption": output_text
        })
        
    print(f"Saving CHAIR predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
        
    return predictions
