import argparse
import os
import sys
import json
import pickle
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

# Append parent dir to path if running directly so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.load_model import model, processor
from utils.generation import generate_answers_for_chair
from eval.chair import CHAIR
from hypothesis.h3 import run_h3
from solution.h3_attention_amplified import intervene_h3_attention_amplified

def build_h3_results_on_the_fly(model, processor, coco_dir, coco_annotations, limit=500):
    from pycocotools.coco import COCO
    print("Initializing COCO...")
    coco = COCO(coco_annotations)
    img_ids = coco.getImgIds()[:limit]
    imgs = coco.loadImgs(img_ids)
    
    chair_dataset = []
    print(f"Loading {len(imgs)} baseline images for H3 probing...")
    for img_info in imgs:
        img_path = os.path.join(coco_dir, img_info['file_name'])
        try:
            image = Image.open(img_path).convert("RGB")
            chair_dataset.append({
                "image_id": img_info['id'],
                "image": image
            })
        except Exception as e:
            continue
            
    temp_baseline_file = "results/temp_h3_baseline_chair.json"
    generate_answers_for_chair(model, processor, chair_dataset, temp_baseline_file)
    
    print("Evaluating baseline captions using CHAIR...")
    # Get CHAIR annotations directory from coco_annotations path
    coco_anno_dir = os.path.dirname(os.path.abspath(coco_annotations))
    evaluator = CHAIR(coco_anno_dir)
    output = evaluator.compute_chair(temp_baseline_file, "image_id", "caption")
    
    annotations = {}
    for item in output['sentences']:
        imid = item['image_id']
        cap = item['caption']
        h_words = [tup[1] for tup in item['mscoco_hallucinated_words']]
        gt = set(item['mscoco_gt_words'])
        gen = item['mscoco_generated_words']
        g_words = [w for w in gen if w in gt]
        
        img_info = coco.loadImgs(imid)[0]
        file_name = img_info['file_name']
        
        annotations[imid] = {
            "image_id": imid,
            "file_name": file_name,
            "generated_caption": cap,
            "grounded_words": g_words,
            "hallucinated_words": h_words
        }
        
    print("Running H3 to compute existence direction...")
    h3_results = run_h3(model, processor, annotations, image_dir=coco_dir, max_images=limit)
    return h3_results

def generate_answers_for_pope_h3(model, processor, dataset, output_file, h3_results, coco, max_new_tokens=10):
    predictions = []
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Starting H3 Attention Amplified generation for POPE ({len(dataset)} examples)...")
    for item in tqdm(dataset):
        image = item['image']
        question = item['question']
        q_id = item.get('question_id', item.get('id', None))
        
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        try:
            output_text, _, _ = intervene_h3_attention_amplified(
                model=model,
                processor=processor,
                image=image,
                h3_results=h3_results,
                coco=coco,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"Error generating for image {q_id}: {e}")
            output_text = ""
            
        predictions.append({
            "question_id": q_id,
            "text": output_text.strip()
        })
        
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)

def generate_answers_for_chair_h3(model, processor, images_dataset, output_file, h3_results, coco, max_new_tokens=100):
    predictions = []
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Starting H3 Attention Amplified generation for CHAIR ({len(images_dataset)} examples)...")
    for item in tqdm(images_dataset):
        image = item['image']
        image_id = item['image_id']
        
        prompt = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
        
        try:
            output_text, _, _ = intervene_h3_attention_amplified(
                model=model,
                processor=processor,
                image=image,
                h3_results=h3_results,
                coco=coco,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
        except Exception as e:
            print(f"Error generating for image {image_id}: {e}")
            output_text = ""
            
        predictions.append({
            "image_id": image_id,
            "caption": output_text.strip()
        })
        
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="H3 Attention Amplified Benchmark")
    parser.add_argument("--run_pope", action="store_true")
    parser.add_argument("--run_chair", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--coco_dir", type=str, default=None)
    parser.add_argument("--coco_annotations", type=str, default=None)
    parser.add_argument("--h3_results_path", type=str, default="results/h3_results.pkl")
    parser.add_argument("--pope_output", type=str, default="results/h3_attention_amplified_pope.json")
    parser.add_argument("--chair_output", type=str, default="results/h3_attention_amplified_chair.json")
    
    args = parser.parse_args()
    
    if not (args.run_pope or args.run_chair):
        print("Please specify --run_pope and/or --run_chair")
        return
        
    coco = None
    if args.run_chair or not os.path.exists(args.h3_results_path):
        from pycocotools.coco import COCO
        if not args.coco_annotations:
            print("Please provide --coco_annotations for COCO object needed by H3/CHAIR.")
            return
            
        print("Loading COCO annotations...")
        coco = COCO(args.coco_annotations)

    # Load or generate h3_results
    if os.path.exists(args.h3_results_path):
        print(f"Loading pre-computed H3 results from {args.h3_results_path}")
        with open(args.h3_results_path, 'rb') as f:
            h3_results = pickle.load(f)
    else:
        print("H3 results not found. Generating on-the-fly using 500 images...")
        if not args.coco_dir:
            print("Please provide --coco_dir to generate h3_results on the fly.")
            return
        h3_results = build_h3_results_on_the_fly(model, processor, args.coco_dir, args.coco_annotations, limit=500)
        os.makedirs(os.path.dirname(args.h3_results_path), exist_ok=True)
        with open(args.h3_results_path, 'wb') as f:
            pickle.dump(h3_results, f)
            print(f"Saved h3_results to {args.h3_results_path}")

    if not h3_results:
        print("Failed to obtain valid h3_results. Aborting.")
        return

    # POPE
    if args.run_pope:
        print("Loading POPE dataset...")
        try:
            pope_dataset = load_dataset("lmms-lab/POPE", split="test")
            if args.limit is not None:
                pope_dataset = pope_dataset.select(range(min(args.limit, len(pope_dataset))))
                
            generate_answers_for_pope_h3(model, processor, pope_dataset, args.pope_output, h3_results, coco)
        except Exception as e:
            print(f"Failed to run POPE: {e}")

    # CHAIR
    if args.run_chair:
        if not args.coco_dir:
            print("Please provide --coco_dir for CHAIR.")
            return
            
        print("Preparing CHAIR dataset...")
        chair_dataset = []
        if os.path.exists(args.coco_dir):
            for filename in os.listdir(args.coco_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(args.coco_dir, filename)
                    try:
                        image_id_str = ''.join(filter(str.isdigit, filename.split('_')[-1]))
                        if not image_id_str:
                            continue
                        image_id = int(image_id_str)
                    except Exception:
                        continue
                        
                    chair_dataset.append({
                        "image_id": image_id,
                        "image": Image.open(img_path).convert("RGB")
                    })
                    
                    if args.limit and len(chair_dataset) >= args.limit:
                        break
                        
        if chair_dataset:
            generate_answers_for_chair_h3(model, processor, chair_dataset, args.chair_output, h3_results, coco)
        else:
            print("No valid CHAIR images found.")

if __name__ == "__main__":
    main()
