import json
import logging
from typing import List, Dict, Union
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import argparse
import os

logger = logging.getLogger(__name__)

# Try to download NLTK data. In production, this should be handled once during setup.
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK datasets for CHAIR metric...")
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')

class CHAIREvaluator:
    """
    Evaluator for Captioning Hallucination Assessment with Image Relevance (CHAIR).
    Measures CHAIRs (per sentence) and CHAIRi (per image) metrics.
    Requires MS-COCO annotations to evaluate against ground truth.
    """
    def __init__(self, coco_annotation_file: str = None, synonyms_file: str = None):
        self.lemmatizer = WordNetLemmatizer()
        self.image_to_objects = {}  # image_id -> set of COCO object classes
        self.synonyms_dict = {}     # word -> COCO main class
        
        # Load default synonyms mapping (usually mappings from custom words to 80 COCO categories)
        if synonyms_file:
            self._load_synonyms(synonyms_file)
        else:
            self._build_default_synonyms()
            
        if coco_annotation_file and os.path.exists(coco_annotation_file):
            self._load_coco_annotations(coco_annotation_file)
        else:
            logger.warning("No valid COCO annotation file provided or found. GT objects won't be loaded.")
            
    def _build_default_synonyms(self):
        """
        In standard CHAIR eval, a mapping from synonym to COCO 80 classes is used.
        Here we define a minimal fallback. 
        For robust evaluation, please provide the standard MSCOCO synonyms.txt 
        used in the original CHAIR paper.
        """
        self.synonyms_dict = {
            "person": "person", "man": "person", "woman": "person", "child": "person", "boy": "person", "girl": "person",
            "car": "car", "auto": "car", "automobile": "car",
            "dog": "dog", "puppy": "dog",
            "cat": "cat", "kitten": "cat",
            # ... add more as required
        }

    def _load_synonyms(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # typical format: word class
                    self.synonyms_dict[parts[0]] = parts[1]

    def _load_coco_annotations(self, path: str):
        logger.info(f"Loading COCO annotations from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
            
        # Build mapping from category_id -> name
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Build ground truth objects per image
        for ann in coco_data['annotations']:
            img_id = str(ann['image_id'])
            # Sometimes image_ids are padded. Clean it to make sure it matches predictions.
            # Convert to int then str to remove leading zeros, standard practice.
            clean_img_id = str(int(img_id))
            cat_name = categories[ann['category_id']]
            
            if clean_img_id not in self.image_to_objects:
                self.image_to_objects[clean_img_id] = set()
            self.image_to_objects[clean_img_id].add(cat_name)

    def extract_objects(self, caption: str) -> List[str]:
        """
        Tokenizes caption, extracts nouns, and returns mapped COCO categories.
        """
        tokens = nltk.word_tokenize(caption.lower())
        tagged = nltk.pos_tag(tokens)
        
        objects_found = []
        for word, tag in tagged:
            # Look for nouns
            if tag.startswith('NN'):
                lemma = self.lemmatizer.lemmatize(word)
                if lemma in self.synonyms_dict:
                    objects_found.append(self.synonyms_dict[lemma])
                    
        return list(set(objects_found)) # unique objects

    def evaluate(self, predictions_dict: Dict[str, str]) -> Dict[str, float]:
        """
        Evaluates predictions against ground truth.
        predictions_dict: Dictionary mapping image_id to generated caption.
        """
        if not self.image_to_objects:
            logger.error("COCO annotations not loaded. Cannot evaluate CHAIR correctly.")
            return {}
            
        total_hallucinated_objects = 0
        total_objects_mentioned = 0
        total_hallucinated_images = 0
        total_images = len(predictions_dict)
        evaluated_images = 0
        
        for img_id, caption in predictions_dict.items():
            # Clean image id string to remove zero-padding if any
            clean_img_id = str(int(img_id)) if img_id.isdigit() else str(img_id)
            
            if clean_img_id not in self.image_to_objects:
                # If ground truth for this image is missing, assume it's empty
                # Only objects hallucinated would be ones found.
                gt_objects = set()
            else:
                gt_objects = self.image_to_objects[clean_img_id]
                
            extracted_objs = self.extract_objects(caption)
            evaluated_images += 1
            
            has_hallucination = False
            for obj in extracted_objs:
                total_objects_mentioned += 1
                if obj not in gt_objects:
                    total_hallucinated_objects += 1
                    has_hallucination = True
                    
            if has_hallucination:
                total_hallucinated_images += 1
                
        chairs = 0.0
        if total_objects_mentioned > 0:
            chairs = (total_hallucinated_objects / total_objects_mentioned) * 100.0
            
        chairi = 0.0
        if evaluated_images > 0:
            chairi = (total_hallucinated_images / evaluated_images) * 100.0
            
        return {
            "CHAIRs": chairs,
            "CHAIRi": chairi,
            "Total Objects Mentioned": total_objects_mentioned,
            "Total Hallucinated Objects": total_hallucinated_objects,
            "Total Images Evaluated": evaluated_images,
        }

def read_predictions(file_path: str) -> Dict[str, str]:
    predictions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    img_id = item.get('image_id', item.get('id', None))
                    ans = item.get('caption', item.get('text', str(item)))
                    if img_id is not None:
                        predictions[str(img_id)] = ans
            elif isinstance(data, dict):
                predictions = {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            # Try reading as JSONL
            f.seek(0)
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                img_id = item.get('image_id', item.get('id', None))
                ans = item.get('caption', item.get('text', str(item)))
                if img_id is not None:
                    predictions[str(img_id)] = ans
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CHAIR metrics")
    parser.add_argument("--input_file", type=str, required=True, help="Path to predictions JSON/JSONL")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSON")
    parser.add_argument("--coco_annotations", type=str, default="", help="Path to instances_val2014.json")
    parser.add_argument("--synonyms", type=str, default="", help="Path to COCO synonyms mapping")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    print(f"Reading predictions from {args.input_file}...")
    predictions = read_predictions(args.input_file)
    print(f"Loaded {len(predictions)} prediction entries.")

    print(f"Initializing CHAIR evaluator...")
    evaluator = CHAIREvaluator(
        coco_annotation_file=args.coco_annotations if args.coco_annotations else None,
        synonyms_file=args.synonyms if args.synonyms else None
    )
    
    print("Evaluating metrics...")
    metrics = evaluator.evaluate(predictions)
    
    if metrics:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Successfully evaluated! Results saved to {args.output_file}")
        print(json.dumps(metrics, indent=4))
    else:
        print("Evaluation failed. Could not compute metrics (possibly no ground truths loaded).")
