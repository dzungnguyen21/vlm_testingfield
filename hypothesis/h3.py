import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
import os
import re

def extract_hidden_states_before_object(
    model,
    processor,
    image: Image.Image,
    object_word: str,
    generated_ids: list,
    prompt: str = "USER: <image>\nDescribe this image in detail.\nASSISTANT:",
    layers_to_probe: list = None,

):
    """
    Run a forward pass and extract hidden states at the token position
    JUST BEFORE the object word is first completed in generation.

    Returns dict: {layer_idx: hidden_vector (1D numpy array)} or None if
    object word not found in generated text.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    # Find position where object word first completes
    target_pos = None
    for i in range(len(generated_ids)):
        prefix = processor.tokenizer.decode(
            generated_ids[:i+1], skip_special_tokens=True
        ).lower()
        prev = processor.tokenizer.decode(
            generated_ids[:i], skip_special_tokens=True
        ).lower()
        if object_word.lower() in prefix and object_word.lower() not in prev:
            target_pos = i
            break

    if target_pos is None:
        return None

    # Build full input including generated tokens up to target position
    full_ids = (
        inputs["input_ids"][0].tolist()
        + generated_ids[:target_pos]   # tokens BEFORE the object word
    )
    full_ids_tensor = torch.tensor([full_ids], device=model.device)

    n_layers = len(model.model.language_model.layers)
    if layers_to_probe is None:
        # Probe every 4th layer across early-to-middle range
        layers_to_probe = list(range(0, n_layers // 2, 4)) + \
                          list(range(n_layers // 2, int(n_layers * 0.75), 2))

    with torch.inference_mode():
        outputs = model.model.language_model(
            input_ids=full_ids_tensor,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract hidden state at the LAST position (token before object word)
    # for each probed layer
    hidden_by_layer = {}
    for layer_idx in layers_to_probe:
        # outputs.hidden_states[0] = embedding layer
        # outputs.hidden_states[i+1] = after layer i
        h = outputs.hidden_states[layer_idx + 1]  # (1, seq, hidden)
        vec = h[0, -1, :].float().cpu().numpy()   # last position
        hidden_by_layer[layer_idx] = vec

    return hidden_by_layer, target_pos


def get_hallucinated_from_generation(generated_text, image_id, coco):
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns    = coco.loadAnns(ann_ids)
    cat_ids = list(set(a["category_id"] for a in anns))
    true_categories = set(
        coco.loadCats(cat_ids)[i]["name"].lower()
        for i in range(len(cat_ids))
    )
    all_cats   = set(c["name"].lower() for c in coco.loadCats(coco.getCatIds()))
    text_lower = generated_text.lower()

    # use word boundary match — more accurate than plain substring
    def mentioned(word):
        return bool(re.search(r'\b' + re.escape(word) + r'\b', text_lower))

    grounded_found     = [c for c in all_cats if mentioned(c) and c in true_categories]
    hallucinated_found = [c for c in all_cats if mentioned(c) and c not in true_categories]
    return grounded_found, hallucinated_found


def collect_existence_probe_data(
    model,
    processor,
    annotations: dict,
    image_dir: str = "coco/images/val2014",
    max_images: int = 500,
    layers_to_probe: list = None,
):
    n_layers = len(model.model.language_model.layers)
    if layers_to_probe is None:
        layers_to_probe = list(range(0, n_layers // 2, 4)) + \
                          list(range(n_layers // 2, int(n_layers * 0.75), 2))

    data_by_layer = {l: {"X": [], "y": []} for l in layers_to_probe}
    skipped = 0
    n_grounded_total = 0
    n_hall_total = 0

    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    with tqdm(list(annotations.values())[:max_images], desc="Collecting H3 data") as pbar:
        for ann in pbar:
            if not ann.get("generated_caption"):
                continue

            grounded_words     = ann.get("grounded_words", [])
            hallucinated_words = ann.get("hallucinated_words", [])

            if not grounded_words and not hallucinated_words:
                continue

            file_name = ann["file_name"]
            try:
                image = Image.open(
                    os.path.join(image_dir, file_name)
                ).convert("RGB")
            except Exception:
                continue

            # ── Single generation per image ──────────────────────────
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=80, do_sample=False
                )

            prompt_len    = inputs["input_ids"].shape[1]
            generated_ids = generated[0][prompt_len:].tolist()
            generated_text = processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).lower()

            # ── Re-detect grounded/hallucinated from THIS generation ──
            # Don't rely on stored words — re-check against what was
            # actually generated this time
            this_grounded, this_hallucinated = get_hallucinated_from_generation(
                generated_text, ann["image_id"], coco
            )

            # ── Collect hidden states for each object word ────────────
            for obj, label in (
                [(w, 1) for w in this_grounded] +
                [(w, 0) for w in this_hallucinated]
            ):
                result = extract_hidden_states_before_object(
                    model, processor, image, obj,
                    generated_ids, prompt, layers_to_probe
                )

                if result is None:
                    skipped += 1
                    continue

                hidden_by_layer, target_pos = result

                for layer_idx, vec in hidden_by_layer.items():
                    data_by_layer[layer_idx]["X"].append(vec)
                    data_by_layer[layer_idx]["y"].append(label)

                if label == 1:
                    n_grounded_total += 1
                else:
                    n_hall_total += 1

            pbar.set_postfix(
                grounded=n_grounded_total,
                hallucinated=n_hall_total,
                skipped=skipped
            )

    return data_by_layer, layers_to_probe

def run_existence_probe(data_by_layer: dict, layers_to_probe: list):
    """
    Train a logistic regression probe at each layer.
    Uses 5-fold cross-validation to get unbiased AUC.

    AUC > 0.5 means the layer contains a linearly separable
    existence direction.
    AUC near 1.0 means the direction is very clear.
    AUC near 0.5 means no linear signal.
    """
    results = {}

    print("\n  Probing layers for existence direction:")
    print("  " + "-" * 55)
    print(f"  {'Layer':<8} {'n_samples':<12} {'AUC (CV)':<12} {'Interpretation'}")
    print("  " + "-" * 55)

    for layer_idx in layers_to_probe:
        X = np.array(data_by_layer[layer_idx]["X"])
        y = np.array(data_by_layer[layer_idx]["y"])

        if len(X) < 10 or len(np.unique(y)) < 2:
            print(f"  {layer_idx:<8} {'insufficient data'}")
            continue

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 5-fold stratified CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for train_idx, val_idx in cv.split(X_scaled, y):
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_scaled[train_idx], y[train_idx])
            probs = clf.predict_proba(X_scaled[val_idx])[:, 1]
            aucs.append(roc_auc_score(y[val_idx], probs))

        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)

        if mean_auc > 0.70:
            interp = "STRONG direction"
        elif mean_auc > 0.60:
            interp = "moderate direction"
        elif mean_auc > 0.55:
            interp = "weak direction"
        else:
            interp = "no direction"

        print(f"  {layer_idx:<8} {len(X):<12} {mean_auc:.3f}±{std_auc:.3f}  {interp}")

        results[layer_idx] = {
            "auc": mean_auc,
            "auc_std": std_auc,
            "n_samples": len(X),
            "n_grounded": int(y.sum()),
            "n_hallucinated": int((1 - y).sum()),
        }

    return results


def extract_existence_direction(data_by_layer: dict, best_layer: int):
    """
    Fit a final logistic regression on all data at the best layer
    and extract the direction vector.
    This vector points from "not present" toward "present".
    """
    X = np.array(data_by_layer[best_layer]["X"])
    y = np.array(data_by_layer[best_layer]["y"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_scaled, y)

    # Direction vector in original space
    direction = clf.coef_[0]  # (hidden_dim,)

    # Project all samples onto this direction
    projections = X_scaled @ direction
    proj_grounded     = projections[y == 1]
    proj_hallucinated = projections[y == 0]

    return direction, scaler, clf, proj_grounded, proj_hallucinated

# ── Helper: evaluate one set of captions ──────────────────────────────
def evaluate_captions(caption_dict: dict, annotations: dict, coco) -> dict:
    """
    caption_dict: {key: caption_string}
    Returns a deepcopy of annotations with grounded/hallucinated filled in,
    ready for compute_chair_score.
    """
    ann_copy = deepcopy(annotations)
    for key, caption in caption_dict.items():
        if key not in ann_copy:
            continue
        cap_lower = caption.lower()
        g, h = get_hallucinated_from_generation(
            cap_lower, ann_copy[key]["image_id"], coco
        )
        ann_copy[key]["generated_caption"]  = cap_lower
        ann_copy[key]["grounded_words"]      = g
        ann_copy[key]["hallucinated_words"]  = h
    return ann_copy

def plot_h3_results(probe_results: dict, proj_grounded, proj_hallucinated, best_layer: int):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Plot 1: AUC per layer
    ax = axes[0]
    layers = sorted(probe_results.keys())
    aucs   = [probe_results[l]["auc"]     for l in layers]
    stds   = [probe_results[l]["auc_std"] for l in layers]

    ax.errorbar(layers, aucs, yerr=stds,
                fmt="o-", color="#534AB7", linewidth=1.5,
                markersize=5, capsize=3)
    ax.axhline(0.5,  color="#888780", linewidth=0.8,
               linestyle="--", label="chance (0.5)")
    ax.axhline(0.7,  color="#1D9E75", linewidth=0.8,
               linestyle="--", label="strong (0.7)")
    ax.axvline(best_layer, color="#D85A30", linewidth=1,
               linestyle=":", label=f"best layer ({best_layer})")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("AUC (5-fold CV)")
    ax.set_title("Existence direction strength per layer")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 1.0)

    # Plot 2: projection distributions
    ax2 = axes[1]
    ax2.hist(proj_grounded,     bins=30, alpha=0.6,
             color="#1D9E75", label=f"grounded (n={len(proj_grounded)})",
             density=True)
    ax2.hist(proj_hallucinated, bins=30, alpha=0.6,
             color="#D85A30", label=f"hallucinated (n={len(proj_hallucinated)})",
             density=True)
    ax2.axvline(0, color="#888780", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Projection onto existence direction")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Hidden state separation at layer {best_layer}")
    ax2.legend(fontsize=8)

    plt.suptitle(
        "H3: Does a linear existence direction appear in early layers?",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("exp_h3_existence_direction.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: exp_h3_existence_direction.png")


def run_h3(model, processor, annotations, image_dir="coco/images/val2014",
           max_images=300):
    print("=" * 60)
    print("EXPERIMENT H3: Existence Direction Probe")
    print("=" * 60)

    # Step 1: collect hidden states
    data_by_layer, layers_to_probe = collect_existence_probe_data(
        model, processor, annotations,
        image_dir=image_dir,
        max_images=max_images,
    )

    total_samples = len(data_by_layer[layers_to_probe[0]]["X"])
    print(f"\n  Total samples collected: {total_samples}")
    if total_samples < 20:
        print("  Too few samples — increase max_images")
        return None

    # Step 2: probe each layer
    probe_results = run_existence_probe(data_by_layer, layers_to_probe)

    if not probe_results:
        print("  No valid probe results")
        return None

    # Step 3: find best layer
    best_layer = max(probe_results, key=lambda l: probe_results[l]["auc"])
    best_auc   = probe_results[best_layer]["auc"]
    print(f"\n  Best layer: {best_layer}  AUC={best_auc:.3f}")

    if best_auc > 0.60:
        print("  SUPPORTS H3 — existence direction found")
    else:
        print("  NOT significant — no clear existence direction")

    # Step 4: extract direction and plot
    direction, scaler, clf, proj_g, proj_h = extract_existence_direction(
        data_by_layer, best_layer
    )

    plot_h3_results(probe_results, proj_g, proj_h, best_layer)

    # Store the full results object (including non-serializable components) for later use
    h3 = {
        "probe_results"  : probe_results,
        "best_layer"     : best_layer,
        "best_auc"       : best_auc,
        "direction"      : direction,
        "scaler"         : scaler,
        "clf"            : clf,
        "data_by_layer"  : data_by_layer,
        "layers_to_probe": layers_to_probe,
    }


    return h3