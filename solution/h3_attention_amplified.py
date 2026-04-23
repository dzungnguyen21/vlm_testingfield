import re
import torch
from transformers import LogitsProcessor
import numpy as np

def get_image_positions(input_ids, image_token_id=32000):
    return torch.where(input_ids[0] == image_token_id)[0]

def intervene_h3_attention_amplified(
    model,
    processor,
    image,
    h3_results,
    coco=None,
    absence_threshold=-2.0,
    amplification_strength=2.0,    # how much to amplify visual attention
    suppression_strength=3.0,      # lighter logit suppression as backup
    localization_heads=None,       # heads [28, 27, 31, 3, 5] from H2
    middle_layer_range=(0.25, 0.75),
    prompt="USER: <image>\nDescribe this image in detail.\nASSISTANT:",
    max_new_tokens=100,
):
    """
    H3 + Attention Intervention combined method.

    Pass 1: dry-run to get per-object existence projections
    Pass 2: regenerate with two simultaneous interventions:
      A) Attention amplification (Devils paper): boost image token attention
         in middle layers, weighted by H3 existence score
      B) Logit suppression (H3 surgical): light penalty on absent-predicted
         object tokens as a secondary guard
    """
    from transformers import LogitsProcessor

    best_layer = h3_results["best_layer"]
    scaler     = h3_results["scaler"]
    direction  = h3_results["direction"]
    if coco is not None:
        all_cats = set(c["name"].lower() for c in coco.loadCats(coco.getCatIds()))
    else:
        all_cats = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'}


    n_layers  = len(model.model.language_model.layers)
    mid_start = int(n_layers * middle_layer_range[0])
    mid_end   = int(n_layers * middle_layer_range[1])

    if localization_heads is None:
        localization_heads = [28, 27, 31, 3, 5]   # from H2 analysis

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    image_positions = get_image_positions(inputs["input_ids"])
    img_start = image_positions[0].item()
    img_end   = image_positions[-1].item() + 1
    prompt_len = inputs["input_ids"].shape[1]

    # ── Pass 1: dry-run + per-object existence probe ──────────────────
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    generated_ids  = generated[0][prompt_len:].tolist()
    generated_text = processor.tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).lower()

    mentioned_objects = [
        cat for cat in all_cats
        if re.search(r'\b' + re.escape(cat) + r'\b', generated_text)
    ]
    if not mentioned_objects:
        return generated_text, {}, set()

    # Per-object probing
    object_projections = {}
    for obj in mentioned_objects:
        target_pos = None
        for i in range(len(generated_ids)):
            prefix = processor.tokenizer.decode(
                generated_ids[:i+1], skip_special_tokens=True
            ).lower()
            prev = processor.tokenizer.decode(
                generated_ids[:i], skip_special_tokens=True
            ).lower()
            if obj in prefix and obj not in prev:
                target_pos = i
                break
        if target_pos is None:
            continue

        context_tensor = torch.tensor(
            [inputs["input_ids"][0].tolist() + generated_ids[:target_pos]],
            device=model.device
        )
        with torch.inference_mode():
            out = model.model.language_model(
                input_ids=context_tensor,
                output_hidden_states=True,
                return_dict=True,
            )
        h     = out.hidden_states[best_layer + 1][0, -1, :].float().cpu().numpy()
        h_std = (h - scaler.mean_) / scaler.scale_
        object_projections[obj] = float(h_std @ direction)

    absent_objects = {
        obj for obj, proj in object_projections.items()
        if proj < absence_threshold
    }

    # Mean existence score across all mentioned objects
    # Positive = scene has strong visual grounding overall
    # Negative = model is relying heavily on language prior
    mean_projection = float(np.mean(list(object_projections.values()))) \
                      if object_projections else 0.0

    # ── Pass 2: attention amplification + light logit suppression ─────

    # Build logit suppress set (lighter than v1 — attention intervention
    # does the heavy lifting)
    suppress_ids = {
        tid
        for obj in absent_objects
        for variant in [obj, " " + obj]
        for tid in processor.tokenizer.encode(variant, add_special_tokens=False)
        if processor.tokenizer.decode([tid]).strip()
    }

    # Attention intervention hook (inspired by Devils in Middle Layers)
    # At each decode step in middle layers:
    #   1. Identify localization heads (focused, low entropy)
    #   2. Average their attention maps → consensus visual grounding signal
    #   3. Amplify attention to image tokens proportionally
    #   4. Scale amplification by mean_projection (H3 signal):
    #      if mean_projection is very negative → model is drifting from image
    #      → amplify more aggressively

    model.model.language_model.config.output_attentions = True
    _current_step = [0]

    def make_attention_hook(layer_idx):
        def hook_fn(module, input, output):
            if output[1] is None:
                return output

            attn_weights = output[1]   # (batch, heads, tgt, src)
            tgt_len = attn_weights.shape[2]

            # Only intervene on decode steps (tgt_len == 1)
            if tgt_len != 1:
                return output

            attn = attn_weights.clone()

            # Step 1: extract localization head attention to image
            loc_attn_maps = []
            for h_idx in localization_heads:
                if h_idx < attn.shape[1]:
                    src_len = attn.shape[-1]
                    if img_end <= src_len:
                        img_attn = attn[0, h_idx, 0, img_start:img_end]
                        loc_attn_maps.append(img_attn)

            if not loc_attn_maps:
                return output

            # Step 2: consensus map = mean across localization heads
            consensus = torch.stack(loc_attn_maps).mean(dim=0)  # (n_img_tokens,)

            # Step 3: scale amplification by H3 existence signal
            # More negative projection → model drifting from image → amplify more
            h3_scale = 1.0 + max(0.0, -mean_projection) * 0.5
            effective_strength = amplification_strength * h3_scale

            # Step 4: amplify all heads' attention to image tokens
            # using the consensus map as a spatial guide
            src_len = attn.shape[-1]
            if img_end <= src_len:
                # Normalize consensus to use as multiplicative weight
                consensus_norm = consensus / (consensus.sum() + 1e-9)

                for h_idx in range(attn.shape[1]):
                    img_slice = attn[0, h_idx, 0, img_start:img_end]
                    # Amplify toward consensus direction
                    amplified = img_slice + effective_strength * consensus_norm
                    # Renormalize the full attention row to sum to 1
                    attn[0, h_idx, 0, img_start:img_end] = amplified
                    row_sum = attn[0, h_idx, 0, :].sum()
                    attn[0, h_idx, 0, :] = attn[0, h_idx, 0, :] / (row_sum + 1e-9)

            return (output[0], attn) + output[2:]
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.model.language_model.layers):
        if mid_start <= i < mid_end:
            hooks.append(layer.self_attn.register_forward_hook(
                make_attention_hook(i)
            ))

    class LightSuppressor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            for tid in suppress_ids:
                if tid < scores.shape[-1]:
                    scores[:, tid] -= suppression_strength
            return scores

    logits_processors = [LightSuppressor()] if suppress_ids else []

    try:
        with torch.inference_mode():
            generated_int = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                logits_processor=logits_processors,
            )
    finally:
        model.model.language_model.config.output_attentions = False
        for h in hooks:
            h.remove()

    generated_text_int = processor.tokenizer.decode(
        generated_int[0][prompt_len:].tolist(), skip_special_tokens=True
    )
    return generated_text_int, object_projections, absent_objects