import gc
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

if "model" in globals():
    del model  # noqa: F821
gc.collect()
torch.cuda.empty_cache()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id)