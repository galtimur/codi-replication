import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


class CODIModel(nn.Module):
    def __init__(
        self, model_path, icot_length=6, alpha=1, beta=1, gamma=1, max_length=256
    ):
        self.max_length = max_length
        self.icot_length = icot_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bot>", "<eot>"]}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
