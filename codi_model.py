import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from omegaconf import OmegaConf


class CODIModel(nn.Module):
    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        super().__init__()

        self.config = OmegaConf.load(config_path).model
        self.model_name = self.config.model_name_or_path
        self.max_length = self.config.max_length
        self.cot_length = self.config.cot_length

        self.init_tokenizer_and_model()
        self.init_projections()

    def init_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bot>", "<eot>"]}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
        if self.config.use_lora:
            lora_config = LoraConfig(**self.config.lora)
            self.model = get_peft_model(self.model, lora_config).to("cuda")

    def init_projections(self):
        self.proj = nn.Sequential(
            nn.Linear(
                self.model.config.hidden_size,
                self.model.config.hidden_size,
                dtype=torch.bfloat16,
            ),
            nn.GELU(),
            nn.Linear(
                self.model.config.hidden_size,
                self.model.config.hidden_size,
                dtype=torch.bfloat16,
            ),
            nn.LayerNorm(self.model.config.hidden_size, dtype=torch.bfloat16),
        ).to("cuda")

    def forward(self, inputs):
        pass

if __name__ == "__main__":
    model = CODIModel()
