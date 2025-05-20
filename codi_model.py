from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class CODIModel(nn.Module):
    def __init__(self, config_path: str | Path = "configs/config.yaml"):
        super().__init__()

        self.config = OmegaConf.load(config_path).model
        self.model_name = self.config.model_name_or_path
        self.max_length = self.config.max_length
        self.cot_length = self.config.cot_length

        self.init_tokenizer_and_model()
        self.init_projections()
        self.get_cot_vectors()

    def init_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<bot>", "<eot>"]}
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
        if self.config.use_lora:
            lora_config = LoraConfig(**self.config.lora)
            self.llm = get_peft_model(self.llm, lora_config)
        self.llm = self.llm.to("cuda")

    def init_projections(self):
        self.proj = nn.Sequential(
            nn.Linear(
                self.llm.config.hidden_size,
                self.llm.config.hidden_size,
                dtype=torch.bfloat16,
            ),
            nn.GELU(),
            nn.Linear(
                self.llm.config.hidden_size,
                self.llm.config.hidden_size,
                dtype=torch.bfloat16,
            ),
            nn.LayerNorm(self.llm.config.hidden_size, dtype=torch.bfloat16),
        ).to("cuda")

    def get_cot_vectors(self):
        bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")

        embedding_layer = self.llm.get_input_embeddings()

        self.bot_embedding = embedding_layer.weight[bot_token_id]
        self.eot_embedding = embedding_layer.weight[eot_token_id]

    def run_cot_loop(self, input_embeds: torch.Tensor) -> torch.Tensor:
        past_key_values = None
        for i in range(self.cot_length):
            # No causal masking, as the model is already causal
            student_outputs = self.llm(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            last_output_vectors = self.proj(
                student_outputs.hidden_states[-1][:, -1].unsqueeze(dim=1)
            )
            input_embeds = torch.cat([input_embeds, last_output_vectors], dim=1)
            past_key_values = student_outputs.past_key_values

        del last_output_vectors, past_key_values, student_outputs
        return input_embeds

    def think(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Takes question, returns inputs_embeds of question + <bot> + <CoT> + <eot>
        """
        input_ids = inputs["question_ids"].to("cuda")
        expanded_eot_emb = (
            self.eot_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(input_ids.size(0), 1, -1)
        )
        expanded_bot_emb = (
            self.bot_embedding.unsqueeze(0)
            .unsqueeze(0)
            .expand(input_ids.size(0), 1, -1)
        )
        # TODO would be here <bot>?
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        input_embeds = torch.cat([input_embeds, expanded_bot_emb], dim=1)
        input_embeds = self.run_cot_loop(input_embeds)
        input_embeds = torch.cat([input_embeds, expanded_eot_emb], dim=1)

        return input_embeds


if __name__ == "__main__":
    model = CODIModel()
    inputs = {"question_ids": torch.tensor(2 * [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])}
    pre_inputs = model.think(inputs)
