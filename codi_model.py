from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def get_eot_vectors(self):

        eot_token_id = self.tokenizer.convert_tokens_to_ids("<eot>")
        embedding_layer = self.llm.get_input_embeddings()
        self.eot_embedding = embedding_layer.weight[eot_token_id]

    def run_cot_loop(
        self, quest_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_key_values = None
        for i in range(self.cot_length):
            # No causal masking, as the model is already causal
            student_outputs = self.llm(
                inputs_embeds=quest_embeds,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            last_output_vectors = self.proj(
                student_outputs.hidden_states[-1][:, -1].unsqueeze(dim=1)
            )
            quest_embeds = last_output_vectors
            past_key_values = student_outputs.past_key_values

        del last_output_vectors, student_outputs, quest_embeds
        return past_key_values

    def forward(self, inputs: dict[str, torch.Tensor]):
        question_ids = inputs["question_ids"].to("cuda")
        answer_ids = inputs["answer_ids"].to("cuda")
        question_embeds = self.llm.get_input_embeddings()(question_ids)
        answer_embed = self.llm.get_input_embeddings()(answer_ids)

        past_key_values = self.run_cot_loop(question_embeds)
        answer_result = self.llm(
            inputs_embeds=answer_embed, past_key_values=past_key_values
        )

        return answer_result

    def generate(
            self,
            question_ids: torch.Tensor,
            max_length: int = 50,
            temperature: float = 0.2,
            do_sample: bool = True,
            top_p: float = 0.9,
            top_k: int = 50,
    ):
        """
        Generate answers for given questions, using the model's chain-of-thought reasoning.

        Args:
            question_ids: Tensor of shape [batch_size, seq_len] containing tokenized questions
            max_length: Maximum number of tokens to generate for the answer
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to sample from distribution (True) or use greedy decoding (False)
            top_p: Nucleus sampling parameter (keep tokens with cumulative probability >= top_p)
            top_k: Number of highest probability tokens to keep for top-k sampling

        Returns:
            Generated answer token IDs
        """

        question_ids = question_ids.to(self.llm.device)
        batch_size = question_ids.size(0)

        question_embeds = self.llm.get_input_embeddings()(question_ids)
        past_key_values = self.run_cot_loop(question_embeds)
        expanded_eot_emb = self.eot_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)

        generated_token_ids = []
        current_embed = expanded_eot_emb

        # Generation loop
        for i in range(max_length):
            with torch.no_grad():
                outputs = self.llm(
                    inputs_embeds=current_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = next_token_logits / temperature
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')

            # Get next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Add to generated tokens
            generated_token_ids.append(next_token)

            # Check for EOS token
            if (next_token == self.tokenizer.eos_token_id):
                break

            # Update past key values for efficiency
            past_key_values = outputs.past_key_values

            # Get embedding for the next token
            current_embed = self.llm.get_input_embeddings()(next_token)


        # Concatenate all tokens
        answer_ids = torch.cat(generated_token_ids, dim=1)

        return answer_ids


if __name__ == "__main__":
    model = CODIModel()
    inputs = {
        "question_ids": torch.tensor(2 * [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
        "answer_ids": torch.tensor(2 * [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
    }
    pre_inputs = model.forward(inputs)
