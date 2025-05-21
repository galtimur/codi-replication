from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.nn.functional import smooth_l1_loss
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class BaseModel(nn.Module):
    def __init__(
        self,
        config: DictConfig | None = None,
        config_path: str | Path = "configs/config.yaml",
    ):
        super().__init__()

        if config is not None:
            self.config = OmegaConf.load(config_path).model
        else:
            self.config = config
        self.model_name = self.config.model_name_or_path
        self.max_length = self.config.max_length
        self.device = self.config.device

        self.init_tokenizer_and_model()

    def init_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        if self.config.use_lora:
            lora_config = LoraConfig(**self.config.lora)
            self.llm = get_peft_model(self.llm, lora_config)
        self.llm = self.llm.to(self.device)

    def forward(self, batch, *args, **kwargs):
        input_ids = batch["teacher_full_input_ids"]
        attention_mask = batch["teacher_full_attention_mask"]
        loss_mask = batch["teacher_full_loss_mask"]
        labels = input_ids.masked_fill(loss_mask == 0, -100)
        outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            *args,
            **kwargs
        )
        outputs["num_tokens"] = torch.sum(attention_mask)
        return outputs

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


class CODIModel(nn.Module):
    def __init__(
        self,
        config: DictConfig | None = None,
        config_path: str | Path = "configs/config.yaml",
    ):
        super().__init__()

        if config is not None:
            self.config = OmegaConf.load(config_path).model
        else:
            self.config = config
        self.device = "cuda"
        self.model_name = self.config.model_name_or_path
        self.max_length = self.config.max_length
        self.cot_length = self.config.cot_length
        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.gamma = self.config.gamma

        self.init_tokenizer_and_model()
        self.init_projections()
        self.get_eot_vector()

    def init_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.config.bot_token,
                    self.config.eot_token,
                ]
            }
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)
        if self.config.use_lora:
            lora_config = LoraConfig(**self.config.lora)
            self.llm = get_peft_model(self.llm, lora_config)
        self.llm = self.llm.to(self.device)

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
        ).to(self.device)

    def get_eot_vector(self):
        eot_token_id = self.tokenizer.convert_tokens_to_ids(self.config.eot_token)
        embedding_layer = self.llm.get_input_embeddings()
        self.eot_embedding = embedding_layer.weight[eot_token_id]

    def run_cot_loop(
        self, quest_embeds: torch.Tensor, q_attn_mask: torch.Tensor
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        past_key_values = None
        current_attention_mask = q_attn_mask
        for i in range(self.cot_length):
            if i > 0:
                new_token_mask = torch.ones(
                    current_attention_mask.size(0),
                    1,
                    device=current_attention_mask.device,
                )
                current_attention_mask = torch.cat(
                    [current_attention_mask, new_token_mask], dim=1
                )
            # No causal masking, as the model is already causal
            student_outputs = self.llm(
                inputs_embeds=quest_embeds,
                output_hidden_states=True,
                past_key_values=past_key_values,
                atttention_mask=current_attention_mask,
            )
            last_output_vectors = self.proj(
                student_outputs.hidden_states[-1][:, -1].unsqueeze(dim=1)
            )
            quest_embeds = last_output_vectors
            past_key_values = student_outputs.past_key_values

        del last_output_vectors, student_outputs, quest_embeds
        return past_key_values, current_attention_mask

    def calc_distil_loss(
        self,
        batch: dict[str, torch.Tensor],
        teacher_outputs: dict[str, torch.Tensor],
        student_outputs: dict[str, torch.Tensor],
    ):
        semicolon_pos = batch["semicolon_position_from_end_answer"]
        batch_indices = torch.arange(student_outputs.logits.size(0), device=self.device)
        actual_positions_stud = student_outputs.logits.size(1) - semicolon_pos
        actual_positions_teach = teacher_outputs.logits.size(1) - semicolon_pos
        distill_loss = torch.tensor(0.0).to(self.device)
        for layer_stud, layer_teach in zip(
            student_outputs.hidden_states, teacher_outputs.hidden_states
        ):
            semicolon_stud_hid = layer_stud[batch_indices, actual_positions_stud]
            semicolon_teach_hid = layer_teach[batch_indices, actual_positions_teach]
            distill_loss += (
                smooth_l1_loss(semicolon_stud_hid, semicolon_teach_hid)
                / semicolon_teach_hid.std()
            )
        return distill_loss

    def forward_student(self, batch: dict[str, torch.Tensor]):
        """
        Takes question and answer embeddings and runs the chain-of-thought reasoning.
        """

        question_ids = batch["question_input_ids"].to("cuda")
        answer_ids = batch["answer_input_ids"].to("cuda")
        q_attn_mask = batch["question_attention_mask"].to("cuda")
        a_attn_mask = batch["answer_attention_mask"].to("cuda")
        question_embeds = self.llm.get_input_embeddings()(question_ids)
        answer_embed = self.llm.get_input_embeddings()(answer_ids)

        past_key_values, attn_msk_q_cot = self.run_cot_loop(
            question_embeds, q_attn_mask
        )
        attn_msk_q_cot_a = torch.cat([attn_msk_q_cot, a_attn_mask], dim=1)
        labels = answer_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        answer_result = self.llm(
            inputs_embeds=answer_embed,
            labels=labels,
            past_key_values=past_key_values,
            attention_mask=attn_msk_q_cot_a,
            output_hidden_states=True,
        )
        answer_result["num_tokens"] = torch.sum(batch["teacher_full_attention_mask"])

        return answer_result

    def forward(self, batch: dict[str, torch.Tensor]):
        input_ids = batch["teacher_full_input_ids"]
        attention_mask = batch["teacher_full_attention_mask"]
        loss_mask = batch["teacher_full_loss_mask"]
        labels = input_ids.masked_fill(loss_mask == 0, -100)
        teacher_outputs = self.llm.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        student_outputs = self.forward_student(batch)

        distill_loss = self.calc_distil_loss(batch, teacher_outputs, student_outputs)
        total_loss = (
            self.alpha * teacher_outputs.loss
            + self.beta * student_outputs.loss
            + self.gamma * distill_loss
        )

        output = CausalLMOutputWithCrossAttentions(
            loss=total_loss,
            logits=student_outputs.logits,
            # Set other required parameters to None
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None
        )
        output.num_tokens = student_outputs["num_tokens"]
        output.teacher_loss = teacher_outputs.loss
        output.student_loss = student_outputs.loss
        output.distill_loss = distill_loss

        return output

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
        q_attn_mask = (question_ids != self.tokenizer.pad_token_id).int()
        batch_size = question_ids.size(0)

        question_embeds = self.llm.get_input_embeddings()(question_ids)
        past_key_values, q_cot_mask = self.run_cot_loop(question_embeds, q_attn_mask)
        expanded_eot_emb = (
            self.eot_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        )

        generated_token_ids = []
        current_embed = expanded_eot_emb
        current_mask = q_cot_mask

        # Generation loop
        for i in range(max_length):
            with torch.no_grad():
                outputs = self.llm(
                    inputs_embeds=current_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                    attention_mask=current_mask,
                )
                new_token_mask = torch.ones(
                    current_attention_mask.size(0),
                    1,
                    device=current_attention_mask.device,
                )
                current_attention_mask = torch.cat(
                    [current_attention_mask, new_token_mask], dim=1
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = next_token_logits / temperature
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            # Get next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Add to generated tokens
            generated_token_ids.append(next_token)

            # Check for EOS token
            if (next_token == self.tokenizer.eos_token_id).any():
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
