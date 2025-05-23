from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from codi_model import CODIModel


class CODIModelFreeze(CODIModel):
    def __init__(
        self,
        config: DictConfig | None = None,
        config_path: str | Path = "configs/config.yaml",
    ):
        super().__init__(config, config_path)
        checkpoint_path = self.config.teacher_path
        self.init_teacher(checkpoint_path)
        pass

    def init_teacher(self, checkpoint_path: str | Path | None = None):
        self.llm_teacher = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")[
                "model_state_dict"
            ]
            dict_keys = list(state_dict.keys())
            for key in dict_keys:
                if key.startswith("llm."):
                    state_dict[key[4:]] = state_dict.pop(key)
            self.llm_teacher.load_state_dict(state_dict)

        self.llm_teacher = self.llm_teacher.to(self.device)
        self.llm_teacher.eval()
        # Freeze all parameters of the teacher model
        for param in self.llm_teacher.parameters():
            param.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]):
        input_ids = batch["teacher_full_input_ids"]
        attention_mask = batch["teacher_full_attention_mask"]
        loss_mask = batch["teacher_full_loss_mask"]
        labels = input_ids.masked_fill(loss_mask == 0, -100)
        with torch.no_grad():
            teacher_outputs = self.llm_teacher.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
        student_outputs = self.forward_student(batch)

        distill_loss = self.calc_distil_loss(batch, teacher_outputs, student_outputs)
        total_loss = self.beta * student_outputs.loss + self.gamma * distill_loss

        output = CausalLMOutputWithCrossAttentions(
            loss=total_loss,
            logits=student_outputs.logits,
            # Set other required parameters to None
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
        output.num_tokens = student_outputs["num_tokens"]
        output.teacher_loss = teacher_outputs.loss
        output.student_loss = student_outputs.loss
        output.distill_loss = distill_loss

        return output


if __name__ == "__main__":
    model = CODIModel()
    inputs = {
        "question_ids": torch.tensor(2 * [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
        "answer_ids": torch.tensor(2 * [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
    }
    pre_inputs = model.forward(inputs)
