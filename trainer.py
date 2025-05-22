"""
File is taken from kotlin-initiative repo
"""

import math
import re
import time
import traceback
import warnings
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Iterable

import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, get_cosine_schedule_with_warmup

import wandb

def extract_number(pred_answer):
    # This pattern matches integers or floating point numbers
    match = re.match(r'^[-+]?(\d+(\.\d*)?|\.\d+)', pred_answer)
    if match:
        return match.group(0)
    return pred_answer


def scale_grads(model: nn.Module, scaler: torch.Tensor) -> None:
    """
    Utility to scale the gradients of a model.
    This is useful for gradient accumulation where we want to normalize
    the gradients by the total number of tokens seen.

    Inputs:
        model (nn.Module): model whose gradients should be scaled
        scaler (torch.Tensor): scaling factor to apply to the gradients

    Outputs:
        None (grad fields are modified in place)
    """
    device = None
    for p in model.parameters():
        # First ensure scaler is on the same device as the model
        if not device:
            device = p.device
            scaler = scaler.to(device)
        if p.grad is not None:
            p.grad *= scaler


@torch.no_grad()
def get_norm(model_parameters: Iterable[torch.Tensor], max_norm: float) -> float:
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model_parameters, max_norm, norm_type=2.0
    )

    return grad_norm.item()


class PytorchTrainer:
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloaders: DataLoader,
        model_type: str,
        local_rank: int = 0,
        global_rank: int = 0,
        world_size: int = 1,
        perform_sanity_check: bool = True,
        is_mixed: bool = False,
        use_grad_scaler: bool = False,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.is_main_process = global_rank == 0
        self.is_distributed = world_size > 1
        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders
        self._device = model.device
        self.model = model
        self.model_type = model_type

        micro_batch_size = self.train_dataloader.batch_size
        if micro_batch_size is None:
            raise ValueError(
                "Batch size of dataloader cannot be None. Please check its configuration."
            )
        self.total_micro_batches = len(self.train_dataloader)

        # Calculate accumulation steps based on (effective) batch size and distributed setup
        self.accum_steps = math.ceil(config.effective_batch_size / micro_batch_size)

        if self.is_distributed:
            if self.accum_steps % world_size != 0:
                warnings.warn(
                    f"Accumulation steps ({self.accum_steps}) not divisible by "
                    f"world size ({world_size}). This may impact training."
                )
            self.accum_steps = math.ceil(self.accum_steps / world_size)

        # Validate batch size configuration
        if config.effective_batch_size % micro_batch_size != 0:
            warnings.warn(
                f"Batch size ({config.effective_batch_size}) not divisible by "
                f"micro-batch size ({micro_batch_size}). Using {self.accum_steps} accumulation steps."
            )

        total_steps = (self.total_micro_batches // self.accum_steps) * config.num_epochs
        warmup_steps = math.ceil(total_steps * config.warmup_ratio)

        self.setup_optimizer_and_scheduler(config, total_steps, warmup_steps)

        self.config = config

        self.val_steps: list[int] = []
        self.setup_validation_and_checkpoint_schedule()

        self.batches_done = 0
        self.micro_batches_done = 0
        self.loss_tot = torch.tensor(0.0, device=self._device)
        self.loss_teach = torch.tensor(0.0, device=self._device)
        self.loss_stud = torch.tensor(0.0, device=self._device)
        self.loss_distil = torch.tensor(0.0, device=self._device)
        self.num_tokens = torch.tensor(0, device=self._device)
        self.total_tokens = 0

        self.seq_len = config.max_length
        # Upper bound for the number of padding tokens
        # This value cancels out later, introduced for grad stability.
        # Not sure that it's really needed
        self.compensation_constant = self.seq_len * micro_batch_size * self.accum_steps

        # Add timing variables for throughput calculation
        self.step_start_time = time.time()
        self.tokens_per_second = 0.0
        self.total_training_time = 0.0
        self.moving_avg_throughput = 0.0
        self.throughput_history = []
        self.throughput_window_size = 20  # For moving average calculation

        if self.config.resume_from is not None and self.config.resume_from != "":
            self.load_checkpoint(self.config.resume_from)

        if perform_sanity_check:
            self.sanity_check()
            self._barrier()

    def setup_optimizer_and_scheduler(
        self, config: DictConfig, total_steps: int, warmup_steps: int
    ):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),  # this is changed from model.llm, be careful?
            **config.optimizer,
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

    def save_checkpoint(self, batches_done: int):
        if self.is_main_process:
            if batches_done % self.config.save_checkpoints_every == 0:
                time_start = time.time()

                checkpoint_dir = Path(self.config.save_checkpoints_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_path = checkpoint_dir / f"checkpoint_{batches_done}.pt"

                checkpoint_dict = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict()
                    if self.scheduler
                    else None,
                    "batches_done": batches_done,
                    "micro_batches_done": self.micro_batches_done,
                    "total_tokens": self.total_tokens,
                    "throughput_history": self.throughput_history,
                    "total_training_time": self.total_training_time,
                }

                torch.save(checkpoint_dict, checkpoint_path)
                print(f"Checkpoint saved in {time.time() - time_start} s.")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.batches_done = checkpoint["batches_done"]
        self.micro_batches_done = checkpoint["micro_batches_done"]
        self.total_tokens = checkpoint["total_tokens"]
        self.throughput_history = checkpoint["throughput_history"]
        self.total_training_time = checkpoint["total_training_time"]

    def setup_validation_and_checkpoint_schedule(self):
        self.val_steps = list(
            range(
                1,
                (len(self.train_dataloader) * self.config.num_epochs)
                // self.accum_steps,
                self.config.validate_every,
            )
        )
        # No need to save and validate the model at the first step.
        # Val is performed at the start of the training and end of each epoch
        self.val_steps = self.val_steps[1:]

        print(
            f"Validation every: {self.config.validate_every} optimizer step, in total: {len(self.val_steps)} validations"
        )

    def sanity_check(self):
        for stage_name, stage in [
            # TODO May be add test on a single batch. Problem - you can not pass num batches into test
            ("saving", partial(self.save_checkpoint, batches_done=0)),
            ("training", self._single_training_step),
        ]:
            try:
                print(f"Running {stage_name} sanity check")
                stage()
                print(f"{stage_name} OK")
            except Exception as e:
                tb = traceback.format_exc()
                raise Exception(
                    f"{stage_name} fails with error: {e}, \nTraceback:\n{tb}"
                ) from e

        print(
            "Sanity check complete, but remember -- real insanity is doing the same thing "
            "over and over and expecting different results"
        )

    def _single_training_step(self):
        micro_batch = next(iter(self.train_dataloader))
        self.training_step(micro_batch)

    def _barrier(self):
        if self.is_distributed:
            dist.barrier()

    @staticmethod
    def _sort_df_by_task_id(df: pd.DataFrame) -> pd.DataFrame:
        sorted_df = df.sort_values(
            by="task_id", key=lambda x: x.apply(lambda x: int(x.split("/")[-1]))
        )
        sorted_df.reset_index(drop=True, inplace=True)
        return sorted_df

    # pure inference_mode is incompatible with FSDP
    # https://github.com/pytorch/pytorch/issues/113203
    # @torch.inference_mode()
    def validation(self) -> Annotated[dict, "to_log"]:
        self.model.eval()
        to_log: dict[str, Any] = {}

        for val_dataloader in tqdm(
            iterable=self.val_dataloaders, disable=not self.is_main_process
        ):
            val_loss_acc = torch.tensor(0.0, device=self._device)
            val_num_tokens = torch.tensor(0, device=self._device)
            match_count = 0
            total_count = 0

            for i, batch in enumerate(val_dataloader):
                # GT: batch["answer_text"]
                # Pred: decoded_text
                # with open("test_results.txt", "a") as f:
                #     f.write(70*"-")

                self.batch_to_device(batch)

                # pure inference_mode is incompatible with FSDP, so I use no_grad()
                with torch.no_grad():
                    outputs = self.model(batch)
                val_loss_acc += outputs.loss
                val_num_tokens += outputs.num_tokens

                input_ids = batch["teacher_full_input_ids"]
                loss_mask = batch["teacher_full_loss_mask"]

                # Calculate exact match for every sample in batch
                decoded_texts = []
                for j in range(len(batch["answer_text"])):
                    logits = outputs.logits[j]
                    # Only keep predictions where loss_mask is 1
                    masked_input_ids = (logits[loss_mask[j] == 1]).argmax(dim=-1)
                    decoded_text = self.model.tokenizer.decode(masked_input_ids, skip_special_tokens=True)[:-1]
                    decoded_texts.append(decoded_text)
                for decoded_text, true_answer in zip(decoded_texts, batch["answer_text"]):
                    pred_answer = decoded_text.split(":")[-1].strip().lower()
                    pred_answer = extract_number(pred_answer)
                    true_answer = true_answer.strip().lower()
                    if pred_answer == true_answer:
                        match_count += 1
                    total_count += 1

                # Log just the first sample text from first batch
                if i == 0 and self.is_main_process:
                    # Get only the tokens where loss_mask is 1 for the first example
                    masked_input = input_ids[0][loss_mask[0] == 1]
                    decoded_input = self.model.tokenizer.decode(masked_input, skip_special_tokens=True)
                    
                    with open("test_results.txt", "a") as f:
                        print(f"decoded input 0 (masked): {decoded_input}", file=f) 
                        print(f"decoded_texts 0: {decoded_texts[0]}", file=f)
                        print(f"batch['answer_text'] 0: {batch['answer_text'][0]}", file=f)
                        print('pred-true is ---', pred_answer, "---", true_answer, "---", pred_answer == true_answer, file=f)
                        print(70*"-" + "\n", file=f)
                    text_to_log = f"Validation Sample:\n{decoded_texts[0]}"
                    text_sample_table = wandb.Table(columns=["Generated Text"], data=[[text_to_log]])
                    wandb.log({
                        "val/generated_text_sample": text_sample_table
                    }, step=self.batches_done)


            val_loss_mean = val_loss_acc
            exact_match_rate = match_count / total_count if total_count > 0 else 0.0

            to_log[f"val/val_loss_mean"] = val_loss_mean.item()
            to_log[f"val/exact_match"] = exact_match_rate

        self.model.train()
        return to_log

    def run_training(self):
        # Val score for the base model
        # The test operation is not distributed and takes a lot of time
        # if self.is_main_process:
        #    to_log |= self.test()
        #    wandb.log(to_log, step=self.batches_done)
        print("Validation of the base model")
        to_log = self.validation()
        wandb.log(to_log, step=self.batches_done)

        # Synchronize processes before starting training
        self._barrier()

        total_epochs = self.config.num_epochs
        for epoch_id in range(total_epochs):
            print(f"Epoch {epoch_id} / {total_epochs} start")

            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch_id)

            if len(self.train_dataloader) < self.accum_steps:
                warnings.warn(
                    "No optimizer steps will be done since self.accum_steps > num_of_batches_in_dataloader"
                )

            self.run_epoch()

        # if self.is_main_process:
        #     # TODO make an issue to rewrite the test code to be able run it using distributed setup.
        #     to_log |= self.test()  # Final validation is performed at the last epoch
        #     wandb.log(to_log, step=self.batches_done + 1)
        #     print("To log: ", to_log)

        self._barrier()  # Synchronize processes after training

    def run_epoch(self):
        for micro_batch in tqdm(
            iterable=self.train_dataloader,
            disable=not self.is_main_process,  # Only show progress bar on main process
        ):
            to_log = {}
            to_log |= self.training_step(micro_batch)

            if self.batches_done in self.val_steps:
                print(f"validation go on step: {self.batches_done}")
                to_log |= self.validation()
                self.val_steps.remove(self.batches_done)

            # inside is decided to save the checkpoint or not
            self.save_checkpoint(self.batches_done)

            if to_log:
                wandb.log(to_log, step=self.batches_done)

        # TODO validation repeats if self.val_steps contains batch index that ends epoch
        to_log = self.validation()
        wandb.log(to_log, step=self.batches_done)

        self._barrier()
        self.save_checkpoint(self.batches_done)

    def training_step(self, micro_batch: BatchEncoding) -> Annotated[dict, "to_log"]:
        self.accumulation_step(micro_batch)

        if self.micro_batches_done % self.accum_steps == 0:
            # TODO add profiling
            return self.optimization_step()
        else:
            return {}

    def batch_to_device(self, micro_batch: BatchEncoding):
        for key, value in micro_batch.items():
            if isinstance(value, torch.Tensor):
                micro_batch[key] = value.to(self._device)

    def accumulation_step(self, micro_batch: BatchEncoding):
        # If this is the first micro-batch in an accumulation cycle, start timing
        if (self.micro_batches_done + 1) % self.accum_steps == 0:
            self.step_start_time = time.time()

        self.batch_to_device(micro_batch)
        forward_out = self.model(micro_batch)
        loss = forward_out["loss"]
        loss.backward()

        # Note that number of micro_batches_done is counted on each device
        self.micro_batches_done += 1
        self.loss_tot += loss
        if self.model_type == "codi":
            self.loss_teach += forward_out.teacher_loss
            self.loss_stud += forward_out.teacher_loss
            self.loss_distil += forward_out.distill_loss
        self.num_tokens += forward_out.num_tokens

    def optimization_step(self) -> Annotated[dict, "to_log"]:
        to_log = {}

        # Calculate time taken for this step
        step_end_time = time.time()
        step_time = step_end_time - self.step_start_time
        self.total_training_time += step_time

        # Summarizes values across all devices
        if self.is_distributed:
            dist.all_reduce(self.loss_tot, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.loss_teach, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.loss_stud, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.loss_distil, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.num_tokens, op=dist.ReduceOp.SUM)

        # Calculate throughput (tokens/second)
        num_tokens = self.num_tokens.item()
        if step_time > 0:
            current_throughput = num_tokens / step_time
            self.throughput_history.append(current_throughput)
            # Keep only the most recent window_size entries
            if len(self.throughput_history) > self.throughput_window_size:
                self.throughput_history = self.throughput_history[
                    -self.throughput_window_size :
                ]
            self.moving_avg_throughput = sum(self.throughput_history) / len(
                self.throughput_history
            )
            self.tokens_per_second = current_throughput

        # Scale the gradients from unnormalized loss by total # of tokens
        # FSDP averages gradient during reduction, to make grad scale agnostic towards wold_size,
        # we have to multiply them on world_size
        # grad_scaler = self.compensation_constant * self.world_size / self.num_tokens
        # scale_grads(self.model, grad_scaler)
        # max_norm = float(self.config.model.max_norm)
        # grad_norm = get_norm(self.model.parameters(), max_norm)
        # to_log["batch_grad_norm"] = grad_norm

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        lr = next(iter(self.optimizer.param_groups))["lr"]

        if self.scheduler is not None:
            self.scheduler.step()

        loss_to_log_tot = self.loss_tot / self.accum_steps
        to_log["train/loss_total"] = loss_to_log_tot.item()

        if self.model_type == "codi":
            loss_to_log_teach = self.loss_teach / self.accum_steps
            loss_to_log_stud = self.loss_stud / self.accum_steps
            loss_to_log_distil = self.loss_distil / self.accum_steps

            to_log["train/loss_teach"] = loss_to_log_teach.item()
            to_log["train/loss_stud"] = loss_to_log_stud.item()
            to_log["train/loss_distil"] = loss_to_log_distil.item()
        to_log["lr"] = lr
        self.total_tokens += int(self.num_tokens.item())
        to_log["tokens"] = self.total_tokens

        # Log throughput metrics
        to_log["performance/tokens_per_second"] = self.tokens_per_second
        to_log["performance/moving_avg_throughput"] = self.moving_avg_throughput

        # Calculate and log additional performance metrics
        if self.total_training_time > 0:
            to_log["performance/avg_tokens_per_second"] = (
                self.total_tokens / self.total_training_time
            )

        # Log batch statistics
        to_log["performance/batch_time_seconds"] = step_time
        to_log["performance/tokens_in_batch"] = num_tokens
        to_log["performance/total_training_time"] = self.total_training_time

        # If using multiple GPUs, log per-GPU throughput
        if self.world_size > 1:
            to_log["performance/tokens_per_second_per_gpu"] = (
                self.tokens_per_second / self.world_size
            )
            to_log["performance/moving_avg_throughput_per_gpu"] = (
                self.moving_avg_throughput / self.world_size
            )

        self.batches_done += 1
        self.loss_tot.zero_()
        self.loss_teach.zero_()
        self.loss_stud.zero_()
        self.loss_distil.zero_()
        self.num_tokens.zero_()

        # Reset timer for next step
        self.step_start_time = time.time()

        return to_log