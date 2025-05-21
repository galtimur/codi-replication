import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class GSM8kDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_name,
        eot_token,
        bot_token,
        split="train",
    ):
        self.tokenizer = tokenizer
        self.hf_dataset = load_dataset(dataset_name, split=split)
        self.eot_token = eot_token
        self.bot_token = bot_token

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        question_text = sample["question"]
        original_cot = sample["cot"]
        answer_text = sample["answer"]

        ### Delete last CoT step if the answer is in last step
        cot_steps = original_cot.split(">>")
        if len(cot_steps) > 1:
            last_step = cot_steps[-2]
            if last_step.endswith(
                answer_text
            ):  ### TODO: this can be done as "str contains answer" rather than "str in answer", but i think this is correct due to the dataset
                processed_cot = ">>".join(cot_steps[:-2]) + ">>"
            else:
                processed_cot = original_cot
        else:
            # for this we leave the original cot
            processed_cot = original_cot
            ### TODO: this is a hack, we should actually skip this sample

        ### TODO: check all the spaces!?

        # 1) bos+question+bot
        # question_str = f"{self.tokenizer.bos_token}{question_text}{self.bot_token}"
        question_str = question_text

        # 2) eot+TheAnswerIs:{answer}+eos
        # answer_str = (
        #     f"{self.eot_token}The answer is:{answer_text}{self.tokenizer.eos_token}"
        # )
        answer_str = f"The answer is:{answer_text}"

        # 3) bos+question+cot+TheAnswerIs:{answer}+eos
        # teacher_full_str = f"{self.tokenizer.bos_token}{question_text}{processed_cot}The answer is:{answer_text}{self.tokenizer.eos_token}"
        teacher_full_str = f"{question_text}{processed_cot}The answer is:{answer_text}"

        return {
            "question_str": question_str,
            "answer_str": answer_str,
            "teacher_full_str": teacher_full_str,
            "raw_cot": original_cot,
            "processed_cot": processed_cot,
            "answer_text": answer_text,
        }


def collate_fn(batch, tokenizer, max_seq_length):
    question_strs = [item["question_str"] for item in batch]
    answer_strs = [item["answer_str"] for item in batch]
    teacher_full_strs = [item["teacher_full_str"] for item in batch]


    # 1. Tokenize questions (bos+question+bot) with padding on the left
    tokenizer.padding_side = "left"
    tokenized_questions = tokenizer(
        question_strs,
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    question_input_ids = tokenized_questions.input_ids
    question_attention_mask = tokenized_questions.attention_mask

    # 2. Tokenize answers (TheAnswerIs:{answer}+eos) with padding on the right
    tokenizer.padding_side = "right"
    tokenized_answer = tokenizer(
        answer_strs,
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    answer_input_ids = tokenized_answer.input_ids
    answer_attention_mask = tokenized_answer.attention_mask

    # 3. Tokenize teacher_full sequences (bos+question+cot+TheAnswerIs:{answer}+eos) with padding on the left
    tokenizer.padding_side = "left"
    tokenized_teacher_full = tokenizer(
        teacher_full_strs,
        padding="longest",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    teacher_full_input_ids = tokenized_teacher_full.input_ids
    teacher_full_attention_mask = tokenized_teacher_full.attention_mask
    teacher_full_loss_mask = teacher_full_attention_mask.clone()

    # Get question lengths for each item in batch
    question_lengths = tokenized_questions.attention_mask.sum(dim=1).tolist()
    
    batch_size = len(question_lengths)
    for i in range(batch_size):
        padding_end = teacher_full_loss_mask[i].nonzero()[0].item()
        mask_end = padding_end + question_lengths[i] - 3
        teacher_full_loss_mask[i, padding_end:mask_end-1] = 0
    
    # Get the token ID for ":"
    colon_token_id = tokenizer.encode(":")[0]
    # Compute distance from end for the last ":" in each sequence
    def get_last_colon_pos(sequences, colon_id):
        colon_positions = (sequences == colon_id).nonzero(as_tuple=True)[1]
        batch_indices = torch.arange(sequences.size(0), device=sequences.device)
        last_colons = torch.searchsorted(colon_positions, batch_indices[1:], right=True) - 1
        distances = sequences.size(1) - colon_positions[last_colons]
        return distances.to(dtype=torch.long)

    answer_semicolon_pos = get_last_colon_pos(answer_input_ids, colon_token_id)
    teacher_semicolon_pos = get_last_colon_pos(teacher_full_input_ids, colon_token_id)

    return {
        "question_input_ids": question_input_ids,
        "question_attention_mask": question_attention_mask,
        "answer_input_ids": answer_input_ids,
        "answer_attention_mask": answer_attention_mask,
        "teacher_full_input_ids": teacher_full_input_ids,
        "teacher_full_attention_mask": teacher_full_attention_mask,
        "teacher_full_loss_mask": teacher_full_loss_mask,
        "semicolon_position_from_end_answer": answer_semicolon_pos,
        "semicolon_position_from_end_teacher_full": teacher_semicolon_pos,
        "answer_text": [item["answer_text"] for item in batch],
    }


def get_datasets(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = GSM8kDataset(
        tokenizer=tokenizer,
        dataset_name=config.data.dataset.name,
        eot_token=config.model.eot_token,
        bot_token=config.model.bot_token,
        split=config.data.dataset.split_train,
    )
    test_dataset = GSM8kDataset(
        tokenizer=tokenizer,
        dataset_name=config.data.dataset.name,
        eot_token=config.model.eot_token,
        bot_token=config.model.bot_token,
        split=config.data.dataset.split_test,
    )

    return train_dataset, test_dataset


def get_dataloader(config):
    train_dataset, test_dataset = get_datasets(config)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.dataloader.batch_size,
        shuffle=config.data.dataloader.train_shuffle,
        num_workers=config.data.dataloader.num_workers,
        collate_fn=lambda batch_arg: collate_fn(
            batch_arg, train_dataset.tokenizer, config.model.max_length
        ),  ### TODO: this is a lambda function, I can change it if needed
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.data.dataloader.batch_size,
        shuffle=config.data.dataloader.test_shuffle,
        num_workers=config.data.dataloader.num_workers,
        collate_fn=lambda batch_arg: collate_fn(
            batch_arg, test_dataset.tokenizer, config.model.max_length
        ),
    )
    return train_dataloader, [test_dataloader]


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = Config(yaml.safe_load(f))

    train_dataloader, test_dataloader = get_dataloader(config)
    tokenizer = train_dataloader.dataset.tokenizer

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")

    for i, batch_data in enumerate(test_dataloader):
        if i >= config.NUM_BATCHES_TO_SHOW:
            break
        print(f"\nBatch {i+1}:")
        if batch_data["question_input_ids"].shape[0] > 0:
            print(
                f"Full decoded question: {tokenizer.decode(batch_data['question_input_ids'][0], skip_special_tokens=True)}"
            )
            print(
                f"Full decoded answer: {tokenizer.decode(batch_data['answer_input_ids'][0], skip_special_tokens=True)}"
            )
            print(
                f"Full decoded teacher: {tokenizer.decode(batch_data['teacher_full_input_ids'][0], skip_special_tokens=True)}"
            )

            # Robustly get the token ID and decode it for the debug print
            idx_from_end_answer = batch_data["semicolon_position_from_end_answer"][
                0
            ].item()
            idx_from_end_teacher = batch_data[
                "semicolon_position_from_end_teacher_full"
            ][0].item()
            token_id_at_pos_answer = batch_data["answer_input_ids"][0][
                -idx_from_end_answer
            ].item()
            token_id_at_pos_teacher = batch_data["teacher_full_input_ids"][0][
                -idx_from_end_teacher
            ].item()
            decoded_token_at_pos_answer = tokenizer.decode([token_id_at_pos_answer])
            decoded_token_at_pos_teacher = tokenizer.decode([token_id_at_pos_teacher])

            print(
                f"Place of ':' from the end in answer is: {idx_from_end_answer} \
                  and it is actually - '{decoded_token_at_pos_answer}'"
            )
            print(
                f"Place of ':' from the end in teacher is: {idx_from_end_teacher} \
                  and it is actually - '{decoded_token_at_pos_teacher}'"
            )
