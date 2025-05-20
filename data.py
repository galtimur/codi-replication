import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class GSM8kDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, eot_token, bot_token, split="train", num_samples=None):
        self.tokenizer = tokenizer
        self.hf_dataset = load_dataset(dataset_name, split=split)
        if num_samples is not None and num_samples < len(self.hf_dataset):
            self.hf_dataset = self.hf_dataset.select(range(num_samples))
        self.eot_token = eot_token
        self.bot_token = bot_token

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        question_text = sample['question']
        original_cot = sample['cot']
        answer_text = sample['answer']
        
        ### Delete last CoT step if the answer is in last step
        cot_steps = original_cot.split('>>')
        if len(cot_steps) > 1:
            last_step = cot_steps[-2]
            if last_step.endswith(answer_text):  ### TODO: this can be done as "str contains answer" rather than "str in answer", but i think this is correct due to the dataset
                processed_cot = '>>'.join(cot_steps[:-2]) + '>>'
            else:
                processed_cot = original_cot
        else:
            # TODO: this actually happened so we need to skip those I guess
            raise ValueError("There is only one step in the CoT and it contains the answer")


        ### TODO: check all the spaces!?
            
        # 1) bos+question+bot
        question_str = f"{self.tokenizer.bos_token}{question_text}{self.bot_token}"
        
        # 2) eot+TheAnswerIs:{answer}+eos
        answer_str = f"{self.eot_token}The answer is:{answer_text}{self.tokenizer.eos_token}"
        
        # 3) bos+question+cot+TheAnswerIs:{answer}+eos
        teacher_full_str = f"{self.tokenizer.bos_token}{question_text}{processed_cot}The answer is:{answer_text}{self.tokenizer.eos_token}"
        
        return {
            "question_str": question_str,
            "answer_str": answer_str,
            "teacher_full_str": teacher_full_str,
            
            "raw_cot": original_cot,
            "processed_cot": processed_cot,
            "answer_text": answer_text
        }


def collate_fn(batch, tokenizer, max_seq_length):
    question_strs = [item['question_str'] for item in batch]
    answer_strs = [item['answer_str'] for item in batch]
    teacher_full_strs = [item['teacher_full_str'] for item in batch]

    original_padding_side = tokenizer.padding_side

    # 1. Tokenize questions (bos+question+bot) with padding on the left
    tokenizer.padding_side = "left"
    tokenized_questions = tokenizer(
        question_strs,
        padding='longest',
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False
    )
    question_input_ids = tokenized_questions.input_ids
    question_attention_mask = tokenized_questions.attention_mask
    
    # 2. Tokenize answers (TheAnswerIs:{answer}+eos) with padding on the left
    tokenized_answer = tokenizer(
        answer_strs,
        padding='longest',
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False
    )
    answer_input_ids = tokenized_answer.input_ids
    answer_attention_mask = tokenized_answer.attention_mask

    # 3. Tokenize teacher_full sequences (bos+question+cot+TheAnswerIs:{answer}+eos) with padding on the right
    tokenized_teacher_full = tokenizer(
        teacher_full_strs,
        padding='longest',
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
        add_special_tokens=False
    )
    teacher_full_input_ids = tokenized_teacher_full.input_ids
    teacher_full_attention_mask = tokenized_teacher_full.attention_mask

    # Calculate answer lengths once
    answer_lengths = [len(tokenizer.encode(item['answer_text'], add_special_tokens=False)) for item in batch]
    
    # For both answer and teacher sequences, semicolon is at: answer_length + 2 (1 for eos, 1 for semicolon) from the end of the sequence
    semicolon_pos = [length + 2 for length in answer_lengths]
    
    return {
        "question_input_ids": question_input_ids,
        "question_attention_mask": question_attention_mask,
        "answer_input_ids": answer_input_ids,
        "answer_attention_mask": answer_attention_mask,
        "teacher_full_input_ids": teacher_full_input_ids,
        "teacher_full_attention_mask": teacher_full_attention_mask,
        "semicolon_position_from_end_answer": torch.tensor(semicolon_pos, dtype=torch.long),
        "semicolon_position_from_end_teacher_full": torch.tensor(semicolon_pos, dtype=torch.long),
        
        # This is needed for debugging
        "question_str": [item['question_str'] for item in batch],
        "answer_str": [item['answer_str'] for item in batch],
        "teacher_full_str": [item['teacher_full_str'] for item in batch],
        "raw_cot": [item['raw_cot'] for item in batch],
        "processed_cot": [item['processed_cot'] for item in batch]
    }

def get_dataset(config):

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = GSM8kDataset(
        tokenizer=tokenizer,
        dataset_name=config.DATASET_NAME,
        eot_token=config.model.eot_token,
        bot_token=config.model.bot_token,
        split=config.DATASET_SPLIT_TEST,
        num_samples=config.NUM_SAMPLES_TEST
    )

    return dataset

def get_dataloader(config):

    dataset = get_dataset(config)

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # TODO: change to True for training
        num_workers=config.NUM_WORKERS,
        collate_fn=lambda batch_arg: collate_fn(batch_arg, dataset.tokenizer, config.MAX_SEQ_LENGTH)  ### TODO: this is a lambda function, I can change it if needed
    )

    return dataloader



if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as f:
        config = Config(yaml.safe_load(f))

    dataloader = get_dataloader(config)
    tokenizer = dataloader.dataset.tokenizer

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
    print(f"BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")


    for i, batch_data in enumerate(dataloader):
        if i >= config.NUM_BATCHES_TO_SHOW:
            break
        print(f"\nBatch {i+1}:")
        if batch_data['question_input_ids'].shape[0] > 0:
            print(f"Full decoded question: {tokenizer.decode(batch_data['question_input_ids'][0], skip_special_tokens=True)}")
            print(f"Full decoded answer: {tokenizer.decode(batch_data['answer_input_ids'][0], skip_special_tokens=True)}")
            print(f"Full decoded teacher: {tokenizer.decode(batch_data['teacher_full_input_ids'][0], skip_special_tokens=True)}")
            
            # Robustly get the token ID and decode it for the debug print
            idx_from_end_answer = batch_data['semicolon_position_from_end_answer'][0].item()
            idx_from_end_teacher = batch_data['semicolon_position_from_end_teacher_full'][0].item()
            token_id_at_pos_answer = batch_data['answer_input_ids'][0][-idx_from_end_answer].item()
            token_id_at_pos_teacher = batch_data['teacher_full_input_ids'][0][-idx_from_end_teacher].item()
            decoded_token_at_pos_answer = tokenizer.decode([token_id_at_pos_answer])
            decoded_token_at_pos_teacher = tokenizer.decode([token_id_at_pos_teacher])
            
            print(f"Place of ':' from the end in answer is: {idx_from_end_answer} \
                  and it is actually - '{decoded_token_at_pos_answer}'")
            print(f"Place of ':' from the end in teacher is: {idx_from_end_teacher} \
                  and it is actually - '{decoded_token_at_pos_teacher}'")