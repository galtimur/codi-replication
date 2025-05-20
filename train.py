import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from codi_model import CODIModel
from data import GSM8kDataset, collate_fn, Config
from torch.utils.data import DataLoader


def main():
    # Load configuration
    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    data_config = Config(config) # GSM8kDataset and DataLoader expect a dot-accessible config for data part

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config.TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = CODIModel(config_path=config_path)

    # Initialize dataset and dataloader
    dataset = GSM8kDataset(
        tokenizer=tokenizer,
        dataset_name=data_config.DATASET_NAME,
        eot_token=config.model.eot_token,
        bot_token=config.model.bot_token,
        split=data_config.DATASET_SPLIT_TRAIN,    # Using train split for training
        num_samples=data_config.NUM_SAMPLES_TRAIN 
    )

    dataloader = DataLoader(
        dataset,
        batch_size=data_config.BATCH_SIZE,
        shuffle=data_config.SHUFFLE_DATALOADER,
        num_workers=data_config.NUM_WORKERS,
        collate_fn=lambda batch_arg: collate_fn(batch_arg, tokenizer, data_config.MAX_SEQ_LENGTH)
    )

    batch = next(iter(dataloader))

    # Prepare inputs for the model
    # The CODIModel.forward expects a dictionary with 'question_ids' and 'answer_ids'
    inputs = {
        "question_ids": batch["question_input_ids"],
        "answer_ids": batch["answer_input_ids"] 
        # teacher_full_input_ids will be needed later
    }

    print('first question:', batch["question_str"][0])
    print('first answer:', batch["answer_str"][0])

    # Perform a forward pass
    print("Performing a forward pass...")
    outputs = model.forward(inputs)
    print("Forward pass successful.")
    print("Output logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
