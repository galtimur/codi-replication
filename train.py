import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from codi_model import CODIModel
from data import Config, get_dataloader, get_dataset


def main():
    # Load configuration
    config_path = "configs/config.yaml"
    config = OmegaConf.load(config_path)
    data_config = Config(
        config
    )  # GSM8kDataset and DataLoader expect a dot-accessible config for data part

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_config.model.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model
    model = CODIModel(config_path=config_path)

    # Get dataloader using helper functions (gets dataset inside)
    train_dataloader, val_dataloader = get_dataloader(data_config)

    batch = next(iter(train_dataloader))

    # Prepare inputs for the model
    # The CODIModel.forward expects a dictionary with 'question_ids' and 'answer_ids'
    inputs = {
        "question_ids": batch["question_input_ids"],
        "answer_ids": batch["answer_input_ids"]
        # teacher_full_input_ids will be needed later
    }

    print("first question:", batch["question_str"][0])
    print("first answer:", batch["answer_str"][0])

    # Perform a forward pass
    print("Performing a forward pass...")
    outputs = model.forward(inputs)
    print("Forward pass successful.")
    print("Output logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    main()
