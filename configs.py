'''
File is taken from kotlin-initiative repo
'''

from typing import List, Literal

from pydantic.main import BaseModel as PydanticBaseModel


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


DatasetName = Literal[
    "kexer_train",
    "kexer_test",
    "JetBrains-Research/kexer-gen:train",
    "JetBrains-Research/kexer-gen:validation",
    "JetBrains/KStack-clean",
    "JetBrains-Research/lca-codegen-train:train",
    "lca-codegen-test",
    "kexer_public_train",
    "kexer_public_test",
]

FIM_LEVEL = Literal["CHAR", "TOKEN"]


class DatasetConfig(PydanticBaseModel):
    name: DatasetName
    local_folder: str
    remote_uri: str
    sequence_length: int
    batch_size: int
    num_workers: int
    subset: int | str = ""
    source: Literal["hf", "s3"] = "s3"
    # Preprocessor + dataloader
    data_preprocessor: str = "KexerDataPreprocessor"
    data_loader: str = "PyDataloader"
    tokenizer_wrapper: str = ""
    # FIM
    fim_level: FIM_LEVEL | None = None
    fim_proportion: float | None = None
    psm_to_spm_ratio: float | None = None


class DatasetGroupConfig(PydanticBaseModel):
    train: DatasetConfig
    validation: List[DatasetConfig]


class ModelConfig(PydanticBaseModel):
    hf_model: str
    max_lr: float
    weight_decay: float
    warmup_steps: int | float
    max_norm: float | Literal["inf"]
    scheduler: Literal["linear", "cosine", "cosine_warmup", "cosine_warmup_restarts"]
    optimizer: Literal["AdamW", "Adam", "SGD"]
    use_neftune: bool
    neftune_alpha: int
    z_loss: float
    wrapper_model: str = "HFModel"

    # Mixed Precision training
    train_precision: Literal["bfloat16", "float16", "float32"]
    mixed_precision: bool
    use_grad_scaler: bool
    compile_model: bool


class FSDPConfig(PydanticBaseModel):
    enable_activation_checkpointing: bool = False  # True reduces memory
    enable_activation_offloading: bool = False  # True reduces memory
    custom_sharded_layers: list[str] | None = (
        None  # Layers to shard separately (useful for large vocab size models). Lower Memory, but lower speed.
    )
    fsdp_cpu_offload: bool = (
        False  # Offloads model parameters to CPU during training. Used to fit model into GPU
    )
    # Setting fsdp_reshard_after_forward to True corresponds to the FULL_SHARD sharding strategy from FSDP1,
    # while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
    fsdp_reshard_after_forward: bool = True
    optimizer_in_bwd: bool = (
        False  # True saves memory. Requires gradient_accumulation_steps=1.
    )


class DDPConfig(PydanticBaseModel):
    dummy: bool | None = None


class DistributedConfig(PydanticBaseModel):
    framework: str
    fsdp: FSDPConfig
    ddp: DDPConfig


class TrainConfig(PydanticBaseModel):
    effective_batch_size: int
    num_epochs: int
    local_checkpoint_path: str
    s3_checkpoint_path: str | None
    load_from_checkpoint: bool = False
    load_ckpt_path: str | None = None
    save_checkpoints_every: int
    validate_every: int
    eval_device: str

    # Reproducibility
    seed: int
    cudnn_benchmark: bool

    # Datasets
    datasets: DatasetGroupConfig
    model: ModelConfig
    distributed: DistributedConfig
