import json
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

class CODIModel(nn.Module):

    def __init__(self, model_path, icot_length=6, alpha=1, beta=1, gamma=1, max_length=256):
