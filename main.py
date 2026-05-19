
import os
import sys
import torch
torch.set_num_threads(6)
os.environ["WANDB_DISABLED"] = "True"
from ILLMRec.train import train

if __name__ == "__main__":
    train()