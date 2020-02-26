import os
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCH = 10
