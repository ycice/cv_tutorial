import os
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
BATCH_SIZE = 16  # 원래는 256
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCH = 8
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../saved_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
