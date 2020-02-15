import os
import torch

DATASET_DIR = os.path.join(os.path.dirname(__file__), '../data')
BATCH_SIZE = 16  # 한번에 러닝하는 파일의 수
LEARNING_RATE = 0.01
NUM_EPOCH = 20  # 전체를 몇바퀴 돌리는지
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../saved_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
