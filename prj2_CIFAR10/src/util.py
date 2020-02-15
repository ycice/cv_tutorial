import os
import torch
from prj2_CIFAR10.src.constants import SAVED_MODEL_DIR


def save_model(model: torch.nn.Module, model_name: str):
    pretrained_path = os.path.join(SAVED_MODEL_DIR, f'{model_name}.torch')
    torch.save(model.state_dict(), pretrained_path)


def load_model(model: torch.nn.Module, model_name: str):
    pretrained_path = os.path.join(SAVED_MODEL_DIR, f'{model_name}.torch')
    model.load_state_dict(torch.load(pretrained_path))
