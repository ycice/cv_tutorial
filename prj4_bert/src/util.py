import os
import torch
from constants import DATA_DIR


def save_model(model: torch.nn.Module, model_name: str):
    save_location = os.path.join(DATA_DIR, f'{model_name}.torch')
    torch.save(model.state_dict(), save_location)

