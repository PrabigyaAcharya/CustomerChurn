import torch
import json

with open('src/config.json', 'r') as f:
    config = json.load(f)

def save_moodel(model):
    torch.save(model.state_dict(), config["model"]["model_path"])

def load_model(model_class):
    model = model_class(18, 24, 1)
    torch.load_state_dict()
    model.eval()
    return model
