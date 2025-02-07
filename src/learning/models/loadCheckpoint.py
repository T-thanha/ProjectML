import os
import torch

def loadCheckpoint(model, optimizer, model_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss = checkpoint["train_loss"]
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        return model, optimizer, train_loss, start_epoch, best_loss