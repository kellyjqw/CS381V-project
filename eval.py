from model import OurModel
import torch


def eval(model, dataloader):
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
    pass

def main(best_model_path):
    # Load best model for analysis
    model = OurModel()
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # Evaluate final model
    val_loader = ...
    eval(model, val_loader)
