from model import OurModel
from eval import eval
import torch
from torch import nn
import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    # TODO, below is only code template, no batching
    model.train()
    running_loss = 0.0
    for idx, data in tqdm(dataloader):
        data = data.to(device)
        expected_outputs = None
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, expected_outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
    return running_loss / len(dataloader)


def main(batch_size=64, epochs=5, lr=1e-3):
    print("Supervised Training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = ...
    val_loader = ...

    # Model
    model = OurModel()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = 100000000
    best_model_path = "best_model.pth"

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model (based on top-1 accuracy)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

