import torch
from tqdm import trange, tqdm
from learning.models import saveBestCheckpoint, loadCheckpoint

def train_model(device, model, train_loader, val_loader="", num_epochs=5,
                lr=0.005, weight_decay=0.0005, model_save_path="", model_name=""):
    
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    train_losses = []
    start_epoch = 0
    best_loss = float('inf')

    # -> Load model checkpoint
    if model_name:
        model, optimizer, train_losses, start_epoch, best_loss = loadCheckpoint(model, optimizer, model_save_path+model_name)
    
    # -> Training epochs loop
    for epoch in trange(start_epoch, start_epoch+num_epochs, desc="Training Epochs"):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch+num_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss)


        print(f"Epoch [{epoch+1}/{start_epoch+num_epochs}], Training Loss: {epoch_loss:.4f}")

        best_loss = saveBestCheckpoint(model_save_path, model, optimizer, train_losses, epoch, lr, weight_decay, best_loss)
        
    print(f"Training completed, Best loss: {best_loss}")
    return model