import os
import torch

def saveCheckpoint(model_save_path, model , optimizer,train_loss_list,
                   epoch,lr,weight_decay):
    '''
    Save model checkpoint
    '''
    # > check save location
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # -> Set checkpoint filename
    model_file = f"MLP" +\
                 f"-train_loss{train_loss_list[-1]:.4f}" +\
                 f"-epoch={epoch}" +\
                 f"-lr={lr}" +\
                 f"-wd={weight_decay}" + ".pt"
    
    # -> Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "train_loss": train_loss_list,
        "epoch": epoch
    }, model_save_path + model_file)

def saveBestCheckpoint(model_save_path, model, optimizer, train_loss_list,
                       epoch, lr, weight_decay, best_loss):
    current_loss = train_loss_list[-1]

    if current_loss < best_loss:

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        for file in os.listdir(model_save_path):
            if file.startswith("MLP-best"):
                os.remove(os.path.join(model_save_path, file))

        model_file = f"MLP-best-train_loss{current_loss:.4f}-epoch={epoch}-lr={lr}-wd={weight_decay}.pt"

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss_list,
            "epoch": epoch,
            "best_loss": current_loss,
        }, os.path.join(model_save_path, model_file))

        return current_loss 

    return best_loss