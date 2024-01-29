import torch
import os

class PostConfig:
    epochs = 5
    max_grad_norm = 10
    lr = 1e-5
    batch_size = 4

    data_path = "/home/cywell/Project/Myungbin-bot/data/raw/all_group.txt"

class FineConfig:
    epochs = 5
    max_grad_norm = 10
    lr = 1e-6
    batch_size = 4
    

pcfg = PostConfig()
fcfg = FineConfig()

def SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, "post_model.pth"))
    
def FineSaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, "fine_model.pth"))