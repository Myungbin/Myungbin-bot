import torch
import os


class PostConfig:
    epochs = 10
    max_grad_norm = 10
    lr = 1e-5
    batch_size = 4

    data_path = "/home/cywell/Project/Myungbin-bot/data/processed/all_group.txt"


class FineConfig:
    epochs = 10
    max_grad_norm = 10
    lr = 1e-6
    batch_size = 4


class InferenceConfig:
    batch_size = 32
    top_n = 10


pcfg = PostConfig()
fcfg = FineConfig()


def SaveModel(model, path, model_name="post_model.pth"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, model_name))


def FineSaveModel(model, path, model_name="fine_model.pth"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, model_name))
