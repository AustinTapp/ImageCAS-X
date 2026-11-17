"""Install this file in the external nnU-Net trainer search directory."""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSHK25(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.num_epochs = 25
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 48
        self.save_every = 5


class nnUNetTrainerSHK5(nnUNetTrainerSHK25):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.num_epochs = 5


class nnUNetTrainerSHK80(nnUNetTrainerSHK25):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.num_epochs = 80
