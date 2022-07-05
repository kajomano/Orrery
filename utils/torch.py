import torch

class DmModule():
    def __init__(self, device = 'cpu'):
        self.device = torch.device(device)

    def to(self, device):
        for name, attr in self.__dict__.items():
            if torch.is_tensor(attr) or isinstance(attr, DmModule):
                self.__dict__[name] = attr.to(torch.device(device))

        self.device = device
        
        return(self)
