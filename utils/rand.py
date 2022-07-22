import torch
from torch.nn.functional import normalize

from utils.consts import ftype, pi

def randOnSphere(n, device):
    return(normalize(torch.randn((n, 3), dtype = ftype, device = device), dim = 1))

def randInCircle(n, device):
    r     = torch.sqrt(torch.rand((n, 1), dtype = ftype, device = device))
    theta = torch.rand((n, 1), dtype = ftype, device = device) * (2 * pi)
    
    return(torch.cos(theta) * r, torch.sin(theta) * r)

def randInSquare(n, device):
    return(
        torch.rand((n, 1), dtype = ftype, device = device) - 0.5,
        torch.rand((n, 1), dtype = ftype, device = device) - 0.5
    )