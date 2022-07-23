import torch

import multiprocessing as mp
from multiprocessing import shared_memory

def printShape(shm):
    tensor = torch.frombuffer(shm.buf, dtype = torch.float32)
    print(tensor)
    print(tensor.shape)

if __name__ == '__main__':
    test_array = torch.tensor([1, 2, 3], dtype = torch.float32)
    test_shm   = shared_memory.SharedMemory(create = True, size = test_array.element_size() * test_array.numel())
    torch.frombuffer(test_shm.buf, dtype = torch.float32).copy_(test_array)

    printShape(test_shm)

    p = mp.Process(target = printShape, args = (test_shm,))
    p.start()
    p.join()