import torch
import numpy as np

def create_tensors(tensors):
    lst = [1,2,3,4,5]
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    tensor_lst = torch.tensor(lst)                  #tensor from list
    tensor_np = torch.from_numpy(np_arr)            #tensor from np array
    tensor_rand = torch.rand(1,5)                   #tensor from random values
    tensor_ones = torch.ones(1,5)                   #tensor of ones
    tensor_zeroes = torch.zeros(1,5)                #tensor of zeroes

    tensors.append(tensor_lst)
    tensors.append(tensor_np)
    tensors.append(tensor_rand)
    tensors.append(tensor_ones)
    tensors.append(tensor_zeroes)

    print(f"\nTensor from list: {tensor_lst}")      
    print(f"Tensor from np array: {tensor_np}")
    print(f"Tensor from random: {tensor_rand}")
    print(f"Tensor from ones: {tensor_ones}")
    print(f"Tensor from zeroes: {tensor_zeroes}\n")

def tensor_details(tensors):
    for tensor in tensors:
        print("*" * 100)
        print(f"tensor shape: {tensor.shape}")
        print(f"tensor dtype: {tensor.dtype}")
        print(f"tensor device: {tensor.device}")
        print(f"full tensor: {tensor}")





if __name__ == "__main__":
    tensors = []
    create_tensors(tensors)
    tensor_details(tensors)