import torch
import time
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

def transform_tensor():
    vector = torch.arange(16)
    print(f"\nVector of tensors: {vector}")
    matrix = vector.reshape(4,4)
    print(f"\nMatrix of 4 x 4 tensors: {matrix}")
    #alternative use matrix = torch.arange(16).reshape(4,4)

def transform_tensor_view():
    vector = torch.arange(16)
    matrix = vector.view(4, -1)
    print(f"\nVector of tensors: {vector}")
    print(f"\nReshaped Matrix: {matrix}")

#Find the best available device: CUDA -> MPS -> CPU
def find_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("-"*60)
        print(f"CUDA is availabe, using device: {torch.cuda.get_device_name(0)}.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("-"*60)
        print(f"MPS is available, using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        print("-"*60)
        print(f"No GPU available, using CPU.")
    return device

def benchmark(device):
    size = 2000
    tensor_a_cpu = torch.rand(size, size)
    tensor_b_cpu = torch.rand(size, size)

    #Time CPU computation
    start_time = time.time()
    tensor_result = torch.matmul(tensor_a_cpu, tensor_b_cpu)
    cpu_time = time.time() - start_time
    print(f"\nCPU time: {cpu_time:.4f} seconds")

    #Time GPU computation if available
    if device.type != "cpu":
        tensor_a_gpu = tensor_a_cpu.to(device)
        tensor_b_gpu = tensor_b_cpu.to(device)

        #Warm up (first operation can be slower)
        _ = torch.matmul(tensor_a_gpu, tensor_b_gpu)

        start_time = time.time()
        result_gpu = torch.matmul(tensor_a_gpu, tensor_b_gpu)
        gpu_time = time.time() - start_time

        print(f"{device.type.upper()} time: {gpu_time:.4f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x faster")
        print("-"*60)

    else:
        print("No GPU available - skipping GPU comparison")
        print("-"*60)

if __name__ == "__main__":
    tensors = []
    #create_tensors(tensors)
    #tensor_details(tensors)
    #transform_tensor()
    #transform_tensor_view()
    device = find_device()
    benchmark(device)