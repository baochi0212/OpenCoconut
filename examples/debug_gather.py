import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def setup(rank, world_size):
    """Set up the distributed process group."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set CUDA device for this process

def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()

def average_with_nccl(rank, world_size):
    """Perform tensor averaging using NCCL backend."""
    setup(rank, world_size)
    # with torch.inference_mode():
    if True: 
        # Create a tensor on the GPU corresponding to this process
        local_tensor = torch.tensor([1.0], device=f"cuda:{rank}")
        
        # Perform all_reduce operation with SUM, then divide by world_size for average
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
        #local_tensor /= world_size  # Average
    
        if rank == 0: 
            print(f"Process {rank} - Averaged Tensor: {local_tensor.item()}")
    
    cleanup()

def run_demo(world_size):
    """Run the distributed computation demo."""
    processes = []
    for rank in range(world_size):
        p = Process(target=average_with_nccl, args=(rank, world_size))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    world_size = 6  # Number of processes (and GPUs)
    run_demo(world_size)
