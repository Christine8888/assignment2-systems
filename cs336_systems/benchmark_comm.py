import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import pandas as pd
import submitit

verbose = False
use_gpu = True

def setup(rank, world_size, backend = "gloo"):
    if use_gpu:
        torch.cuda.set_device(rank)
    
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def all_reduce_benchmark(rank, world_size, n_warmup: int, n_steps: int, data_size: int, backend = "gloo"):
    setup(rank, world_size, backend)
    
    # set up different CUDA devices for each process
    device = torch.device(f"cuda:{rank}" if use_gpu else "cpu")
    print(f"rank {rank} using device {device}")

    all_times = 0
    
    for i in range(n_warmup + n_steps):
        # generate random data
        data = torch.randint(0, 10, (data_size,), device=device, dtype=torch.float32)
        
        if use_gpu:
            torch.cuda.synchronize()

        if i < n_warmup:
            # run warmup steps
            dist.all_reduce(data, async_op=False)
            if use_gpu:
                torch.cuda.synchronize()
            continue
        
        start = timeit.default_timer()
        if verbose:
            print(f"rank {rank} data (before all-reduce): {data.shape}")
    
        dist.all_reduce(data, async_op=False)
        if use_gpu:
            torch.cuda.synchronize()
        
        end = timeit.default_timer()
        
        if verbose:
            print(f"rank {rank} data (after all-reduce): {data.shape}")
        
        all_times += end - start

    all_times_tensor = torch.tensor([all_times / n_steps], device=device)
    dist.all_reduce(all_times_tensor, async_op=False, op=dist.ReduceOp.SUM)

    # clean up
    dist.destroy_process_group()

    if rank == 0:
        # write result to shared memory
        mean_time = all_times_tensor.item() / world_size

        data_size_mb = data_size / 1024**2

        results = pd.DataFrame({
            "data_size (MB)": [data_size_mb],
            "world_size": [world_size],
            "backend": [backend],
            "avg. time (ms)": [mean_time * 1000]
        })
        
        # append to existing file if it exists
        file_name = f"comm_benchmark_{backend}.csv"
        try:
            existing = pd.read_csv(file_name)
            results = pd.concat([existing, results], ignore_index=True)
            # sort on data_size, then world_size (# of processes)
            results = results.sort_values(by=["data_size (MB)", "world_size"])
        except FileNotFoundError:
            pass
            
        results.to_csv(file_name, index=False)
        print(f"Results for data_size={data_size_mb:.2f}MB, world_size={world_size}:")
        print(results.tail(1))

        return mean_time

    return None

def benchmark_comms(backend: str, n_warmup: int, n_steps: int):
    data_sizes = [1024**2, 10 * 1024**2, 100 * 1024**2, 1024**3]
    n_procs = [2, 4, 6]

    ind = 0
    for data_size in data_sizes:
        for n_proc in n_procs:
            print(f"Benchmarking {data_size} with {n_proc} processes and {backend} backend")
            mp.spawn(fn=all_reduce_benchmark,
                       args=(n_proc, n_warmup, n_steps, data_size, backend),
                        nprocs=n_proc, 
                        join=True,)
            
            ind += 1

if __name__ == "__main__":
    n_warmup = 5
    backend = "gloo"
    n_steps = 20
    benchmark_comms(backend, n_warmup, n_steps)
