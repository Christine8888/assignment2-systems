import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import pandas as pd

verbose = False

def setup(rank, world_size, backend = "gloo"):
    if backend == "nccl":
        torch.cuda.set_device(rank)
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def all_reduce_benchmark(rank, world_size, n_warmup: int, n_steps: int, data_size: int, mean_time: torch.Tensor, backend = "gloo"):
    setup(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")

    all_times = 0
    for i in range(n_warmup + n_steps):
        # generate random data
        data = torch.randint(0, 10, (data_size,), device=device, dtype=torch.float32)
        
        if backend == "nccl":
            torch.cuda.synchronize()

        if i < n_warmup:
            # run warmup steps
            dist.all_reduce(data, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()
            continue
        
        start = timeit.default_timer()
        if verbose:
            print(f"rank {rank} data (before all-reduce): {data}")
    
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        
        if verbose:
            print(f"rank {rank} data (after all-reduce): {data}")
        
        all_times += end - start

    all_times_tensor = torch.tensor([all_times / n_steps], device=device)
    dist.all_reduce(all_times_tensor, async_op=False, op=dist.ReduceOp.SUM)

    if rank == 0:
        # write result to shared memory
        mean_time[0] = all_times_tensor.item() / world_size

    return None
            
def run_benchmark(backend: str, data_size: int, n_procs: int, n_warmup: int, n_steps: int):
    mean_time = torch.tensor([0], dtype=torch.float32).share_memory_()
    mp.spawn(fn=all_reduce_benchmark,
                       args=(n_procs, n_warmup, n_steps, data_size, mean_time, backend),
                        nprocs=n_procs, 
                        join=True,)
    return mean_time.item()


def benchmark_comms(backend: str, n_warmup: int, n_steps: int):
    data_sizes = [1024**2, 10 * 1024**2, 100 * 1024**2, 1024**3]
    n_procs = [2, 4, 6]
    

    times = pd.DataFrame(columns = ["data_size (MB)", "n_procs", "backend", "avg. time (ms)"])

    ind = 0
    for data_size in data_sizes:
        for n_proc in n_procs:
            print(f"Benchmarking {data_size} with {n_proc} processes")
            mean_time = run_benchmark(backend, data_size, n_proc, n_warmup, n_steps)
            times.loc[ind] = {"data_size (MB)": int(data_size / 1024**2), 
                              "n_procs": n_proc, "backend": backend,
                              "avg. time (ms)": mean_time * 1000,}
            ind += 1

    # save markdown table
    times.to_csv("comm_benchmark.csv", index=False)
    print(times)

if __name__ == "__main__":
    n_warmup = 5
    n_steps = 20
    benchmark_comms("gloo", n_warmup, n_steps)