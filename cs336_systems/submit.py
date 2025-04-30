import submitit
import cs336_systems.benchmark_lm as benchmark_lm
import sys
from benchmark_comm import benchmark_comms
import torch

def memory_benchmark(context_length: int, include_backward: bool, include_adam: bool, model_size: str, mixed_precision: bool):
    args = benchmark_lm.parse_args()
    args.context_length = context_length
    args.include_backward = include_backward
    args.include_adam = include_adam
    args.model_size = model_size
    args.mixed_precision = mixed_precision
    benchmark_lm.run_time(args, save_times = False)


def submit_memory_benchmark():
    executor = submitit.AutoExecutor(folder="memory_results")
    executor.update_parameters(name = "memory_benchmark", timeout_min=5, gpus_per_node=1, slurm_partition = "a2", slurm_qos = "a2-qos", slurm_array_parallelism = 6)

    # run the benchmark for each context length and include_backward value
    context_lengths = [128, 256]
    include_backward = [True, False]
    include_adam = True
    mixed_precision = True
    model_size = "2.7B"

    for context_length in context_lengths:
        for val in include_backward:
            job = executor.submit(memory_benchmark, context_length, val, include_adam, model_size, mixed_precision)
            print(f"Submitted job for context length {context_length} and include_backward {val}")


def submit_comm_benchmark():
    executor = submitit.AutoExecutor(folder="comm_results")
    executor.update_parameters(name = "comm_benchmark", slurm_gpus_per_task=6, slurm_partition = "a2", slurm_qos = "a2-qos")
    n_warmup = 5
    backend = "nccl"
    n_steps = 20

    job = executor.submit(benchmark_comms, backend, n_warmup, n_steps)

def submit_naive_ddp_benchmark():
    executor = submitit.AutoExecutor(folder="naive_ddp_results")
    args = benchmark_lm.parse_args()
    args.n_procs = 2
    args.warmup = 5
    args.n_steps = 20
    args.model_size = "xl"

    executor.update_parameters(name = "naive_ddp_benchmark", slurm_gpus_per_task=args.n_procs, slurm_partition = "a2", slurm_qos = "a2-qos")

    for batch_size in [4, 8]:
        args.batch_size = batch_size
        job = executor.submit(benchmark_lm.test_ddp, args, test_weights = False, train_flag = "naive")
        print(f"Submitted job for batch size {batch_size}")


if __name__ == "__main__":
    submit_naive_ddp_benchmark()