import submitit
import benchmark
import sys

def memory_benchmark(context_length: int, include_backward: bool, include_adam: bool, model_size: str, mixed_precision: bool):
    args = benchmark.parse_args()
    args.context_length = context_length
    args.include_backward = include_backward
    args.include_adam = include_adam
    args.model_size = model_size
    args.mixed_precision = mixed_precision
    benchmark.main(args, save_times = False)

if __name__ == "__main__":
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
