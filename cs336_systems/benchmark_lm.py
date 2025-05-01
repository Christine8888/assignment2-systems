import cs336_basics.model as model
import argparse
import torch
import cs336_basics.optimizer as optimizer
import cs336_basics.data as data
import numpy as np
import timeit
import pandas as pd
import torch.cuda.nvtx as nvtx
import cs336_systems.ddp as ddp
import random
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from benchmark_utils import parse_args, get_default_dicts, MODEL_SIZES, get_sharded_batch
import cs336_systems.optimizer_sharding as optimizer_sharding

nvtx_profile = False
ddp.nvtx_profile = nvtx_profile
nvtx_range = ddp.nvtx_range

memory_profile = True
jit_compile = False
torch_profile = False

DDP_FLAGS = ["naive", "flattened", "overlap", "bucket"]

# if nvtx_profile:
#    # only replace implementations if nvtx_profile is True
#    model.scaled_dot_product_attention = model.annotated_scaled_dot_product_attention
#    model.CausalMultiHeadSelfAttention.forward = model.CausalMultiHeadSelfAttention.annotated_forward

def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    # random.seed(0)

def benchmark_loop(model_params: dict, optimizer_params: dict, train_params: dict, dataset: np.array, warmup: int, n_steps: int, model_size: str, train_flag: str = "vanilla"):
    memory_flag = "full" if train_params["include_backward"] else "forward" if memory_profile else None
    device = train_params["device"] # note this will be overridden by the device in the trainer

    x_data, y_data = data.get_batch(dataset, batch_size = train_params["batch_size"], context_length = model_params["context_length"], device = device)
    lm = model.BasicsTransformerLM(**model_params).to(device)

    # shard optimizer if specified
    optim = optimizer.AdamW(lm.parameters(), **optimizer_params)

    # initialize trainer and data
    if train_flag == "vanilla":
        trainer = ddp.VanillaTrainer(device, lm, optim, jit_compile = jit_compile)
    
    elif train_flag == "naive":
        trainer = ddp.NaiveDDPTrainer(device, lm, optim, n_procs = train_params["n_procs"], backend = "nccl", jit_compile = jit_compile)
    
    elif train_flag == "flattened":
        trainer = ddp.FlattenedDDPTrainer(device, lm, optim, n_procs = train_params["n_procs"], backend = "nccl", jit_compile = jit_compile)
    
    elif train_flag == "overlap":
        trainer = ddp.OverlapDDPTrainer(device, lm, optim, n_procs = train_params["n_procs"], backend = "nccl", jit_compile = jit_compile)

    elif train_flag == "bucket":
        trainer = ddp.BucketDDPTrainer(device, lm, optim, n_procs = train_params["n_procs"], bucket_size_mb = train_params["bucket_size_mb"], backend = "nccl", jit_compile = jit_compile)

    elif train_flag == "simplest":
        trainer = ddp.SimplestTrainer(device, lm, optim, jit_compile = jit_compile)
        
    else:
        raise ValueError(f"Invalid training flag: {train_flag}")
    
    if train_params["shard_optimizer"]:
        trainer.optimizer = optimizer_sharding.ShardedOptimizer(lm.parameters(), optimizer.AdamW, **optimizer_params)
    
    if train_flag in DDP_FLAGS:
        x_data, y_data = get_sharded_batch(x_data, y_data, train_params["batch_size"], train_params["n_procs"], trainer.rank)
        x_data = x_data.to(trainer.device)
        y_data = y_data.to(trainer.device)

    # training and benchmarking loop
    all_times = []
    print(f"training for {n_steps} steps")
    for i in range(n_steps + warmup):
        if memory_flag == "full":
            torch.cuda.memory._record_memory_history(enabled=False)
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        times = trainer.training_step(x_data, y_data)

        if memory_flag == "full":
            torch.cuda.memory._dump_snapshot(f"{model_size}_sharding_{train_params["shard_optimizer"]}_{train_flag}.pickle")
            torch.cuda.memory._record_memory_history(enabled=False)
        
        if i > warmup:
            all_times.append(times)

    dist.barrier()
    
    # get final model for comparison
    if train_flag in DDP_FLAGS:
        # synchronize all processes
        torch.distributed.barrier()

    final_model = trainer.model

    return final_model, all_times


def launch_model(args, adamw_params, train_params, dataset, train_flag = "vanilla"):
    model_size = train_params["model_size"]
    print(f"benchmarking {train_params['model_size']}...")

    # vary only d_model, num_layers, num_heads, and d_ff
    model_params = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": MODEL_SIZES[model_size]["d_model"],
        "num_layers": MODEL_SIZES[model_size]["num_layers"],
        "num_heads": MODEL_SIZES[model_size]["num_heads"],
        "d_ff": MODEL_SIZES[model_size]["d_ff"],
        "rope_theta": args.rope_theta,
    }

    with nvtx_range(model_size):
        lm, all_times = benchmark_loop(model_params = model_params, optimizer_params = adamw_params, 
                                        train_params = train_params, dataset = dataset, warmup = args.warmup, 
                                        n_steps = args.n_steps, model_size = model_size, train_flag = train_flag)
    
    return lm, all_times

def run_time(args, dataset, save_times = False, to_save = ["forward_mean", "forward_std", "backward_mean", "backward_std"]):
    """
    Run forward/backward pass timings for a given model size and save the results to a csv file.
    """

    adamw_params, train_params = get_default_dicts(args)

    # create pandas dataframe to store results for each model size
    results = pd.DataFrame(index = MODEL_SIZES.keys(), columns = to_save)

    if args.model_size == "all":
        model_sizes = MODEL_SIZES.keys()
    else:
        model_sizes = [args.model_size]

    for model_size in model_sizes:
        lm, all_times = launch_model(args, adamw_params, train_params, dataset, train_flag = "vanilla")
        # all times: list of tuples (forward_time, backward_time, full_step_time)
        # take and save mean and std of forward_time, backward_time, and full_step_time
        forward_times = [time[0] for time in all_times]
        backward_times = [time[1] for time in all_times]
        full_times = [time[2] for time in all_times]

        results.loc[model_size, "forward_mean"] = np.mean(forward_times)
        results.loc[model_size, "forward_std"] = np.std(forward_times)
        results.loc[model_size, "backward_mean"] = np.mean(backward_times)
        results.loc[model_size, "backward_std"] = np.std(backward_times)
    
    print(results.to_csv())

    # save results to txt as markdown table
    if save_times:
        with open(f"warmup_{args.warmup}_backward_{args.include_backward}_{'jit' if jit_compile else 'nojit'}.txt", "w") as f:
            f.write(results.to_csv())
    
    return lm


def ddp_worker(rank, args, adamw_params, train_params, dataset, shared, train_flag):
    print('running ddp_worker with train_flag:', train_flag, 'and n_procs:', args.n_procs)
    os.environ["LOCAL_RANK"] = str(rank)
    
    # must set seeds here, otherwise will get different results on different processes
    set_seeds()
    
    # run training loop
    final_model, all_times = launch_model(args = args, adamw_params = adamw_params, train_params = train_params, dataset = dataset, train_flag = train_flag)

    # only let rank 0 return the real payload
    if rank == 0:
        shared["final_model"] = [p.detach().cpu() for p in final_model.parameters()]
    
        # append all_times to shared["all_times"]
        current_list = shared["all_times"]
        current_list.extend(all_times)
        shared["all_times"] = current_list
    
    dist.destroy_process_group()
    
    
def run_train(args, train_flag = "vanilla"):
    """
    Run training loop for a given model and distributed training strategy, and save the results to a CSV file.
    """

    print('running run_train with train_flag:', train_flag, 'and n_procs:', args.n_procs)

    adamw_params, train_params = get_default_dicts(args)
    dataset = np.random.randint(0, args.vocab_size, size = args.context_length * 10)

    if train_flag in DDP_FLAGS:
        shared = mp.Manager().dict()
        shared["all_times"] = []
        mp.spawn(ddp_worker, args = (args, adamw_params, train_params, dataset, shared, train_flag), nprocs = args.n_procs, join = True)
        final_model = shared["final_model"]
        all_times = shared["all_times"]
    else:
        # only spawn one process
        final_model, all_times = launch_model(args, adamw_params, train_params, dataset, train_flag)
    
    # save all_times as list of tuples (time_per_step, collect_time)
    time_per_step = [time[0] for time in all_times]
    collect_times = [time[1] for time in all_times]

    print(f"collect_time: {np.mean(collect_times)}, time_per_step: {np.mean(time_per_step)}")

    return collect_times, time_per_step, dataset, final_model

def compare_weights(vanilla_parameters, ddp_parameters):
    """
    Compare the weights of two models.
    """
    all_match = True
    for i, (vanilla_param, ddp_param) in enumerate(zip(vanilla_parameters, ddp_parameters)):
        if not torch.allclose(vanilla_param, ddp_param, rtol=1e-4, atol=1e-5):
            diff = (vanilla_param - ddp_param).abs()
            print(f"❌ Parameter {i} mismatch:")
            print(f"   Max diff: {diff.max().item()}")
            print(f"   Mean diff: {diff.mean().item()}")
            print(f"   Vanilla param stats: min={vanilla_param.min().item()}, max={vanilla_param.max().item()}")
            print(f"   DDP param stats: min={ddp_param.min().item()}, max={ddp_param.max().item()}")
            all_match = False
    
    return all_match

def test_ddp(args, test_weights = False, train_flag = "flattened", comparison_flag = "simplest"):
    """Test DDP training accuracy by parameter comparison to a non-DDP training run."""
    set_seeds()
    collect_times_ddp, time_per_step_ddp, data_ddp, ddp_parameters = run_train(args, train_flag = train_flag)

    if test_weights:
        set_seeds()
        collect_times_vanilla, time_per_step_vanilla, data_vanilla, vanilla_parameters = run_train(args, train_flag = comparison_flag)
        
        # check same data, print with fun checkmark
        assert np.allclose(data_vanilla, data_ddp)
        print("✅ same data")

        # check same weights
        all_match = compare_weights(vanilla_parameters, ddp_parameters)
        if all_match:
            print("✅ same weights")
        else:
            print("❌ weights differ")

        del vanilla_parameters

    # save results to CSV
    results = pd.DataFrame({
            "bucket_size_mb": [args.bucket_size_mb],
            #"batch_size": [args.batch_size],
            "world_size": [args.n_procs],
            "avg. collection time (ms)": [np.mean(collect_times_ddp) * 1000],
            "avg. time per step (ms)": [np.mean(time_per_step_ddp) * 1000]
    })
    
    # append to existing file if it exists
    file_name = f"{train_flag}_ddp_benchmark.csv"
    try:
        existing = pd.read_csv(file_name)
        results = pd.concat([existing, results], ignore_index=True)
        # sort on data_size, then world_size (# of processes)
        # results = results.sort_values(by=["batch_size", "world_size"])
        results = results.sort_values(by=["bucket_size_mb", "world_size"])
    except FileNotFoundError:
        pass
        
    results.to_csv(file_name, index=False)
    print(f"Finished for batch_size={args.batch_size}, world_size={args.n_procs}")

    del ddp_parameters

    
if __name__ == "__main__":
    # set all random seeds
    args = parse_args()
    args.model_size = "large"
    args.n_steps = 10
    args.n_procs = 2
    args.batch_size = 4
    args.include_backward = True

    args.shard_optimizer = True
    # run basic training
    run_train(args, train_flag = "naive")

    # result is just never correct regardless of bucket size
    # if n_procs is 1, result is correct
    # test_ddp(args, True, "overlap", "naive")

    # code for running benchmarking
    # for batch_size in [2, 4, 8, 16, 32]:
    #     args.batch_size = batch_size
    #     test_ddp(args, False, "overlap")

    # args.bucket_size_mb = 10
    # test_ddp(args, False, "bucket")