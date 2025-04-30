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
nvtx_profile = False
ddp.nvtx_profile = nvtx_profile
nvtx_range = ddp.nvtx_range

memory_profile = False
jit_compile = False

if nvtx_profile:
    # only replace implementations if nvtx_profile is True
    model.scaled_dot_product_attention = model.annotated_scaled_dot_product_attention
    model.CausalMultiHeadSelfAttention.forward = model.CausalMultiHeadSelfAttention.annotated_forward

MODEL_SIZES = {
    "small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072},
    "medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096},
    "large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120},
    "xl": {"d_model": 1600, "num_layers": 48, "num_heads": 25, "d_ff": 6400},
    "2.7B": {"d_model": 2560, "num_layers": 32, "num_heads": 32, "d_ff": 10240},
}

def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)
    # random.seed(0)

def get_sharded_batch(x_data, y_data, batch_size, n_procs, rank):
    # shard data across processes
    shard_size = int(batch_size / n_procs)
    print(f"shard_size: {shard_size}")

    # check that it divides evenly
    assert batch_size % n_procs == 0

    # shard data
    x_shard = x_data[rank * shard_size:(rank + 1) * shard_size]
    y_shard = y_data[rank * shard_size:(rank + 1) * shard_size]

    return x_shard, y_shard

def benchmark_loop(model_params: dict, optimizer_params: dict, train_params: dict, dataset: np.array, warmup: int, n_steps: int, model_size: str, train_flag: str = "vanilla"):
    memory_flag = "full" if train_params["include_backward"] else "forward" if memory_profile else None
    device = train_params["device"] # note this will be overridden by the device in the trainer

    x_data, y_data = data.get_batch(dataset, batch_size = train_params["batch_size"], context_length = model_params["context_length"], device = device)

    # initialize trainer and data
    if train_flag == "vanilla":
        trainer = ddp.VanillaTrainer(device, model_params, optimizer_params, 
                                    mixed_precision = train_params["mixed_precision"], 
                                    amp_dtype = train_params["amp_dtype"],
                                    memory_flag = memory_flag,
                                    jit_compile = jit_compile)
    
    elif train_flag == "naive":
        # if naive, "backward time" includes only gradient syncing time
        trainer = ddp.NaiveDDPTrainer(device, model_params, optimizer_params, 
                                    n_procs = train_params["n_procs"],
                                    backend = "nccl",
                                    jit_compile = jit_compile)
        x_data, y_data = get_sharded_batch(x_data, y_data, train_params["batch_size"], train_params["n_procs"], trainer.rank)
        x_data = x_data.to(trainer.device)
        y_data = y_data.to(trainer.device)
    
    elif train_flag == "flattened":
        trainer = ddp.FlattenedDDPTrainer(device, model_params, optimizer_params, 
                                    n_procs = train_params["n_procs"],
                                    backend = "nccl",
                                    jit_compile = jit_compile)
        x_data, y_data = get_sharded_batch(x_data, y_data, train_params["batch_size"], train_params["n_procs"], trainer.rank)
        x_data = x_data.to(trainer.device)
        y_data = y_data.to(trainer.device)
        
    elif train_flag == "simplest":
        trainer = ddp.SimplestTrainer(device, model_params, optimizer_params, jit_compile = jit_compile)
        
    else:
        raise ValueError(f"Invalid training flag: {train_flag}")
    
    # training and benchmarking loop
    all_times = []
    
    print(f"training for {n_steps} steps")
    for i in range(n_steps + warmup):
        print(f"training step {i}")
        times = trainer.training_step(x_data, y_data)
        if i > warmup:
            all_times.append(times)
    
    # get final model for comparison
    if train_flag == "naive" or train_flag == "flattened":
        # synchronize all processes
        torch.distributed.barrier()

    final_model = trainer.model

    return final_model, all_times
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--include_backward", default=False, action="store_true")
    parser.add_argument("--model_size", type=str, default="all")
    parser.add_argument("--mixed_precision", default=False, action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16")
    parser.add_argument("--n_procs", type=int, default=1)
    
    args, _ = parser.parse_known_args()
    return args

def get_default_dicts(args):
    adamw_params = {
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }

    # check if already a torch dtype
    if isinstance(args.amp_dtype, torch.dtype):
        pass
    elif args.amp_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == "float16":
        args.amp_dtype = torch.float16
    else:
        raise ValueError(f"Invalid AMP dtype: {args.amp_dtype}")

    train_params = {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "include_backward": args.include_backward,
        "mixed_precision": args.mixed_precision,
        "amp_dtype": args.amp_dtype,
        "n_procs": args.n_procs,
    }

    return adamw_params, train_params

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
        cpu_final_model = final_model.cpu()
        shared["final_model"] = cpu_final_model
        shared["all_times"] = all_times
    
    dist.destroy_process_group()
    
    
def run_train(args, train_flag = "vanilla"):
    print('running run_train with train_flag:', train_flag, 'and n_procs:', args.n_procs)
    adamw_params, train_params = get_default_dicts(args)
    dataset = np.random.randint(0, args.vocab_size, size = args.context_length * 10)

    if train_flag == "naive" or train_flag == "flattened":
        shared = mp.Manager().dict()
        mp.spawn(ddp_worker, args = (args, adamw_params, train_params, dataset, shared, train_flag), nprocs = args.n_procs, join = True)
        final_model = shared["final_model"]
        all_times = shared["all_times"]
    else:
        final_model, all_times = launch_model(args, adamw_params, train_params, dataset, train_flag)
    
    # save all_times as list of tuples (time_per_step, collect_time)
    time_per_step = [time[0] for time in all_times]
    collect_times = [time[1] for time in all_times]

    print(f"collect_time: {np.mean(collect_times)}, time_per_step: {np.mean(time_per_step)}")

    return collect_times, time_per_step, dataset, final_model

def test_ddp(args, compare_weights = False, train_flag = "flattened", comparison_flag = "simplest"):
    print('running test_ddp')
    set_seeds()
    collect_times_naive, time_per_step_naive, data_naive, naive_lm = run_train(args, train_flag = train_flag)
    print('finished training')

    if compare_weights:
        set_seeds()
        collect_times_vanilla, time_per_step_vanilla, data_vanilla, vanilla_lm = run_train(args, train_flag = comparison_flag)
        vanilla_lm = vanilla_lm.cpu()
        
        # check same data, print with fun checkmark
        assert np.allclose(data_vanilla, data_naive)
        print("✅ same data")

        # check same weights
        all_match = True
        for i, (param, naive_param) in enumerate(zip(vanilla_lm.parameters(), naive_lm.parameters())):
            if not torch.allclose(param, naive_param, rtol=1e-4, atol=1e-5):
                diff = (param - naive_param).abs()
                print(f"❌ Parameter {i} mismatch:")
                print(f"   Max diff: {diff.max().item()}")
                print(f"   Mean diff: {diff.mean().item()}")
                print(f"   Vanilla param stats: min={param.min().item()}, max={param.max().item()}")
                print(f"   DDP param stats: min={naive_param.min().item()}, max={naive_param.max().item()}")
                all_match = False
    
        if all_match:
            print("✅ same weights")
        else:
            print("❌ weights differ")

        del vanilla_lm
    

    results = pd.DataFrame({
            "batch_size": [args.batch_size],
            "world_size": [args.n_procs],
            "avg. collection time (ms)": [np.mean(collect_times_naive) * 1000],
            "avg. time per step (ms)": [np.mean(time_per_step_naive) * 1000]
    })
    
    # append to existing file if it exists
    file_name = f"{train_flag}_ddp_benchmark.csv"
    try:
        existing = pd.read_csv(file_name)
        results = pd.concat([existing, results], ignore_index=True)
        # sort on data_size, then world_size (# of processes)
        results = results.sort_values(by=["batch_size", "world_size"])
    except FileNotFoundError:
        pass
        
    results.to_csv(file_name, index=False)
    print(f"Finished for batch_size={args.batch_size}, world_size={args.n_procs}")

    del naive_lm

    
if __name__ == "__main__":
    # set all random seeds
    args = parse_args()
    args.n_procs = 2
    args.model_size = "xl"
    for batch_size in [2, 4, 8]:
        args.batch_size = batch_size
        test_ddp(args, False, "flattened", "simplest")