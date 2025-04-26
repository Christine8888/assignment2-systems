import cs336_basics.model as model
import argparse
import torch
import cs336_basics.optimizer as optimizer
import cs336_basics.nn_utils as nn_utils
import cs336_basics.data as data
import numpy as np
import timeit
import pandas as pd
import torch.cuda.nvtx as nvtx
import einops
from contextlib import nullcontext

nvtx_profile = False
memory_profile = True

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

def nvtx_range(name):
    if nvtx_profile:
        return nvtx.range(name)
    else:
        return nullcontext()

def benchmark_model(model_params: dict, optimizer_params: dict, train_params: dict, dataset: np.array, warmup: int, n_steps: int, model_size: str):
    device = train_params["device"]
    lm = model.BasicsTransformerLM(**model_params).to(device)
    optim = optimizer.AdamW(lm.parameters(), **optimizer_params)
    use_amp = train_params["mixed_precision"]

    if use_amp:
        cast_context = torch.autocast(device_type = device, dtype = train_params["amp_dtype"])
    else:
        cast_context = nullcontext()
    
    # no cosine lr scheduling or gradient clipping for the benchmarking
    x_data, y_data = data.get_batch(dataset, batch_size = train_params["batch_size"], context_length = model_params["context_length"], device = device)
    
    for _ in range(warmup):
        with cast_context:
            logits = lm(x_data)
            torch.cuda.synchronize()
            loss = nn_utils.cross_entropy(logits, y_data)
            loss.backward()
        torch.cuda.synchronize()
    
    times = np.zeros(n_steps)

    with nvtx_range("benchmark steps"):
        for i in range(n_steps):
            torch.cuda.memory._record_memory_history(enabled=False)
            start_time = timeit.default_timer()

            with cast_context:
                # forward pass
                if memory_profile:
                    torch.cuda.memory._record_memory_history(max_entries=1000000)

                with nvtx_range("forward pass"):
                    logits = lm(x_data)
                    torch.cuda.synchronize()

                if memory_profile and not train_params["include_backward"]: # end here for forward pass
                    torch.cuda.memory._dump_snapshot(f"{model_size}_forward_{model_params['context_length']}_{train_params['mixed_precision']}.pickle")
                    torch.cuda.memory._record_memory_history(enabled=False)
                
                end_time = timeit.default_timer()

                # backwards pass
                with nvtx_range("backwards pass"):
                    loss = nn_utils.cross_entropy(logits, y_data)
                    loss.backward()
                    torch.cuda.synchronize()

            if train_params["include_backward"]:
                end_time = timeit.default_timer()
            
            # optimizer step
            if train_params["include_adam"]:
                with nvtx_range("optimizer step"):
                    optim.step()
                    optim.zero_grad(set_to_none = True)
            
            if memory_profile and train_params["include_backward"]: # end here for full step
                torch.cuda.memory._dump_snapshot(f"{model_size}_fullstep_{model_params['context_length']}_{train_params['mixed_precision']}.pickle")
                torch.cuda.memory._record_memory_history(enabled=False)

            times[i] = end_time - start_time
    
    return times
    
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
    parser.add_argument("--include_adam", default=False, action="store_true")
    parser.add_argument("--model_size", type=str, default="all")
    parser.add_argument("--mixed_precision", default=False, action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16")
    
    args, _ = parser.parse_known_args()
    return args

def get_default_dicts(args):
    adamw_params = {
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }

    if args.amp_dtype == "bfloat16":
        args.amp_dtype = torch.bfloat16
    elif args.amp_dtype == "float16":
        args.amp_dtype = torch.float16
    else:
        raise ValueError(f"Invalid AMP dtype: {args.amp_dtype}")

    train_params = {
        "batch_size": args.batch_size,
        "device": args.device,
        "include_backward": args.include_backward,
        "include_adam": args.include_adam,
        "mixed_precision": args.mixed_precision,
        "amp_dtype": args.amp_dtype,
    }

    return adamw_params, train_params

def run_benchmark(model_size, args, adamw_params, train_params, data):
    print(f"benchmarking {model_size}...")

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
        times = benchmark_model(model_params = model_params, optimizer_params = adamw_params, train_params = train_params, dataset = data, warmup = args.warmup, n_steps = args.n_steps, model_size = model_size)
    
    return [np.mean(times), np.std(times)]


def main(args, save_times = False):
    adamw_params, train_params = get_default_dicts(args)

    # generate random data
    data = np.random.randint(0, args.vocab_size, size = args.context_length * 10)

    # create pandas dataframe to store results for each model size
    results = pd.DataFrame(index = MODEL_SIZES.keys(), columns = ["time", "std"])

    if args.model_size == "all":
        model_sizes = MODEL_SIZES.keys()
    else:
        model_sizes = [args.model_size]

    for model_size in model_sizes:
        results.loc[model_size] = run_benchmark(model_size, args, adamw_params, train_params, data)
        
    # save results to txt as markdown table
    if save_times:
        with open(f"warmup_{args.warmup}_backward_{args.include_backward}.txt", "w") as f:
            print(results.to_markdown())
            f.write(results.to_markdown())

if __name__ == "__main__":
    main(parse_args(), save_times = False)