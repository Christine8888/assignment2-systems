import model
import argparse
import torch
import optimizer
import nn_utils
import data
import numpy as np
import timeit


def benchmark_model(model_params: dict, optimizer_params: dict, train_params: dict, dataset: np.array, warmup: int, n_steps: int):
    device = train_params["device"]
    lm = model.BasicTransformerLM(**model_params).to(device)
    optim = optimizer.AdamW(**optimizer_params)
    
    # no cosine lr scheduling or gradient clipping for the benchmarking
    x_data, y_data = data.get_batch(dataset, train_params["batch_size"], model_params["context_length"], device)

    for _ in range(warmup):
        logits = lm(x_data)
        loss = nn_utils.cross_entropy(logits, y_data)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none = True)
        torch.cuda.synchronize()
    
    times = np.zeros(n_steps)

    for i in range(n_steps):
        start_time = timeit.default_timer()
        logits = lm(x_data)
        
        end_time = timeit.default_timer()
        
        torch.cuda.synchronize()

        loss = nn_utils.cross_entropy(logits, y_data)
        loss.backward()
        optim.step()
        torch.cuda.synchronize()

        if train_params["include_backward"]:
            end_time = timeit.default_timer()
        
        optim.zero_grad(set_to_none = True)

        times[i] = end_time - start_time
    
    return times
    

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--rope_theta", type=float, default=10000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--include_backward", action="store_true")
    
    args = parser.parse_args()

    model_params = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.n_layers,
        "num_heads": args.n_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }

    adamw_params = {
        "lr": 1e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }

    train_params = {
        "batch_size": args.batch_size,
        "device": args.device,
    }

    # generate random data
    data = np.random.randint(0, args.vocab_size, size = (args.batch_size, args.seq_len))

    # benchmark the model
    times = benchmark_model(model_params, adamw_params, train_params, data, args.warmup, args.n_steps)

    print("Mean time per step: ", np.mean(times))
    print("STD: ", np.std(times))

    