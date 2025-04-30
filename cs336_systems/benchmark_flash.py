import torch
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attn import TritonAttention, TorchAttention
import pandas as pd
import timeit
import einx
import triton
import triton.testing

BATCH_SIZE = 1
NUM_HEADS = 1
jit_compile = False

def reset(params):
    # should I be zeroing out the gradients here?
    for param in params:
        param.grad.zero_()
        param.requires_grad = True
    torch.cuda.empty_cache()

def benchmark(implementation: str, d_model: int, seq_len: int, device: torch.device, precision: torch.dtype, warmup: int = 10, steps: int = 20):
    if implementation == "pytorch":
        if jit_compile:
            attention = torch.compile(scaled_dot_product_attention)
        else:
            attention = scaled_dot_product_attention
        
        # create causal mask
        seq = torch.arange(seq_len, device=device, dtype=precision)
        qi = einx.rearrange('query -> 1 1 query 1', seq)
        kj = einx.rearrange('key   -> 1 1 1   key', seq)
        causal_mask = qi >= kj  # (query, key)

    elif implementation == "triton":
        attention = torch.compile(TritonAttention.apply)
        causal_mask = True
    
    elif implementation == "torchflash":
        attention = torch.compile(TorchAttention.apply)
        causal_mask = True
    
    else:
        raise ValueError(f"Invalid implementation: {implementation}")
    
    # create random inputs
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device=device, requires_grad=True, dtype=precision)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device=device, requires_grad=True, dtype=precision)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device=device, requires_grad=True, dtype=precision)

    def forward_fn():
        with torch.no_grad():
            return attention(Q, K, V, causal_mask)
            
    def backward_fn():
        output = attention(Q, K, V, causal_mask)
        gradient = torch.ones_like(output, device=device, dtype=precision)
        output.backward(gradient)
        reset([Q, K, V])
        return output
    
    def full_fn():
        output = attention(Q, K, V, causal_mask)
        gradient = torch.ones_like(output, device=device, dtype=precision)
        output.backward(gradient)
        reset([Q, K, V])
        return output
    
    
    # use triton.testing.do_bench for benchmarking
    try:
        print("forward pass")
        forward_time = triton.testing.do_bench(forward_fn, warmup=warmup, rep=steps)
        
        # measure memory usage before backward pass
        with torch.no_grad():
            _ = attention(Q, K, V, causal_mask)
        pre_back_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # get it in MB

        print("backward pass")
        backward_time = triton.testing.do_bench(backward_fn, warmup=warmup, rep=steps)
        
        print("end to end pass")
        full_time = triton.testing.do_bench(full_fn, warmup=warmup, rep=steps)
        
    except Exception as e:
        print(f"Benchmarking error: {e}")
        return forward_time, None, None, pre_back_memory
    
    return forward_time, backward_time, full_time, pre_back_memory

def main():
    implementation = "pytorch"
    device = "cuda"
    seq_lens = list(2 ** torch.arange(7, 17))
    d_models = list(2 ** torch.arange(4, 8))
    precisions = [torch.bfloat16, torch.float32]

    results = pd.DataFrame(columns=['d_model', 'seq_len', 'precision', 'forward_time', 'backward_time', 'full_time', 'pre_back_memory'])

    for d_model in d_models:
        for seq_len in seq_lens:
            for precision in precisions:
                print(f"d_model={d_model}, seq_len={seq_len}, precision={precision}")
                try:
                    forward_time, backward_time, full_time, pre_back_memory = benchmark(implementation = implementation, d_model = d_model, device = device, seq_len = seq_len, precision = precision)
                    results.loc[len(results)] = {'d_model': d_model, 'seq_len': seq_len, 'precision': precision, 'forward_time': forward_time, 'backward_time': backward_time, 'full_time': full_time, 'pre_back_memory': pre_back_memory}
                except Exception as e:
                    print(f"Error benchmarking d_model={d_model}, seq_len={seq_len}: {e}")
                    results.loc[len(results)] = {'d_model': d_model, 'seq_len': seq_len, 'forward_time': None, 'backward_time': None, 'pre_back_memory': None}
                    continue
    
    # save results as markdown table
    # remove index column
    results = results.reset_index(drop=True)
    print(results.to_markdown())
    
    # save markdown table to file
    with open(f'{implementation}_attn.txt', 'w') as f:
        f.write(results.to_markdown())

if __name__ == "__main__":
    main()
