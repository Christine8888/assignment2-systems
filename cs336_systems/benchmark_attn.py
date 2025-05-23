import torch
from cs336_systems.flash_attn import TritonAttention, TorchAttention
from cs336_basics.model import scaled_dot_product_attention
import pandas as pd
import timeit
import einx

BATCH_SIZE = 8
NUM_HEADS = 1
device = torch.device("cuda")
jit_compile = False

def benchmark(d_model: int, seq_len: int, warmup: int = 10, steps: int = 100):
    # create random inputs
    Q = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, NUM_HEADS, seq_len, d_model, device=device, requires_grad=True)

    # create causal mask
    seq = torch.arange(seq_len, device=device)
    qi = einx.rearrange('query -> 1 1 query 1', seq)
    kj = einx.rearrange('key   -> 1 1 1   key', seq)
    causal_mask = qi >= kj  # (query, key)

    # run warmup steps
    for _ in range(warmup):
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        torch.cuda.synchronize()

        gradient = torch.ones_like(attn_output, device=device)
        attn_output.backward(gradient)
        torch.cuda.synchronize()

        for param in [Q, K, V]:
            param.grad.zero_()
        
        torch.cuda.empty_cache()
    
    # record forward and backward times
    forward_time = 0
    backward_time = 0
    pre_back_memory = 0
    
    # run actual steps
    for _ in range(steps):
        # measure forward pass
        forward_start_time = timeit.default_timer()
        attn_output = scaled_dot_product_attention(Q, K, V, causal_mask)
        torch.cuda.synchronize()
        forward_end_time = timeit.default_timer()
        forward_time += (forward_end_time - forward_start_time)

        # measure memory usage before backward pass
        memory = torch.cuda.memory_allocated() / 1024**2 # get it in GB
        pre_back_memory += memory

        # measure backward pass
        backward_start_time = timeit.default_timer()
        gradient = torch.ones_like(attn_output, device=device)
        attn_output.backward(gradient)
        torch.cuda.synchronize()
        backward_end_time = timeit.default_timer()
        backward_time += (backward_end_time - backward_start_time)

        # reset gradients
        for param in [Q, K, V]:
            param.grad.zero_()
        
        torch.cuda.empty_cache()

    # take averages
    forward_time /= steps
    backward_time /= steps
    pre_back_memory /= steps
    
    return forward_time, backward_time, pre_back_memory



def main():
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    results = pd.DataFrame(columns=['d_model', 'seq_len', 'forward_time', 'backward_time', 'pre_back_memory'])

    for d_model in d_models:
        for seq_len in seq_lens:
            print(d_model, seq_len)
            try:
                forward_time, backward_time, pre_back_memory = benchmark(d_model, seq_len)
                results.loc[len(results)] = {'d_model': d_model, 'seq_len': seq_len, 'forward_time': forward_time, 'backward_time': backward_time, 'pre_back_memory': pre_back_memory}
            except Exception as e:
                print(f"Error benchmarking d_model={d_model}, seq_len={seq_len}: {e}")
                results.loc[len(results)] = {'d_model': d_model, 'seq_len': seq_len, 'forward_time': None, 'backward_time': None, 'pre_back_memory': None}
                continue
    
    # save results as markdown table
    # remove index column
    results = results.reset_index(drop=True)
    print(results.to_markdown())
    
    # save markdown table to file
    with open(f'pytorch_attn_{"jit" if jit_compile else "nojit"}.txt', 'w') as f:
        f.write(results.to_markdown())

if __name__ == "__main__":
    main()
    
    
