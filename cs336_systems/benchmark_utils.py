import argparse
import torch

MODEL_SIZES = {
    "tiniest": {"d_model": 4, "num_layers": 1, "num_heads": 1, "d_ff": 16},
    "tiny": {"d_model": 64, "num_layers": 12, "num_heads": 4, "d_ff": 256},
    "small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072},
    "medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096},
    "large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120},
    "xl": {"d_model": 1600, "num_layers": 48, "num_heads": 25, "d_ff": 6400},
    "2.7B": {"d_model": 2560, "num_layers": 32, "num_heads": 32, "d_ff": 10240},
}

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
    parser.add_argument("--bucket_size_mb", type=float, default=100)
    parser.add_argument("--shard_optimizer", default=False, action="store_true")

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
        "bucket_size_mb": args.bucket_size_mb,
        "shard_optimizer": args.shard_optimizer,
    }

    return adamw_params, train_params