from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import torch.nn as nn
import timeit
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
import torch.distributed as dist
import os
import random
from torch.utils.hooks import unserializable_hook

nvtx_profile = True
verbose = False

def nvtx_range(name):
    if nvtx_profile:
        return nvtx.range(name)
    else:
        return nullcontext()

class BaseTrainer:
    def __init__(self, device, model, optimizer, jit_compile = True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.jit_compile = jit_compile
        if self.jit_compile: self.model = torch.compile(self.model)
        
    def init_model(self, model_params, device):
        self.model = BasicsTransformerLM(**model_params).to(device)
        if self.jit_compile: self.model = torch.compile(self.model)
    
    def init_optimizer(self, optimizer_params):
        self.optimizer = AdamW(self.model.parameters(), **optimizer_params)
        
    def training_step(self, x_data, y_data):
        """Base implementation of a single training step"""
        raise NotImplementedError("Subclasses must implement training_step")

class DDPOverlapWrapper(nn.Module):
    def __init__(self, module: torch.nn.Module):
        # inherit all methods and attributes of the module
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.handles = []

        for param in self.module.parameters():
            if param.requires_grad:
                # param.register_hook(self.make_hook(param))
                # grad hook passes through the *parameter* not the gradient
                param.register_post_accumulate_grad_hook(self.make_hook(param))
        
        # broadcast parameters across devices
        self.param_sync()

        if verbose: print(f"Initializing DDPOverlapWrapper for rank {self.rank} with world size {self.world_size}")
    
    def make_hook(self, param):
        @unserializable_hook
        def hook(param):
            # hook should operate on the *parameter* not the gradient
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            # handle.wait()
            self.handles.append(handle)
            # return grad
        
        return hook
    
    def param_sync(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src = 0)

    def forward(self, *inputs, **kwargs):
        """Just a wrapper to call the module's forward method"""
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        if verbose: print(f"Synchronizing gradients for rank {self.rank}")
        for handle in self.handles:
            if handle is not None:
                handle.wait()
        
        # scale gradients at the end of synchronizing
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.div_(self.world_size)
        
        self.handles.clear()

class DDPOverlapBucket(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float, run_async: bool = True):
        # set up parameters, dist properties
        super().__init__()
        self.module = module
        self.bucket_size = bucket_size_mb
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.run_async = run_async

        # setup variables
        self.buckets = []
        self.pending_in_bucket = []
        self.handles = []

        # setup functions
        self.param_sync() # broadcast parameters across devices
        self.assign_buckets() # assign parameters into buckets of size bucket_size_mb
        self.register_bucket_hooks() # register hooks for each bucket
    
    def param_sync(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src = 0)

    def assign_buckets(self):
        # assign parameters into buckets in reverse order of model.parameters()
        current_bucket = []
        current_size = 0

        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
            
            # get size of parameter in MB
            size_mb = param.numel() * param.element_size() / 1024**2
            
            if current_size + size_mb > self.bucket_size:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param)
            current_size += size_mb
        
        self.buckets.append(current_bucket)

        # register pending counters for new buckets
        for idx, bucket in enumerate(self.buckets):
            self.pending_in_bucket.append(0)
            
            for p in bucket:
                p.bucket_idx = idx
            
        if verbose: print(f'created {len(self.buckets)} buckets')
    
    def register_bucket_hooks(self):
        # register hooks for each bucket
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            
            # passes through parameter when gradient is accumulated
            p.register_post_accumulate_grad_hook(self.make_bucket_hook())
    
    def make_bucket_hook(self):
        @unserializable_hook
        def hook(param):
            bucket_idx = param.bucket_idx
            self.pending_in_bucket[bucket_idx] += 1

            if self.pending_in_bucket[bucket_idx] == len(self.buckets[bucket_idx]):
                self.pending_in_bucket[bucket_idx] = 0 # reset bucket
                self.reduce_bucket(bucket_idx) # reduce bucket
        
        return hook
    
    def reduce_bucket(self, bucket_idx):
        if verbose: print('reducing bucket', bucket_idx)
        
        # get bucket and parameters
        bucket = self.buckets[bucket_idx]
        source_grads = [param.grad for param in bucket if param.grad is not None]
        source_params = [param for param in bucket if param.grad is not None]

        if source_grads:
            # mark parameters as reduced
            for param in source_params:
                param.reduced = True
            
            # flatten gradients and all-reduce
            flattened = torch._utils._flatten_dense_tensors(source_grads)
            handle = dist.all_reduce(flattened, op=dist.ReduceOp.SUM, async_op = True)

            if not self.run_async:
                handle.wait()
                flattened.div_(self.world_size)
                unflattened = torch._utils._unflatten_dense_tensors(flattened, source_grads)

                for param, updated_grad in zip(source_params, unflattened):
                    param.grad.copy_(updated_grad)
            else:
                self.handles.append((handle, flattened, bucket_idx))

    def forward(self, *inputs, **kwargs):
        # mark parameters as not gradient-reduced
        for param in self.module.parameters():
            param.reduced = False

        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        # wait for all handles to finish
        # reshape and scale gradients
        # note we had to save a buffer of flattened gradients to do this
        
        if self.run_async:
            for handle, flattened, bucket_idx in self.handles:
                handle.wait()

                flattened.div_(self.world_size)
                bucket_grads = [p.grad for p in self.buckets[bucket_idx] if p.grad is not None]
                bucket_params = [p for p in self.buckets[bucket_idx] if p.grad is not None]
                unflattened = torch._utils._unflatten_dense_tensors(flattened, bucket_grads)

                for param, updated_grad in zip(bucket_params, unflattened):
                    param.grad.copy_(updated_grad)
                
                # free flattened, unflattened
                del flattened, unflattened
            
        # check that all parameters were reduced
        for param in self.module.parameters():
            if not param.reduced and param.grad is not None:
                print('param not reduced')
                param.reduced = False
        
        # delete handles
        del self.handles
        self.handles = []

class NaiveDDPTrainer(BaseTrainer):
    """Naive DDP trainer"""
    
    def __init__(self, device, model, optimizer, n_procs: int, rank = None, backend = "nccl", jit_compile = True):
        self.n_procs = n_procs
        self.jit_compile = jit_compile

        if rank is None:
            self.rank = int(os.environ.get("LOCAL_RANK", -1))
        else:
            self.rank = rank

        self.setup(self.rank, self.n_procs, backend)
        self.device = torch.device(f"cuda:{self.rank}")

        # make deep copy of model on this device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.jit_compile = jit_compile
        if self.jit_compile: self.model = torch.compile(self.model)
        self.param_sync()
    
    def setup(self, rank, world_size: int, backend: str = "nccl"):
        print(f"Setting up process {rank} with backend {backend}")

        # set up master address and port for multiprocessing
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        # master_port = os.environ.get("MASTER_PORT", str(random.randint(29500, 29600)))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)

        dist.barrier()
    
    def param_sync(self):
        # broadcast parameters and optimizer state
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                dist.broadcast(v, src=0)
        
        dist.barrier()

    def training_step(self, x_data, y_data):
        with nvtx_range("naive DDP training step"):
            # forward pass
            with nvtx_range("forward pass"):
                dist.barrier()
                full_start = timeit.default_timer()
                logits = self.model(x_data)
                torch.cuda.synchronize()
                
            # backward pass
            with nvtx_range("backward pass"):
                loss = cross_entropy(logits, y_data)
                loss.backward()
                torch.cuda.synchronize()
                dist.barrier()

            # sync gradients across processes
            with nvtx_range("gradient synchronization"):
                collect_start = timeit.default_timer()
                for param in self.model.parameters():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(self.n_procs)

                
                torch.cuda.synchronize()
                dist.barrier()
                collect_end = timeit.default_timer()

            # optimizer step
            with nvtx_range("optimizer step"):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                dist.barrier()
            
            full_step_end = timeit.default_timer()

        return full_step_end - full_start, collect_end - collect_start

class OverlapDDPTrainer(NaiveDDPTrainer):
    """Overlap DDP trainer"""
    def __init__(self, device, model, optimizer, n_procs: int, rank = None, backend = "nccl", jit_compile = True):
        super().__init__(device, model, optimizer, n_procs, rank, backend, jit_compile)
        self.model = DDPOverlapWrapper(self.model)

    def training_step(self, x_data, y_data):
        with nvtx_range("overlap DDP training step"):
            # forward pass
            with nvtx_range("forward pass"):
                full_start = timeit.default_timer()
                logits = self.model(x_data)
            
            # backward pass
            with nvtx_range("backward pass"):
                loss = cross_entropy(logits, y_data)
                loss.backward()

            # sync gradients across processes
            with nvtx_range("gradient synchronization"):
                self.model.finish_gradient_synchronization()
            
            dist.barrier()

            # optimizer step
            with nvtx_range("optimizer step"):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                full_step_end = timeit.default_timer()

            return full_step_end - full_start, 0

class BucketDDPTrainer(OverlapDDPTrainer):
    """Bucket DDP trainer"""
    def __init__(self, device, model, optimizer, n_procs: int, bucket_size_mb: float, rank = None, backend = "nccl", jit_compile = True):
        super().__init__(device, model, optimizer, n_procs, rank, backend, jit_compile)
        self.model = DDPOverlapBucket(self.model, bucket_size_mb)

class FlattenedDDPTrainer(NaiveDDPTrainer):
    """Flattened DDP trainer"""
    # def __init__(self, device, model_params, optimizer_params, n_procs: int, rank = None, backend = "nccl", jit_compile = True):
    #     super().__init__(device, model_params, optimizer_params, n_procs, rank, backend, jit_compile)
    
    def __init__(self, device, model, optimizer, n_procs: int, rank = None, backend = "nccl", jit_compile = True):
        super().__init__(device, model, optimizer, n_procs, rank, backend, jit_compile)

    def training_step(self, x_data, y_data):
        # forward pass
        full_start = timeit.default_timer()
        logits = self.model(x_data)
        torch.cuda.synchronize()
        
        # backward pass
        loss = cross_entropy(logits, y_data)
        loss.backward()
        torch.cuda.synchronize()

        # sync gradients across processes, with flattening
        collect_start = timeit.default_timer()
        source_grads = [param.grad for param in self.model.parameters() if param.grad is not None]
        source_params = [param for param in self.model.parameters() if param.grad is not None]
        flattened = torch._utils._flatten_dense_tensors(source_grads)
        dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
        flattened.div_(self.n_procs)
        
        # update with unflattened gradients
        unflattened = torch._utils._unflatten_dense_tensors(flattened, source_grads)
        for param, unflattened_grad in zip(source_params, unflattened):
            param.grad = unflattened_grad
            
        torch.cuda.synchronize()
        dist.barrier()
        collect_end = timeit.default_timer()

        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        full_step_end = timeit.default_timer()

        return full_step_end - full_start, collect_end - collect_start


class SimplestTrainer(BaseTrainer):
    """Sanity check simple trainer"""
    # def __init__(self, device, model_params, optimizer_params, jit_compile = True):
    #     super().__init__(device, model_params, optimizer_params, jit_compile)

    def __init__(self, device, model, optimizer, jit_compile = True):
        super().__init__(device, model, optimizer, jit_compile)

    def training_step(self, x_data, y_data):
        forward_start = timeit.default_timer()
        logits = self.model(x_data)
        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        
        backward_start = timeit.default_timer()
        loss = cross_entropy(logits, y_data)
        loss.backward()
        torch.cuda.synchronize()
        backward_end = timeit.default_timer()

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        full_step_end = timeit.default_timer()

        return forward_end - forward_start, backward_end - backward_start, full_step_end - forward_start

# vanilla training step with profiling
class VanillaTrainer(BaseTrainer):
    # def __init__(self, device, model_params, optimizer_params, mixed_precision = False, amp_dtype = torch.bfloat16, memory_flag = None, jit_compile = True):
    #     super().__init__(device, model_params, optimizer_params, jit_compile)
    #     self.mixed_precision = mixed_precision
    #     self.amp_dtype = amp_dtype
    #     self.memory_flag = memory_flag

    def __init__(self, device, model, optimizer, mixed_precision = False, amp_dtype = torch.bfloat16, memory_flag = None, jit_compile = True):
        super().__init__(device, model, optimizer, jit_compile)
        self.mixed_precision = mixed_precision
        self.amp_dtype = amp_dtype
        self.memory_flag = memory_flag
    
    def get_context(self):
        """Get the appropriate casting context for mixed precision"""
        if self.mixed_precision:
            return torch.autocast(device_type=self.device, dtype=self.amp_dtype)
        else:
            return nullcontext()

    def training_step(self, x_data, y_data):
        with nvtx_range("standard training step"):
            cast_context = self.get_context()
            
            if self.memory_flag:
                torch.cuda.memory._record_memory_history(enabled=False)
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            with cast_context:
                # forward pass
                forward_start = timeit.default_timer()
                with nvtx_range("forward pass"):
                    logits = self.model(x_data)
                    torch.cuda.synchronize()

                forward_end = timeit.default_timer()

                if self.memory_flag == "forward":
                    torch.cuda.memory._dump_snapshot(f"{self.model.model_size}_forward_{self.model.context_length}_{self.mixed_precision}.pickle")
                    torch.cuda.memory._record_memory_history(enabled=False)

                # backward pass
                backward_start = timeit.default_timer()
                with nvtx_range("backwards pass"):
                    loss = cross_entropy(logits, y_data)
                    loss.backward()
                    torch.cuda.synchronize()

                backward_end = timeit.default_timer()
            
            # # print grad on attention weights
            # for i, param in enumerate(self.model.parameters()):
            #     if i == 10:
            #         print(f"VanillaTrainer grad: {param.grad}")

            # optimizer step
            with nvtx_range("optimizer step"):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            if self.memory_flag == "full":
                torch.cuda.memory._dump_snapshot(f"{self.model.model_size}_fullstep_{self.model.context_length}_{self.mixed_precision}.pickle")
                torch.cuda.memory._record_memory_history(enabled=False)

            return forward_end - forward_start, backward_end - backward_start