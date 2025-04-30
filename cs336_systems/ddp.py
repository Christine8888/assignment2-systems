from cs336_basics.model import BasicsTransformerLM
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch
import timeit
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
import torch.distributed as dist
import os
import random

nvtx_profile = False

def nvtx_range(name):
    if nvtx_profile:
        return nvtx.range(name)
    else:
        return nullcontext()

class BaseTrainer:
    def __init__(self, device, model_params, optimizer_params, jit_compile = True):
        self.device = device
        self.jit_compile = jit_compile
        self.init_model(model_params, self.device)
        self.init_optimizer(optimizer_params)

    def init_model(self, model_params, device):
        self.model = BasicsTransformerLM(**model_params).to(device)
        if self.jit_compile: self.model = torch.compile(self.model)
    
    def init_optimizer(self, optimizer_params):
        self.optimizer = AdamW(self.model.parameters(), **optimizer_params)
        
    def training_step(self, x_data, y_data):
        """Base implementation of a single training step"""
        raise NotImplementedError("Subclasses must implement training_step")

class NaiveDDPTrainer(BaseTrainer):
    """Naive DDP trainer"""
    def __init__(self, device, model_params, optimizer_params, n_procs: int, rank = None, backend = "nccl", jit_compile = True):
        self.n_procs = n_procs
        self.jit_compile = jit_compile

        if rank is None:
            self.rank = int(os.environ.get("LOCAL_RANK", -1))
        else:
            self.rank = rank

        self.setup(self.rank, self.n_procs, backend)

        # initialize model, optimizer; device set already in setup
        self.device = torch.device(f"cuda:{self.rank}")
        super().init_model(model_params, self.device)
        super().init_optimizer(optimizer_params)

        self.param_sync()
    
    def setup(self, rank, world_size: int, backend: str = "nccl"):
        print(f"Setting up process {rank} with backend {backend}")

        # set up master address and port for multiprocessing
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        # master_port = os.environ.get("MASTER_PORT", "29500")
        master_port = os.environ.get("MASTER_PORT", str(random.randint(29500, 29600)))
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    def param_sync(self):
        # broadcast parameters and optimizer state
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                dist.broadcast(v, src=0)
        
        dist.barrier()

    def training_step(self, x_data, y_data):
        # forward pass
        full_start = timeit.default_timer()
        logits = self.model(x_data)
        torch.cuda.synchronize()
        
        # backward pass
        loss = cross_entropy(logits, y_data)
        loss.backward()
        torch.cuda.synchronize()

        # sync gradients across processes
        collect_start = timeit.default_timer()
        for param in self.model.parameters():
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self.n_procs)
        
        # # print grad on attention weights
        # for i, param in enumerate(self.model.parameters()):
        #     if i == 10:
        #         print(f"NaiveDDPTrainer grad: {param.grad}")
            
        torch.cuda.synchronize()
        dist.barrier()
        collect_end = timeit.default_timer()

        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        full_step_end = timeit.default_timer()

        return full_step_end - full_start, collect_end - collect_start

class FlattenedDDPTrainer(NaiveDDPTrainer):
    """Flattened DDP trainer"""
    def __init__(self, device, model_params, optimizer_params, n_procs: int, rank = None, backend = "nccl", jit_compile = True):
        super().__init__(device, model_params, optimizer_params, n_procs, rank, backend, jit_compile)
    
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
        source_grads = [param.grad for param in self.model.parameters()]
        flattened = torch._utils._flatten_dense_tensors(source_grads)
        dist.all_reduce(flattened, op=dist.ReduceOp.SUM)
        flattened.div_(self.n_procs)
        
        # update with unflattened gradients
        unflattened = torch._utils._unflatten_dense_tensors(flattened, source_grads)
        for param, unflattened_grad in zip(self.model.parameters(), unflattened):
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
    def __init__(self, device, model_params, optimizer_params, jit_compile = True):
        super().__init__(device, model_params, optimizer_params, jit_compile)

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
    def __init__(self, device, model_params, optimizer_params, mixed_precision = False, amp_dtype = torch.bfloat16, memory_flag = None, jit_compile = True):
        super().__init__(device, model_params, optimizer_params, jit_compile)
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