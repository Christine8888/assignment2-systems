import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from collections import defaultdict
import types
from typing import Any, Dict, Type

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        """
        params: iterable or generator of parameters or parameter groups to optimize
        optimizer_cls: The PyTorch optimizer class to use (e.g., torch.optim.AdamW)
        **kwargs: Keyword arguments to pass to the optimizer constructor
        """
        # store optimizer paramters
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # get world size and rank from dist
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        print(f"ShardedOptimizer: world_size: {self.world_size}, rank: {self.rank}")
        
        # parent constructor with empty parameter group
        self.optim = None
        
        # initialize nn.Optimizer, empty parameter group
        super().__init__([{'params': []}], defaults={})
        
        # process input parameters
        if isinstance(params, dict):
            self.add_param_group(params)
        else:
            # iterable or generator, convert to list
            param_groups = list(params)
            if not isinstance(param_groups[0], dict):
                # then param_groups is just a list of nn.Parameters
                # group them into a single parameter group
                param_groups = [{'params': param_groups}]
            
            # add each parameter group to the optimizer
            for param_group in param_groups:
                self.add_param_group(param_group)
    
    def rank_from_idx(self, idx: int) -> int:
        """
        Determine which rank owns the parameter at index idx
        """
        return idx % self.world_size

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Add a new parameter group to an existing sharded optimizer.
        
        Args:
            param_group: dictionary containing parameters and optimizer options
        """
        # copy parameter group
        param_group_copy = {k: v for k, v in param_group.items() if k != 'params'}
        
        if 'params' not in param_group:
            raise ValueError("param group must contain 'params' key")

        # get parameters
        params = list(param_group['params'])
            
        # determine which parameters this rank is responsible for
        # assume fixed ordering of parameters across ranks
        this_rank_params = []
        for i, param in enumerate(params):
            if self.rank_from_idx(i) == self.rank:
                this_rank_params.append(param)
                
        # create a new parameter group for this rank's parameters
        sharded_group = {k: v for k, v in param_group_copy.items()}
        sharded_group['params'] = this_rank_params
        
        # create actual optimizer if not already existing
        if self.optim is None:
            self.optim = self.optimizer_cls(
                [sharded_group],
                **self.optimizer_kwargs
            )
        else:
            # add sharded group to existing optimizer
            self.optim.add_param_group(sharded_group)
            
        # add the full new parameter group to the parent class
        super(ShardedOptimizer, self).add_param_group(param_group)
        
    def step(self, closure=None, **kwargs):
        """
        Perform optimization step and synchronize parameters.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            **kwargs: Additional arguments to pass to the optimizer step method
        """

        # this really matters ? i guess
        if self.optim:
            loss = self.optim.step(closure, **kwargs)
        else:
            loss = closure()
                
        # perform broadcasting to sync parameters across all ranks
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                main_rank = self.rank_from_idx(param_idx)
                
                # broadcast
                dist.broadcast(param.data, src=main_rank)
                
        return loss