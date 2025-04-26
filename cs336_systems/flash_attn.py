
import torch
import triton
import triton.language as tl
from einops import rearrange, einsum, reduce
import math
NUM_TILES = 16
MIN_TILE_SIZE = 16
verbose = False

def cdiv(a, b):
    return (a + b - 1) // b

class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        if verbose:
            print(Q.shape)
        
        D_q, N_q, batch_dims = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        D_k, N_k = K.shape[-1], K.shape[-2]
        D_v, N_v = V.shape[-1], V.shape[-2]
        device = Q.device
        dtype = Q.dtype
        
        # sanity checks
        assert N_k == N_v, "key/value mismatch"
        assert D_q == D_k, "QK dimension mismatch"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "expected contiguous tensors"

        # reshape batch dimensions into a single dimension for convenience
        Q = rearrange(Q, "... n_q d -> (...) n_q d")
        K = rearrange(K, "... n_k d -> (...) n_k d")
        V = rearrange(V, "... n_v d -> (...) n_v d")

        combined_batch_dims = Q.shape[0]

        # compute tile sizes
        ctx.Q_TILE_SIZE = min(triton.next_power_of_2(N_q) // NUM_TILES, MIN_TILE_SIZE) # B_q
        ctx.K_TILE_SIZE = min(triton.next_power_of_2(N_k) // NUM_TILES, MIN_TILE_SIZE) # B_k

        # initialize result tensor
        logsumexp = torch.empty((combined_batch_dims, N_q), device = device, dtype = dtype)
        O = torch.empty((combined_batch_dims, N_q, D_v), device = device, dtype = dtype)

        # launch kernel
        for query_i in range(cdiv(N_q, ctx.Q_TILE_SIZE)):
            # input buffers
            O_i = torch.zeros(combined_batch_dims, ctx.Q_TILE_SIZE, D_q)
            l_i = torch.zeros(combined_batch_dims, ctx.Q_TILE_SIZE)
            m_i = torch.full((combined_batch_dims, ctx.Q_TILE_SIZE), -float('inf'))

            # get Q block
            q_idx = query_i * ctx.Q_TILE_SIZE
            Q_block = Q[:, q_idx:q_idx + ctx.Q_TILE_SIZE, :]

            for key_j in range(cdiv(N_k, ctx.K_TILE_SIZE)):
                # get K and V blocks
                k_idx = key_j * ctx.K_TILE_SIZE
                K_block = K[:, k_idx:k_idx + ctx.K_TILE_SIZE, :]
                V_block = V[:, k_idx:k_idx + ctx.K_TILE_SIZE, :]

                # compute softmaxes; batch x B_q x B_k
                S_ij = einsum(Q_block, K_block, 'b i j, b k j -> b i k') / math.sqrt(D_k)
                if verbose:
                    print('S_ij', S_ij.max())
                
                # get row-wise max (max over diff keys, -1) of S_ij and update m_i
                # batch x B_q
                row_max = torch.max(S_ij, dim = -1).values
                m_i_new = torch.maximum(m_i, row_max)

                # compute P_ij, batch x B_q x B_k
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))
                if verbose:
                    print('P_ij', P_ij.max())

                # compute l_ij, batch x B_q
                l_i_new = torch.exp(m_i - m_i_new) * l_i + reduce(P_ij, 'b q k -> b q', 'sum')
                if verbose:
                    print('m_i - m_i_new', (m_i - m_i_new).max())
                    print('torch.exp(m_i - m_i_new)', torch.exp(m_i - m_i_new).max()) 
                    print('l_i', l_i.max())
                    print('reduce(P_ij, b q k -> b q', reduce(P_ij, 'b q k -> b q', 'sum').max())
                    print('l_i_new', l_i_new.max())

                # compute O_i_new, which is # batch x B_q x D_q
                diag_elements = torch.exp(m_i - m_i_new) # batch x B_q
                identity = torch.eye(m_i.shape[1], device = device)
                diag = diag_elements.unsqueeze(-1) * identity.unsqueeze(0)  # shape: [batch, B_q, B_q]
                diag = einsum(diag, O_i, 'b i q, b q d -> b i d')
                pv = einsum(P_ij, V_block, 'b q k, b k d -> b q d')
                O_i_new = diag + pv

                # update l_i, m_i, and O_i
                m_i = m_i_new
                l_i = l_i_new
                O_i = O_i_new
        
            # compute final O_i, L_i
            L_i = m_i + torch.log(l_i)
            
            diag_elements = 1 / l_i # this is the inversion of the diagonal matrix
            identity = torch.eye(l_i.shape[1], device = device)
            diag = diag_elements.unsqueeze(-1) * identity.unsqueeze(0)  # shape: [batch, B_q, B_q]

            O_i_final = einsum(diag, O_i, 'b i q, b q d -> b i d')

            # write to output
            O[..., q_idx:q_idx + ctx.Q_TILE_SIZE, :] = O_i_final
            logsumexp[..., q_idx:q_idx + ctx.Q_TILE_SIZE] = L_i

        # reshape output
        O = O.reshape(*batch_dims, N_q, D_v)
        logsumexp = logsumexp.reshape(*batch_dims, N_q)
        
        # save for backward
        ctx.save_for_backward(Q, K, V, logsumexp)

        return O
    
    @staticmethod
    def backward(ctx, grad_out, grad_logsumexp):
        raise NotImplementedError()
