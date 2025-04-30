
import torch
import triton
import triton.language as tl
from einops import rearrange, einsum, reduce
import math

MAX_TILE_SIZE = 256
NUM_TILES = 32
MIN_TILE_SIZE = 16
verbose = False

def cdiv(a, b):
    return (a + b - 1) // b

class TorchAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        
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
        TILE_SIZE = min(triton.next_power_of_2(MAX_TILE_SIZE // NUM_TILES), MAX_TILE_SIZE)
        ctx.Q_TILE_SIZE = max(TILE_SIZE, MIN_TILE_SIZE) # B_q
        ctx.K_TILE_SIZE = max(TILE_SIZE, MIN_TILE_SIZE) # B_k

        # initialize result tensor
        logsumexp = torch.empty((combined_batch_dims, N_q), device = device, dtype = dtype)
        O = torch.empty((combined_batch_dims, N_q, D_v), device = device, dtype = dtype)

        # launch kernel
        for query_i in range(cdiv(N_q, ctx.Q_TILE_SIZE)):
            # input buffers
            O_i = torch.zeros(combined_batch_dims, ctx.Q_TILE_SIZE, D_q, device = device, dtype = dtype)
            l_i = torch.zeros(combined_batch_dims, ctx.Q_TILE_SIZE, device = device, dtype = dtype)
            m_i = torch.full((combined_batch_dims, ctx.Q_TILE_SIZE), -1e6, device = device, dtype = dtype)

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
                    print('S_ij', S_ij)
                
                # get row-wise max (max over diff keys, -1) of S_ij and update m_i
                # batch x B_q
                row_max = torch.max(S_ij, dim = -1).values
                m_i_new = torch.maximum(m_i, row_max)

                # compute P_ij, batch x B_q x B_k
                P_ij = torch.exp(S_ij - m_i_new.unsqueeze(-1))
                if verbose:
                    print('P_ij', P_ij)

                # compute l_ij, batch x B_q
                l_i_new = torch.exp(m_i - m_i_new) * l_i + reduce(P_ij, 'b q k -> b q', 'sum')
                if verbose:
                    print('l_i_new', l_i_new)

                # compute O_i_new, which is # batch x B_q x D_q
                diag_elements = torch.exp(m_i - m_i_new) # batch x B_q
                diag = diag_elements.unsqueeze(-1) * O_i
                pv = einsum(P_ij, V_block, 'b q k, b k d -> b q d')
                O_i_new = diag + pv

                if verbose:
                    print('O_i_new', O_i_new)

                # update l_i, m_i, and O_i
                m_i = m_i_new
                l_i = l_i_new
                O_i = O_i_new
        
            # compute final O_i, L_i
            L_i = m_i + torch.log(l_i)
            if verbose:
                print('L_i', L_i)
            
            diag_elements = 1 / l_i # this is the inversion of the diagonal matrix
            identity = torch.eye(l_i.shape[1], device = device, dtype = dtype)
            diag = diag_elements.unsqueeze(-1) * identity.unsqueeze(0)  # shape: [batch, B_q, B_q]

            O_i_final = einsum(diag, O_i, 'b i q, b q d -> b i d')
            if verbose:
                print('O_i_final', O_i_final)

            # write to output
            O[..., q_idx:q_idx + ctx.Q_TILE_SIZE, :] = O_i_final
            logsumexp[..., q_idx:q_idx + ctx.Q_TILE_SIZE] = L_i

        # save for backward
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, logsumexp)

        # reshape output
        O = O.reshape(*batch_dims, N_q, D_v) # will do view if possible
        logsumexp = logsumexp.reshape(*batch_dims, N_q)

        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        # reshape dO
        dO = dO.reshape(-1, dO.shape[-2], dO.shape[-1])

        D_q, N_q, batch_dims = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        D_k, N_k = K.shape[-1], K.shape[-2]
        D_v, N_v = V.shape[-1], V.shape[-2]
        device = Q.device
        dtype = Q.dtype
        
        scale = math.sqrt(D_k)
        D = O * dO # element-wise product, b x N_q x D_v
        D = reduce(D, '... q d -> ... q', 'sum') # rowsum of D
        S = einsum(Q, K, '... q d, ... k d -> ... q k') / scale

        if ctx.is_causal:
            # just do a torch mask for the backward pass
            mask = torch.tril(torch.ones(N_q, N_k, device = device, dtype = dtype))
            S = S.masked_fill(mask == 0, -float('inf'))

        P_ij = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P_ij, dO, '... q k, ... q d -> ... k d') # N_k = N_v
        dP = einsum(dO, V, '... q d, ... v d -> ... q v')
        dS_ij = P_ij * (dP - D.unsqueeze(-1))
        dQ = einsum(dS_ij, K, '... q k, ... k d -> ... q d') / scale
        dK = einsum(dS_ij, Q, '... q k, ... q d -> ... k d') / scale
        
        # return None corresponding to "causal"
        return dQ, dK, dV, None
        

@triton.jit()
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr, 
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr = False,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # make input block pointers
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape = (K_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    # make output block pointers
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES, D),
        strides = (stride_oq, stride_od),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),  
        block_shape = (Q_TILE_SIZE, D),
        order = (1, 0), # major to minor
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_tile_index * Q_TILE_SIZE),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    
    Q = tl.load(Q_block_ptr, boundary_check = (0, 1))
    
    # set up on-chip buffers
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # compute row and column offsets
    row_offsets = tl.arange(0, Q_TILE_SIZE)
    col_offsets = tl.arange(0, K_TILE_SIZE)

    # loop through key tiles
    for key_j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

        process = not is_causal or query_tile_index >= key_j
        
        if process:
            # load in blocks
            K = tl.load(K_block_ptr, boundary_check = (0, 1))
            V = tl.load(V_block_ptr, boundary_check = (0, 1))

            # compute attention dot product
            S_ij = tl.dot(Q, tl.trans(K)) / scale
            if is_causal:
                # create lower triangular mask (i > j)
                row_indices = row_offsets + query_tile_index * Q_TILE_SIZE
                col_indices = col_offsets + key_j * K_TILE_SIZE
                
                row_indices = tl.reshape(row_indices, (Q_TILE_SIZE, 1))
                col_indices = tl.reshape(col_indices, (1, K_TILE_SIZE))
                
                # lower triangular mask (keep only i > j)
                causal_mask = row_indices >= col_indices
                
                # convert to float mask
                mask = tl.where(causal_mask, 0.0, -float('inf'))
                S_ij = S_ij + mask
            
            # get row-wise maximum
            row_max = tl.max(S_ij, axis = -1)
            m_i_new = tl.maximum(m_i, row_max) # element-wise max

            # compute P_ij = D_q x D_k
            P_ij = tl.exp(S_ij - m_i_new[:, None])

            # compute l_ij
            expdiff = tl.exp(m_i - m_i_new)
            l_i_new = expdiff * l_i + tl.sum(P_ij, axis = -1)

            # compute O_i_new 
            diag = expdiff[:, None] * O_i
            pv = tl.dot(P_ij.to(V.dtype), V) # B_q x D

            # update l_i, m_i, and O_i
            l_i = l_i_new
            O_i = diag + pv
            m_i = m_i_new

        # advance K, V block pointers. HAVE TO ASSIGN IT BACK TO THE BLOCK POINTER
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    
    L = m_i + tl.log(l_i)

    tl.store(L_block_ptr, L.to(Q.dtype))

    O_i_final = (1 / l_i)[:, None] * O_i

    tl.store(O_block_ptr, O_i_final.to(Q.dtype))

    
class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal = False):
        ctx.is_causal = is_causal

        # use the same tiling setup as from TorchAttention
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
        # ctx.Q_TILE_SIZE = max(triton.next_power_of_2(N_q) // NUM_TILES, MIN_TILE_SIZE) # B_q
        # ctx.K_TILE_SIZE = max(triton.next_power_of_2(N_k) // NUM_TILES, MIN_TILE_SIZE) # B_k
        TILE_SIZE = min(triton.next_power_of_2(MAX_TILE_SIZE // NUM_TILES), MAX_TILE_SIZE)
        ctx.Q_TILE_SIZE = max(TILE_SIZE, MIN_TILE_SIZE) # B_q
        ctx.K_TILE_SIZE = max(TILE_SIZE, MIN_TILE_SIZE) # B_k

        # initialize result tensor
        L = torch.empty((combined_batch_dims, N_q), device = device, dtype = dtype)
        O = torch.empty((combined_batch_dims, N_q, D_v), device = device, dtype = dtype)

        # launch kernel in 2D grid
        scale = math.sqrt(D_k)
        flash_fwd_kernel[(cdiv(N_q, ctx.Q_TILE_SIZE), combined_batch_dims)] (
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D_k,
            ctx.Q_TILE_SIZE,
            ctx.K_TILE_SIZE,
            is_causal
        )

        # save for backward
        ctx.save_for_backward(Q, K, V, O, L)

        O = O.reshape(*batch_dims, N_q, D_v) # will do view if possible
        L = L.reshape(*batch_dims, N_q)

        return O
        
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        D_q, N_q, batch_dims = Q.shape[-1], Q.shape[-2], Q.shape[:-2]
        D_k, N_k = K.shape[-1], K.shape[-2]
        D_v, N_v = V.shape[-1], V.shape[-2]
        device = Q.device
        dtype = Q.dtype
        
        scale = math.sqrt(D_k)
        D = O * dO # element-wise product, b x N_q x D_v
        D = reduce(D, '... q d -> ... q', 'sum') # rowsum of D
        S = einsum(Q, K, '... q d, ... k d -> ... q k') / scale

        if ctx.is_causal:
            # just do a torch mask for the backward pass
            mask = torch.tril(torch.ones(N_q, N_k, device = device, dtype = dtype))
            S = S.masked_fill(mask == 0, -float('inf'))

        P_ij = torch.exp(S - L.unsqueeze(-1))
        dV = einsum(P_ij, dO, '... q k, ... q d -> ... k d') # N_k = N_v
        dP = einsum(dO, V, '... q d, ... v d -> ... q v')
        dS_ij = P_ij * (dP - D.unsqueeze(-1))
        dQ = einsum(dS_ij, K, '... q k, ... k d -> ... q d') / scale
        dK = einsum(dS_ij, Q, '... q k, ... q d -> ... k d') / scale
        
        # return None corresponding to "causal"
        return dQ, dK, dV, None
        

def check_flash(N_q, N_k, D, batch, causal=False):
    Q = torch.randn(batch, N_q, D, device='cuda', dtype=torch.float16, requires_grad=False)
    K = torch.randn(batch, N_k, D, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, N_k, D, device='cuda', dtype=torch.float16)
    # reference
    O_ref = TorchAttention.apply(Q, K, V, causal)
    # triton
    O_triton = TritonAttention.apply(Q, K, V, causal)
    torch.testing.assert_allclose(O_triton, O_ref, rtol=1e-2, atol=1e-2)
    print(f"✓ match for (N_q={N_q}, N_k={N_k}, D={D}, batch={batch})")

# torch.manual_seed(0)
# check_flash(16, 32, 64, 2)
# # so the problem is with the N_k dimension (tiling over this)
