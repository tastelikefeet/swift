import torch
import torch.distributed as dist
# from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse


from enum import Enum
from flash_attn import flash_attn_func
import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


class AttnType(Enum):
    FA = "fa"
    FA3 = "fa3"
    FLASHINFER = "flashinfer"
    TORCH = "torch"
    SAGE_AUTO = "sage_auto"
    SAGE_FP16 = "sage_fp16"
    SAGE_FP16_TRITON = "sage_fp16_triton"
    SAGE_FP8 = "sage_fp8"
    SAGE_FP8_SM90 = "sage_fp8_sm90"
    SPARSE_SAGE = "sparse_sage"

    @classmethod
    def from_string(cls, s: str):
        for member in cls:
            if member.value == s:
                return member
        raise ValueError(f"'{s}' is not a valid {cls.__name__}")


def flash_attn_forward(q, k, v,
        dropout_p = 0.0,
        softmax_scale = None,
        causal=False,
        window_size=(-1, -1),
        softcap=None,
        alibi_slopes=None,
        return_softmax=False):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < '2.6.3':
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p = dropout_p,
            softmax_scale = softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p = dropout_p,
            softmax_scale = softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    return block_out, block_lse

def flash_attn_backward(dout, q, k, v, out, softmax_lse, block_dq_buffer, block_dk_buffer, block_dv_buffer, dropout_p, softmax_scale,
    bwd_causal, window_size, softcap, alibi_slopes, deterministic, rng_state):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if flash_attn.__version__ < '2.6.3':
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            block_dq_buffer,
            block_dk_buffer,
            block_dv_buffer,
            dropout_p,
            softmax_scale,
            bwd_causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        )
    else:
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            block_dq_buffer,
            block_dk_buffer,
            block_dv_buffer,
            dropout_p,
            softmax_scale,
            bwd_causal,
            window_size[0],  # Pass window_size_left
            window_size[1],  # Pass window_size_right
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        )


def select_flash_attn_impl(
    impl_type: AttnType, stage: str = "fwd-bwd", attn_processor: torch.nn.Module = None
):
    if impl_type == AttnType.FA:
        if stage == "fwd-only":
            return flash_attn_forward
        elif stage == "bwd-only":
            return flash_attn_backward
        else:
            raise ValueError(f"Unknown stage: {stage}")
        

def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    layer_idx = None,
):
    from swift.utils import get_logger
    import math
    logger = get_logger()
    comm = RingComm(process_group)

    # 初始化输出和LSE
    out = None
    lse = None
    next_k, next_v = None, None

    # 记录初始参数
    logger.info(f"=== Ring Flash Attention Forward - Layer {layer_idx}, Rank {comm.rank} ===")
    logger.info(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    logger.info(f"softmax_scale: {softmax_scale}, expected: {1.0 / math.sqrt(q.shape[-1])}")
    logger.info(f"causal: {causal}, dropout_p: {dropout_p}, window_size: {window_size}")
    
    # 检查输入数据的数值范围
    logger.info(f"Input stats - Q: max={q.max():.6f}, min={q.min():.6f}, mean={q.mean():.6f}, std={q.std():.6f}")
    logger.info(f"Input stats - K: max={k.max():.6f}, min={k.min():.6f}, mean={k.mean():.6f}, std={k.std():.6f}")
    logger.info(f"Input stats - V: max={v.max():.6f}, min={v.min():.6f}, mean={v.mean():.6f}, std={v.std():.6f}")
    
    # 检查是否有异常值
    if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
        logger.error(f"NaN detected in inputs - Q: {torch.isnan(q).any()}, K: {torch.isnan(k).any()}, V: {torch.isnan(v).any()}")
    if torch.isinf(q).any() or torch.isinf(k).any() or torch.isinf(v).any():
        logger.error(f"Inf detected in inputs - Q: {torch.isinf(q).any()}, K: {torch.isinf(k).any()}, V: {torch.isinf(v).any()}")

    for step in range(comm.world_size):
        logger.info(f"--- Step {step}/{comm.world_size-1}, Layer {layer_idx}, Rank {comm.rank} ---")
        
        # 准备下一轮的K,V传递
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        # 计算当前块的注意力
        if not causal or step <= comm.rank:
            # 记录当前块的计算参数
            current_causal = causal and step == 0
            logger.info(f"Computing attention block - causal: {current_causal}, step <= rank: {step <= comm.rank}")
            logger.info(f"Current K,V stats - K: max={k.max():.6f}, min={k.min():.6f}")
            logger.info(f"Current K,V stats - V: max={v.max():.6f}, min={v.min():.6f}")
            
            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            
            try:
                block_out, block_lse = fn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=current_causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    return_softmax=True and dropout_p > 0,
                )
                
                # 记录块输出统计
                logger.info(f"Block output stats - out: max={block_out.max():.6f}, min={block_out.min():.6f}, has_nan={torch.isnan(block_out).any()}")
                logger.info(f"Block LSE stats - lse: max={block_lse.max():.6f}, min={block_lse.min():.6f}, has_nan={torch.isnan(block_lse).any()}")
                
                # 检查异常的LSE值
                if block_lse.max() > 50 or block_lse.min() < -50:
                    logger.warning(f"Abnormal LSE values detected! max={block_lse.max():.6f}, min={block_lse.min():.6f}")
                    logger.warning(f"This may indicate numerical instability in attention computation")
                
                if torch.isnan(block_out).any() or torch.isnan(block_lse).any():
                    logger.error(f"NaN detected in block output! out_nan: {torch.isnan(block_out).any()}, lse_nan: {torch.isnan(block_lse).any()}")
                
            except Exception as e:
                logger.error(f"Error in flash attention computation at step {step}: {str(e)}")
                raise e
            
            # 更新累积输出
            if attn_type == AttnType.SPARSE_SAGE:
                out, lse = block_out, block_lse
                logger.info(f"SPARSE_SAGE mode - direct assignment")
            else:
                old_out_max = out.max() if out is not None else None
                old_lse_max = lse.max() if lse is not None else None
                
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                
                logger.info(f"Updated cumulative stats - out: max={out.max():.6f}, min={out.min():.6f}")
                logger.info(f"Updated cumulative stats - lse: max={lse.max():.6f}, min={lse.min():.6f}")
                
                if torch.isnan(out).any() or torch.isnan(lse).any():
                    logger.error(f"NaN in cumulative output after update! out_nan: {torch.isnan(out).any()}, lse_nan: {torch.isnan(lse).any()}")
                    logger.error(f"Previous out_max: {old_out_max}, lse_max: {old_lse_max}")
        else:
            logger.info(f"Skipping computation due to causal mask - step {step} > rank {comm.rank}")

        # 等待通信完成并更新K,V
        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v
            logger.info(f"Updated K,V for next step - K: max={k.max():.6f}, V: max={v.max():.6f}")

    # 最终输出处理
    logger.info(f"Final processing - converting output to {q.dtype}")
    out = out.to(q.dtype)
    
    if attn_type != AttnType.SPARSE_SAGE:
        original_lse_shape = lse.shape
        lse = lse.squeeze(dim=-1).transpose(1, 2)
        logger.info(f"LSE shape transformation: {original_lse_shape} -> {lse.shape}")
    
    # 最终统计
    logger.info(f"=== Final Forward Results - Layer {layer_idx}, Rank {comm.rank} ===")
    logger.info(f"Final out stats: max={out.max():.6f}, min={out.min():.6f}, has_nan={torch.isnan(out).any()}")
    logger.info(f"Final lse stats: max={lse.max():.6f}, min={lse.min():.6f}, has_nan={torch.isnan(lse).any()}")
    
    # 最终异常检查
    if lse.max() > 100 or lse.min() < -100:
        logger.error(f"CRITICAL: Final LSE values are extremely abnormal! This will cause NaN in backward pass.")
        logger.error(f"LSE range: [{lse.min():.6f}, {lse.max():.6f}]")
        logger.error(f"Expected range for seq_len ~{q.shape[1]}: roughly [{math.log(q.shape[1]/2):.2f}, {math.log(q.shape[1]*2):.2f}]")
    
    return out, lse


def ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    layer_idx = None,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None

    block_dq_buffer = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=torch.float32, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=torch.float32, device=v.device)

    next_k, next_v = None, None
    next_dk, next_dv = None, None

    from swift.utils import get_logger
    logger = get_logger()

    for step in range(kv_comm.world_size):
        logger.info(f"=== Step {step}, Layer {layer_idx}, Rank {kv_comm.rank} ===")
        
        # 前向KV传递
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()
            
        # 计算当前块的梯度
        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            fn = select_flash_attn_impl(attn_type, stage="bwd-only")
            
            logger.info(f"Before flash_attn_backward - step {step}, layer {layer_idx}:")
            logger.info(f"  dout: max={dout.max()}, min={dout.min()}, has_nan={torch.isnan(dout).any()}")
            logger.info(f"  q: max={q.max()}, min={q.min()}, has_nan={torch.isnan(q).any()}")
            logger.info(f"  k: max={k.max()}, min={k.min()}, has_nan={torch.isnan(k).any()}")
            logger.info(f"  v: max={v.max()}, min={v.min()}, has_nan={torch.isnan(v).any()}")
            logger.info(f"  out: max={out.max()}, min={out.min()}, has_nan={torch.isnan(out).any()}")
            logger.info(f"  softmax_lse: max={softmax_lse.max()}, min={softmax_lse.min()}, has_nan={torch.isnan(softmax_lse).any()}")
            
            temp_dq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            temp_dk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
            temp_dv = torch.empty(v.shape, dtype=v.dtype, device=v.device)
            
            fn(
                dout, q, k, v, out, softmax_lse,
                temp_dq, temp_dk, temp_dv,
                dropout_p, softmax_scale, bwd_causal, window_size,
                softcap, alibi_slopes, deterministic, rng_state=None,
            )

            logger.info(f"After flash_attn_backward - step {step}, layer {layer_idx}:")
            logger.info(f"  temp_dq: max={temp_dq.max()}, min={temp_dq.min()}, has_nan={torch.isnan(temp_dq).any()}")
            logger.info(f"  temp_dk: max={temp_dk.max()}, min={temp_dk.min()}, has_nan={torch.isnan(temp_dk).any()}")
            logger.info(f"  temp_dv: max={temp_dv.max()}, min={temp_dv.min()}, has_nan={torch.isnan(temp_dv).any()}")

            block_dq_buffer.copy_(temp_dq.to(torch.float32))
            block_dk_buffer.copy_(temp_dk.to(torch.float32))
            block_dv_buffer.copy_(temp_dv.to(torch.float32))

            if dq is None:
                dq = block_dq_buffer.clone()
                dk = block_dk_buffer.clone()
                dv = block_dv_buffer.clone()
                logger.info(f"Initial gradients - dq: max={dq.max()}, dk: max={dk.max()}, dv: max={dv.max()}")
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                
                logger.info(f"Before accumulation - next_dk: {next_dk.max() if next_dk is not None else 'None'}, next_dv: {next_dv.max() if next_dv is not None else 'None'}")
                logger.info(f"Before accumulation - next_dk has_nan: {torch.isnan(next_dk).any() if next_dk is not None else 'None'}")
                logger.info(f"Before accumulation - next_dv has_nan: {torch.isnan(next_dv).any() if next_dv is not None else 'None'}")
                
                if next_dk is not None:
                    dk = block_dk_buffer + next_dk
                else:
                    dk = block_dk_buffer.clone()
                    
                if next_dv is not None:
                    dv = block_dv_buffer + next_dv
                else:
                    dv = block_dv_buffer.clone()
                    
                logger.info(f"After accumulation - dk: max={dk.max()}, has_nan={torch.isnan(dk).any()}")
                logger.info(f"After accumulation - dv: max={dv.max()}, has_nan={torch.isnan(dv).any()}")
                    
        elif step != 0:
            d_kv_comm.wait()
            dk = next_dk if next_dk is not None else torch.zeros_like(k, dtype=torch.float32, device=k.device)
            dv = next_dv if next_dv is not None else torch.zeros_like(v, dtype=torch.float32, device=v.device)

        # 更新KV用于下一轮计算
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        # 发送当前的KV梯度到下一个GPU
        if dk is None:
            dk = torch.zeros_like(k, dtype=torch.float32, device=k.device)
        if dv is None:
            dv = torch.zeros_like(v, dtype=torch.float32, device=v.device)
            
        logger.info(f"Before send - dk: max={dk.max()}, min={dk.min()}, has_nan={torch.isnan(dk).any()}")
        logger.info(f"Before send - dv: max={dv.max()}, min={dv.min()}, has_nan={torch.isnan(dv).any()}")
        logger.info(f"Sending from rank {d_kv_comm.rank} to rank {d_kv_comm.send_rank}")
            
        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()
        
        logger.info(f"After send_recv - next_dk is None: {next_dk is None}")
        logger.info(f"After send_recv - next_dv is None: {next_dv is None}")
        if next_dk is not None:
            logger.info(f"After send_recv - next_dk: max={next_dk.max()}, min={next_dk.min()}, has_nan={torch.isnan(next_dk).any()}")
        if next_dv is not None:
            logger.info(f"After send_recv - next_dv: max={next_dv.max()}, min={next_dv.min()}, has_nan={torch.isnan(next_dv).any()}")

    # 等待最后的通信完成
    d_kv_comm.wait()
    
    logger.info(f"Final results - layer {layer_idx}:")
    logger.info(f'  dout.max: {dout.max()}')
    logger.info(f'  dq.max: {dq.max() if dq is not None else "None"}')
    logger.info(f'  next_dk.max: {next_dk.max() if next_dk is not None else "None"}')
    logger.info(f'  next_dv.max: {next_dv.max() if next_dv is not None else "None"}')
    
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        layer_idx,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            layer_idx=layer_idx
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        ctx.layer_idx = layer_idx
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            attn_type=ctx.attn_type,
            layer_idx=ctx.layer_idx,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def ring_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )


def ring_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
):
    return RingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
    )


def ring_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    layer_idx=None,
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        layer_idx,
    )
