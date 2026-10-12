import torch
import triton
import triton.language as tl

@triton.jit
def dot_i8(A16, A8, B16, B8, C, GEN_A: tl.constexpr, GEN_B: tl.constexpr):
    m = tl.arange(0, 16)
    n = tl.arange(0, 16)
    k = tl.arange(0, 32)
    if GEN_A:
        a_src = tl.load(A16 + m[:, None] * 32 + k[None, :])
        a = (tl.exp2(a_src.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
    else:
        a = tl.load(A8 + m[:, None] * 32 + k[None, :])
    if GEN_B:
        b_src = tl.load(B16 + k[:, None] * 16 + n[None, :])
        b = (tl.exp2(b_src.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
    else:
        b = tl.load(B8 + k[:, None] * 16 + n[None, :])
    acc = tl.dot(a, b, out_dtype=tl.int32)
    tl.store(C + m[:, None] * 16 + n[None, :], acc)

@triton.jit
def chained_dot_i8(A, B, V, P, C):
    m = tl.arange(0, 16)
    n = tl.arange(0, 32)
    k = tl.arange(0, 32)
    d = tl.arange(0, 16)
    a = tl.load(A + m[:, None] * 32 + k[None, :])
    b = tl.load(B + k[:, None] * 32 + n[None, :])
    qk = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
    p = tl.exp2(qk * 0.0001)
    p8 = (p * 64.0 + 0.5).to(tl.int8)
    v = tl.load(V + n[:, None] * 16 + d[None, :])
    out = tl.dot(p8, v, out_dtype=tl.int32)
    tl.store(P + m[:, None] * 32 + n[None, :], p8)
    tl.store(C + m[:, None] * 16 + d[None, :], out)




def main():
    print('torch', torch.__version__, 'triton', triton.__version__, 'device', torch.cuda.get_device_name())
    vals = torch.tensor([1, 2, 4, 8], device='cuda', dtype=torch.int8)
    ai = vals[torch.randint(0, 4, (16, 32), device='cuda')]
    bi = vals[torch.randint(0, 4, (32, 16), device='cuda')]
    a16 = torch.log2(ai.float() / 127.0).to(torch.float16)
    b16 = torch.log2(bi.float() / 127.0).to(torch.float16)
    out = torch.empty((16, 16), device='cuda', dtype=torch.int32)
    ref = (ai.float() @ bi.float()).to(torch.int32)
    for gen_a, gen_b in [(False, False), (True, False), (False, True), (True, True)]:
        dot_i8[(1,)](a16, ai, b16, bi, out, gen_a, gen_b, num_warps=4)
        torch.cuda.synchronize()
        err = (out - ref).abs().max().item()
        print('generated', gen_a, gen_b, 'max_int_error', err, 'correct', torch.equal(out, ref))
        if not torch.equal(out, ref):
            print('out', out.cpu())
            print('ref', ref.cpu())
    b_chain = vals[torch.randint(0, 4, (32, 32), device='cuda')]
    v_chain = vals[torch.randint(0, 4, (32, 16), device='cuda')]
    p_chain = torch.empty((16, 32), device='cuda', dtype=torch.int8)
    chained_out = torch.empty((16, 16), device='cuda', dtype=torch.int32)
    chained_dot_i8[(1,)](ai, b_chain, v_chain, p_chain, chained_out, num_warps=4)
    torch.cuda.synchronize()
    chained_ref = (p_chain.float() @ v_chain.float()).to(torch.int32)
    print('qk-generated lhs max_int_error', (chained_out - chained_ref).abs().max().item(),
          'correct', torch.equal(chained_out, chained_ref))

    compiled = chained_dot_i8.warmup(ai, b_chain, v_chain, p_chain, chained_out,
                                     num_warps=4, grid=(1,))
    print('TTGIR_START')
    print(compiled.asm['ttgir'])
if __name__ == '__main__':
    main()
