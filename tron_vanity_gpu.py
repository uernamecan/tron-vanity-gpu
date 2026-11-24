#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRON (波场) 靓号地址生成器 —— NVIDIA GPU 版，自定义「后缀」，针对 RTX 50 系 (Blackwell) 优化
=====================================================================================

用法
----
    pip install "cupy-cuda12x[ctk]"          # 一次性安装（需要 NVIDIA 驱动 570+；RTX 5060 Ti 等 Blackwell 显卡需 CUDA 12.8+ 的 NVRTC）
    python tron_vanity_gpu.py 8888            # 找以 8888 结尾的地址
    python tron_vanity_gpu.py Abc -n 3        # 找 3 个以 Abc 结尾的地址（每个都用全新的随机起点）
    python tron_vanity_gpu.py --selftest      # 自检（CPU 参考向量 + PTX 域运算 + 全长度/全轨道 GPU 已知答案测试）
    python tron_vanity_gpu.py --bench         # 只测速
    python tron_vanity_gpu.py 8888 --tune     # 自动调优并缓存最佳参数（可选，只需做一次）

原理
----
  * 每个 GPU 线程从一个随机私钥对应的公钥点 B 出发，利用「对称点加 B ± d*G 共享模逆 + 批量(Montgomery)求逆」
    配合「secp256k1 点取反 (x, -y) 与 GLV 三次自同态 (βx, ±y), (β²x, ±y)」6 路对称扩展，
    每轮覆盖 2*GROUP-1 个连续基准点 × 6 路独立私钥 = 6*(2*GROUP-1) 个地址，摊销后每候选仅约 1.1 次域乘。
  * 256 位域算术采用手写 PTX 进位链乘法、专用平方 (fe_sqr)、快速模约简与加减取反；
    Keccak-256 采用 32-bit lop3/funnelshift 与首尾轮特化裁剪；SHA-256 采用 lop3 与尾轮裁剪。
  * 后缀判定采用编译期常数线性组合 + 单次 Barrett 约简（无 64 位除法）；
    后缀 >= 6 位时先反推目标校验和 D 过滤，6–7 位用 Warp 级 Shared Memory 聚合满 32 候选再批量并发算 SHA-256d。
  * 起始私钥来自操作系统 CSPRNG（secrets.token_bytes，256 位）；
    GPU 报告的每一个命中，都会用独立的纯 Python CPU 代码从私钥重新推导地址并校验后才显示。

注意
----
  * 私钥会明文打印并追加保存到 tron_vanity_found.txt（与本脚本同目录），请自行妥善保管，用完可删除该文件。
  * 建议断网运行；不要使用来路不明的“靓号生成器”，它们可能偷私钥。
"""
import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time


# =====================================================================================
#  Part 1 : CPU reference implementation (pure Python) - used to verify every GPU hit
# =====================================================================================

P  = 2**256 - 2**32 - 977
N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G  = (GX, GY)

# secp256k1 GLV endomorphism constants:
#   (BETA * x mod P, y) = (LAM * k mod N) * G
#   (BETA^2 * x mod P, y) = (LAM^2 * k mod N) * G,  with  1 + BETA + BETA^2 == 0 (mod P)
BETA  = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
LAM   = 0x5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
LAM2  = (LAM * LAM) % N

SYM_MULTS     = (1, N - 1, LAM, N - LAM, LAM2, N - LAM2)
SYM_INV_MULTS = (1, N - 1, LAM2, N - LAM2, LAM, N - LAM)


def ec_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1; x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % P == 0: return None
        lam = 3 * x1 * x1 * pow(2 * y1, -1, P) % P
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)


def ec_mul(k, pt=G):
    r = None
    while k:
        if k & 1: r = ec_add(r, pt)
        pt = ec_add(pt, pt)
        k >>= 1
    return r


def ec_batch_add_const(pts, q):
    """Compute [p + q for p in pts] using a single modular inversion (falls back to ec_add if any p.x == q.x)."""
    qx, qy = q
    n = len(pts)
    if n == 0:
        return []
    dx = [(qx - p[0]) % P for p in pts]
    if 0 in dx:
        return [ec_add(p, q) for p in pts]
    pre = [0] * n
    acc = 1
    for i in range(n):
        acc = (acc * dx[i]) % P
        pre[i] = acc
    inv = pow(acc, -1, P)
    out = [None] * n
    for i in range(n - 1, -1, -1):
        idx = (inv * pre[i - 1]) % P if i > 0 else inv
        inv = (inv * dx[i]) % P
        px, py = pts[i]
        lam = ((qy - py) * idx) % P
        x3 = (lam * lam - px - qx) % P
        y3 = (lam * (px - x3) - py) % P
        out[i] = (x3, y3)
    return out


_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
       0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
       0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
       0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
       0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
       0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]]
_M64 = (1 << 64) - 1


def _rol(x, n):
    n %= 64
    return ((x << n) | (x >> (64 - n))) & _M64 if n else x


def _keccak_f(A):
    for rc in _RC:
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol(C[(x + 1) % 5], 1) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol(A[x][y], _ROT[x][y])
        A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        A[0][0] ^= rc
    return A


def keccak256(data: bytes) -> bytes:
    rate = 136
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate: msg.append(0)
    msg[-1] |= 0x80
    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(msg[off + 8 * i: off + 8 * i + 8], 'little')
        A = _keccak_f(A)
    out = b''.join(A[i % 5][i // 5].to_bytes(8, 'little') for i in range(4))
    return out[:32]


B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def b58encode(b: bytes) -> str:
    n = int.from_bytes(b, 'big'); s = ''
    while n:
        n, r = divmod(n, 58); s = B58[r] + s
    return '1' * (len(b) - len(b.lstrip(b'\0'))) + s


def hash160_of_priv(k):
    x, y = ec_mul(k)
    return keccak256(x.to_bytes(32, 'big') + y.to_bytes(32, 'big'))[12:]


def tron_addr_from_h160(h):
    payload = b'\x41' + h
    chk = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return b58encode(payload + chk)


def tron_address(k):
    return tron_addr_from_h160(hash160_of_priv(k))


# =====================================================================================
#  Part 2 : CUDA kernel (compiled at run time with NVRTC through CuPy)
# =====================================================================================

CUDA_SRC = r'''
// ============ TRON vanity (suffix) search kernel (optimized) ============
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ u32 bswap32(u32 x) {
    return __byte_perm(x, 0, 0x0123);
}

#if USE_PTX
__device__ __forceinline__ u32 lop3_xor3(u32 a, u32 b, u32 c) {
    u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
__device__ __forceinline__ u32 lop3_chi(u32 a, u32 b, u32 c) {
    u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
__device__ __forceinline__ u32 lop3_ch(u32 e, u32 f, u32 g) {
    u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(e), "r"(f), "r"(g));
    return r;
}
__device__ __forceinline__ u32 lop3_mj(u32 a, u32 b, u32 c) {
    u32 r;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}
/* 64-bit ROL by 1 using dual-issue IADD3 carry chain (2.7x faster throughput than SHF on Blackwell) */
__device__ __forceinline__ void rol64_1(u32 &rl, u32 &rh, u32 lo, u32 hi) {
    u32 o0, o1;
    asm(
        "add.cc.u32  %0, %2, %2;\n\t"
        "addc.cc.u32 %1, %3, %3;\n\t"
        "addc.u32    %0, %0, 0;\n\t"
        : "=&r"(o0), "=&r"(o1)
        : "r"(lo), "r"(hi)
    );
    rl = o0; rh = o1;
}
#else
__device__ __forceinline__ u32 lop3_xor3(u32 a, u32 b, u32 c) { return a ^ b ^ c; }
__device__ __forceinline__ u32 lop3_chi(u32 a, u32 b, u32 c)  { return a ^ (~b & c); }
__device__ __forceinline__ u32 lop3_ch(u32 e, u32 f, u32 g)   { return (e & f) ^ (~e & g); }
__device__ __forceinline__ u32 lop3_mj(u32 a, u32 b, u32 c)   { return (a & b) ^ (a & c) ^ (b & c); }
__device__ __forceinline__ void rol64_1(u32 &rl, u32 &rh, u32 lo, u32 hi) {
    rl = __funnelshift_l(hi, lo, 1);
    rh = __funnelshift_l(lo, hi, 1);
}
#endif

/* ---------- secp256k1 field: p = 2^256 - 2^32 - 977, 8 x 32-bit limbs (little endian) ---------- */

__device__ __forceinline__ void fe_add_c(u32 *r, const u32 *a, const u32 *b) {
    u32 t[8], s[8];
    u64 c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (u64)a[i] + b[i]; t[i] = (u32)c; c >>= 32; }
    u64 d = 0x1000003D1ULL;
    #pragma unroll
    for (int i = 0; i < 8; i++) { d += t[i]; s[i] = (u32)d; d >>= 32; }
    const u32 use_s = (u32)(c | d);
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = use_s ? s[i] : t[i];
}

__device__ __forceinline__ void fe_sub_c(u32 *r, const u32 *a, const u32 *b) {
    u32 t[8];
    u64 bw = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 d = (u64)a[i] - b[i] - bw;
        t[i] = (u32)d;
        bw = (d >> 63) & 1;
    }
    const u32 mask = (u32)0 - (u32)bw;
    u64 c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const u32 pi = (i == 0) ? 0xFFFFFC2Fu : ((i == 1) ? 0xFFFFFFFEu : 0xFFFFFFFFu);
        c += (u64)t[i] + (pi & mask);
        r[i] = (u32)c; c >>= 32;
    }
}

/* r = p - a, valid for 0 < a < p (all affine secp256k1 coordinates satisfy 0 < x, y < p) */
__device__ __forceinline__ void fe_neg_nz_c(u32 *r, const u32 *a) {
    u64 bw = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const u32 pi = (i == 0) ? 0xFFFFFC2Fu : ((i == 1) ? 0xFFFFFFFEu : 0xFFFFFFFFu);
        u64 d = (u64)pi - a[i] - bw;
        r[i] = (u32)d;
        bw = (d >> 63) & 1;
    }
}

/* 256x256 -> 512 bit product, portable version */
__device__ __forceinline__ void mul_wide_c(u32 *t, const u32 *a, const u32 *b) {
    u64 c = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) { u64 cur = (u64)a[0] * b[j] + c; t[j] = (u32)cur; c = cur >> 32; }
    t[8] = (u32)c;
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        c = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            u64 cur = (u64)a[i] * b[j] + t[i + j] + c;
            t[i + j] = (u32)cur; c = cur >> 32;
        }
        t[i + 8] = (u32)c;
    }
}

__device__ __forceinline__ void fe_reduce_c(u32 *r, const u32 *t) {
    /* t (512 bit) = lo + hi*2^256  ==  lo + hi*(2^32 + 977)  (mod p) */
    u32 s[8];
    u64 c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        u64 cur = (u64)t[8 + i] * 977ULL + t[i] + c;
        if (i > 0) cur += t[8 + i - 1];
        s[i] = (u32)cur; c = cur >> 32;
    }
    const u64 top = c + t[15];                 /* < 2^33 */
    c = (u64)s[0] + top * 977ULL;  s[0] = (u32)c; c >>= 32;
    c += (u64)s[1] + (u32)top;     s[1] = (u32)c; c >>= 32;
    c += (u64)s[2] + (top >> 32);  s[2] = (u32)c; c >>= 32;
    #pragma unroll
    for (int i = 3; i < 8; i++) { c += s[i]; s[i] = (u32)c; c >>= 32; }
    c *= 0x1000003D1ULL;                       /* carry out of 2^256 (0 or 1) */
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += s[i]; s[i] = (u32)c; c >>= 32; }
    u32 u[8];
    u64 d = 0x1000003D1ULL;                    /* s >= p  <=>  s + C overflows 2^256 */
    #pragma unroll
    for (int i = 0; i < 8; i++) { d += s[i]; u[i] = (u32)d; d >>= 32; }
    const u32 use_u = (u32)d;
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = use_u ? u[i] : s[i];
}

#if USE_PTX
__device__ __forceinline__ void fe_add_ptx(u32 *r, const u32 *a, const u32 *b) {
    const u32 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    const u32 b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    u32 t0, t1, t2, t3, t4, t5, t6, t7, c;
    asm(
        "add.cc.u32  %0, %9,  %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "=&r"(t0), "=&r"(t1), "=&r"(t2), "=&r"(t3),
          "=&r"(t4), "=&r"(t5), "=&r"(t6), "=&r"(t7), "=&r"(c)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7)
    );
    u32 s0, s1, s2, s3, s4, s5, s6, s7, d;
    asm(
        "add.cc.u32  %0, %9,  977;\n\t"
        "addc.cc.u32 %1, %10, 1;\n\t"
        "addc.cc.u32 %2, %11, 0;\n\t"
        "addc.cc.u32 %3, %12, 0;\n\t"
        "addc.cc.u32 %4, %13, 0;\n\t"
        "addc.cc.u32 %5, %14, 0;\n\t"
        "addc.cc.u32 %6, %15, 0;\n\t"
        "addc.cc.u32 %7, %16, 0;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "=&r"(s0), "=&r"(s1), "=&r"(s2), "=&r"(s3),
          "=&r"(s4), "=&r"(s5), "=&r"(s6), "=&r"(s7), "=&r"(d)
        : "r"(t0), "r"(t1), "r"(t2), "r"(t3), "r"(t4), "r"(t5), "r"(t6), "r"(t7)
    );
    const u32 use_s = c | d;
    r[0] = use_s ? s0 : t0; r[1] = use_s ? s1 : t1; r[2] = use_s ? s2 : t2; r[3] = use_s ? s3 : t3;
    r[4] = use_s ? s4 : t4; r[5] = use_s ? s5 : t5; r[6] = use_s ? s6 : t6; r[7] = use_s ? s7 : t7;
}

__device__ __forceinline__ void fe_sub_ptx(u32 *r, const u32 *a, const u32 *b) {
    const u32 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    const u32 b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    u32 t0, t1, t2, t3, t4, t5, t6, t7, mask;
    asm(
        "sub.cc.u32  %0, %9,  %17;\n\t"
        "subc.cc.u32 %1, %10, %18;\n\t"
        "subc.cc.u32 %2, %11, %19;\n\t"
        "subc.cc.u32 %3, %12, %20;\n\t"
        "subc.cc.u32 %4, %13, %21;\n\t"
        "subc.cc.u32 %5, %14, %22;\n\t"
        "subc.cc.u32 %6, %15, %23;\n\t"
        "subc.cc.u32 %7, %16, %24;\n\t"
        "subc.u32    %8, 0, 0;\n\t"
        : "=&r"(t0), "=&r"(t1), "=&r"(t2), "=&r"(t3),
          "=&r"(t4), "=&r"(t5), "=&r"(t6), "=&r"(t7), "=&r"(mask)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7)
    );
    const u32 c0 = mask & 977u;
    const u32 c1 = mask & 1u;
    asm(
        "sub.cc.u32  %0, %0, %8;\n\t"
        "subc.cc.u32 %1, %1, %9;\n\t"
        "subc.cc.u32 %2, %2, 0;\n\t"
        "subc.cc.u32 %3, %3, 0;\n\t"
        "subc.cc.u32 %4, %4, 0;\n\t"
        "subc.cc.u32 %5, %5, 0;\n\t"
        "subc.cc.u32 %6, %6, 0;\n\t"
        "subc.u32    %7, %7, 0;\n\t"
        : "+r"(t0), "+r"(t1), "+r"(t2), "+r"(t3),
          "+r"(t4), "+r"(t5), "+r"(t6), "+r"(t7)
        : "r"(c0), "r"(c1)
    );
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
    r[4] = t4; r[5] = t5; r[6] = t6; r[7] = t7;
}

__device__ __forceinline__ void fe_neg_nz_ptx(u32 *r, const u32 *a) {
    const u32 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    u32 r0, r1, r2, r3, r4, r5, r6, r7;
    asm(
        "sub.cc.u32  %0, 0xFFFFFC2F, %8;\n\t"
        "subc.cc.u32 %1, 0xFFFFFFFE, %9;\n\t"
        "subc.cc.u32 %2, 0xFFFFFFFF, %10;\n\t"
        "subc.cc.u32 %3, 0xFFFFFFFF, %11;\n\t"
        "subc.cc.u32 %4, 0xFFFFFFFF, %12;\n\t"
        "subc.cc.u32 %5, 0xFFFFFFFF, %13;\n\t"
        "subc.cc.u32 %6, 0xFFFFFFFF, %14;\n\t"
        "subc.u32    %7, 0xFFFFFFFF, %15;\n\t"
        : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3),
          "=&r"(r4), "=&r"(r5), "=&r"(r6), "=&r"(r7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7)
    );
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    r[4] = r4; r[5] = r5; r[6] = r6; r[7] = r7;
}

__device__ __forceinline__ void fe_reduce_ptx(u32 *r, const u32 *t) {
    u32 s0 = t[0], s1 = t[1], s2 = t[2], s3 = t[3], s4 = t[4], s5 = t[5], s6 = t[6], s7 = t[7];
    const u32 h0 = t[8], h1 = t[9], h2 = t[10], h3 = t[11], h4 = t[12], h5 = t[13], h6 = t[14], h7 = t[15];
    u32 c0, c1;
    asm(
        "mad.lo.cc.u32  %0, %10, 977, %0;\n\t"
        "madc.lo.cc.u32 %1, %11, 977, %1;\n\t"
        "madc.lo.cc.u32 %2, %12, 977, %2;\n\t"
        "madc.lo.cc.u32 %3, %13, 977, %3;\n\t"
        "madc.lo.cc.u32 %4, %14, 977, %4;\n\t"
        "madc.lo.cc.u32 %5, %15, 977, %5;\n\t"
        "madc.lo.cc.u32 %6, %16, 977, %6;\n\t"
        "madc.lo.cc.u32 %7, %17, 977, %7;\n\t"
        "addc.u32       %8, 0, 0;\n\t"
        "mad.hi.cc.u32  %1, %10, 977, %1;\n\t"
        "madc.hi.cc.u32 %2, %11, 977, %2;\n\t"
        "madc.hi.cc.u32 %3, %12, 977, %3;\n\t"
        "madc.hi.cc.u32 %4, %13, 977, %4;\n\t"
        "madc.hi.cc.u32 %5, %14, 977, %5;\n\t"
        "madc.hi.cc.u32 %6, %15, 977, %6;\n\t"
        "madc.hi.cc.u32 %7, %16, 977, %7;\n\t"
        "madc.hi.u32    %8, %17, 977, %8;\n\t"
        "add.cc.u32     %1, %1, %10;\n\t"
        "addc.cc.u32    %2, %2, %11;\n\t"
        "addc.cc.u32    %3, %3, %12;\n\t"
        "addc.cc.u32    %4, %4, %13;\n\t"
        "addc.cc.u32    %5, %5, %14;\n\t"
        "addc.cc.u32    %6, %6, %15;\n\t"
        "addc.cc.u32    %7, %7, %16;\n\t"
        "addc.cc.u32    %8, %8, %17;\n\t"
        "addc.u32       %9, 0, 0;\n\t"
        : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3),
          "+r"(s4), "+r"(s5), "+r"(s6), "+r"(s7), "=&r"(c0), "=&r"(c1)
        : "r"(h0), "r"(h1), "r"(h2), "r"(h3), "r"(h4), "r"(h5), "r"(h6), "r"(h7)
    );
    const u64 top = ((u64)c1 << 32) | (u64)c0;
    const u64 top977 = top * 977ULL;
    const u32 p0 = (u32)top977;
    const u32 p1 = (u32)(top977 >> 32);
    u32 ov;
    asm(
        "add.cc.u32  %0, %0, %9;\n\t"
        "addc.cc.u32 %1, %1, %10;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        "add.cc.u32  %1, %1, %11;\n\t"
        "addc.cc.u32 %2, %2, %12;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.cc.u32 %7, %7, 0;\n\t"
        "addc.u32    %8, %8, 0;\n\t"
        : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3),
          "+r"(s4), "+r"(s5), "+r"(s6), "+r"(s7), "=&r"(ov)
        : "r"(p0), "r"(p1), "r"(c0), "r"(c1)
    );
    /* ov is 0 or 1; add ov * (2^32 + 977) - cannot overflow 2^256 */
    const u32 q0 = ov * 977u;
    asm(
        "add.cc.u32  %0, %0, %8;\n\t"
        "addc.cc.u32 %1, %1, %9;\n\t"
        "addc.cc.u32 %2, %2, 0;\n\t"
        "addc.cc.u32 %3, %3, 0;\n\t"
        "addc.cc.u32 %4, %4, 0;\n\t"
        "addc.cc.u32 %5, %5, 0;\n\t"
        "addc.cc.u32 %6, %6, 0;\n\t"
        "addc.u32    %7, %7, 0;\n\t"
        : "+r"(s0), "+r"(s1), "+r"(s2), "+r"(s3),
          "+r"(s4), "+r"(s5), "+r"(s6), "+r"(s7)
        : "r"(q0), "r"(ov)
    );
    /* final canonicalization: s >= p  <=>  s + (2^32 + 977) overflows 2^256 */
    u32 u0, u1, u2, u3, u4, u5, u6, u7, use_u;
    asm(
        "add.cc.u32  %0, %9,  977;\n\t"
        "addc.cc.u32 %1, %10, 1;\n\t"
        "addc.cc.u32 %2, %11, 0;\n\t"
        "addc.cc.u32 %3, %12, 0;\n\t"
        "addc.cc.u32 %4, %13, 0;\n\t"
        "addc.cc.u32 %5, %14, 0;\n\t"
        "addc.cc.u32 %6, %15, 0;\n\t"
        "addc.cc.u32 %7, %16, 0;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "=&r"(u0), "=&r"(u1), "=&r"(u2), "=&r"(u3),
          "=&r"(u4), "=&r"(u5), "=&r"(u6), "=&r"(u7), "=&r"(use_u)
        : "r"(s0), "r"(s1), "r"(s2), "r"(s3), "r"(s4), "r"(s5), "r"(s6), "r"(s7)
    );
    r[0] = use_u ? u0 : s0; r[1] = use_u ? u1 : s1; r[2] = use_u ? u2 : s2; r[3] = use_u ? u3 : s3;
    r[4] = use_u ? u4 : s4; r[5] = use_u ? u5 : s5; r[6] = use_u ? u6 : s6; r[7] = use_u ? u7 : s7;
}
#endif

@@PTXMUL@@
@@PTXSQR@@

__device__ __forceinline__ void fe_add(u32 *r, const u32 *a, const u32 *b) {
#if USE_PTX
    fe_add_ptx(r, a, b);
#else
    fe_add_c(r, a, b);
#endif
}

__device__ __forceinline__ void fe_sub(u32 *r, const u32 *a, const u32 *b) {
#if USE_PTX
    fe_sub_ptx(r, a, b);
#else
    fe_sub_c(r, a, b);
#endif
}

__device__ __forceinline__ void fe_neg_nz(u32 *r, const u32 *a) {
#if USE_PTX
    fe_neg_nz_ptx(r, a);
#else
    fe_neg_nz_c(r, a);
#endif
}

__device__ __forceinline__ void fe_reduce(u32 *r, const u32 *t) {
#if USE_PTX
    fe_reduce_ptx(r, t);
#else
    fe_reduce_c(r, t);
#endif
}

__device__ __forceinline__ void fe_mul(u32 *r, const u32 *a, const u32 *b) {
    u32 t[16];
#if USE_PTX
    mul_wide_ptx(t, a, b);
#else
    mul_wide_c(t, a, b);
#endif
    fe_reduce(r, t);
}

__device__ __forceinline__ void fe_sqr(u32 *r, const u32 *a) {
    u32 t[16];
#if USE_PTX
    sqr_wide_ptx(t, a);
#else
    mul_wide_c(t, a, a);
#endif
    fe_reduce(r, t);
}

__device__ __noinline__ void fe_mul_ni(u32 *r, const u32 *a, const u32 *b) { fe_mul(r, a, b); }

__device__ __noinline__ void fe_sqrn(u32 *r, const u32 *a, int n) {
    #pragma unroll 1
    for (int i = 0; i < 8; i++) r[i] = a[i];
    #pragma unroll 1
    for (int i = 0; i < n; i++) fe_sqr(r, r);
}

/* r = a^(p-2) = a^-1 (a != 0).  addition chain: 255 squarings + 15 multiplications (compact 5-temp stack) */
__device__ __noinline__ void fe_inv(u32 *r, const u32 *a) {
    u32 x2[8], x3[8], x22[8], x44[8], t[8];
    fe_sqrn(x2, a, 1);     fe_mul_ni(x2, x2, a);
    fe_sqrn(x3, x2, 1);    fe_mul_ni(x3, x3, a);
    fe_sqrn(t, x3, 3);     fe_mul_ni(t, t, x3);      /* x6 */
    fe_sqrn(t, t, 3);      fe_mul_ni(t, t, x3);      /* x9 */
    fe_sqrn(t, t, 2);      fe_mul_ni(t, t, x2);      /* x11 */
    fe_sqrn(x22, t, 11);   fe_mul_ni(x22, x22, t);   /* x22 */
    fe_sqrn(x44, x22, 22); fe_mul_ni(x44, x44, x22); /* x44 */
    fe_sqrn(r, x44, 44);   fe_mul_ni(r, r, x44);     /* x88 (reuses output buf r, assumes r != a) */
    fe_sqrn(t, r, 88);     fe_mul_ni(t, t, r);       /* x176 */
    fe_sqrn(t, t, 44);     fe_mul_ni(t, t, x44);     /* x220 */
    fe_sqrn(t, t, 3);      fe_mul_ni(t, t, x3);      /* x223 */
    fe_sqrn(t, t, 23);     fe_mul_ni(t, t, x22);
    fe_sqrn(t, t, 5);      fe_mul_ni(t, t, a);
    fe_sqrn(t, t, 3);      fe_mul_ni(t, t, x2);
    fe_sqrn(t, t, 2);      fe_mul_ni(r, a, t);
}

/* secp256k1 GLV beta constant (8 LE limbs) */
__device__ __constant__ u32 BETA_C[8] = {
    0x719501EEu, 0xC1396C28u, 0x12F58995u, 0x9CF04975u,
    0xAC3434E9u, 0x6E64479Eu, 0x657C0710u, 0x7AE96A2Bu
};

/* ---------- Keccak-256 (32-bit lop3 + funnelshift + dual-issue rol64_1, specialized r=0 and r=23) ---------- */
@@KRC@@
@@KECCAK_FN@@

/* ---------- SHA-256d (specialized for 21-byte 0x41||h160 input, returning only first u32) ---------- */

__device__ __constant__ u32 SHA_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u };

#define ROR32(x, n) __funnelshift_r((x), (x), (n))

__device__ __forceinline__ u32 sha256d_chk_inline(u32 ab0, u32 ab1, u32 ab2, u32 ab3, u32 ab4) {
    u32 W[16];
    W[0] = 0x41000000u | (ab0 >> 8);
    W[1] = __funnelshift_r(ab1, ab0, 8);
    W[2] = __funnelshift_r(ab2, ab1, 8);
    W[3] = __funnelshift_r(ab3, ab2, 8);
    W[4] = __funnelshift_r(ab4, ab3, 8);
    W[5] = (ab4 << 24) | 0x00800000u;
    #pragma unroll
    for (int i = 6; i < 15; i++) W[i] = 0u;
    W[15] = 168u;

    u32 a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    u32 e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            const u32 w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            const u32 s0 = lop3_xor3(ROR32(w15, 7), ROR32(w15, 18), w15 >> 3);
            const u32 s1 = lop3_xor3(ROR32(w2, 17), ROR32(w2, 19), w2 >> 10);
            W[i & 15] = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
        }
        const u32 S1 = lop3_xor3(ROR32(e, 6), ROR32(e, 11), ROR32(e, 25));
        const u32 t1 = h + S1 + lop3_ch(e, f, g) + SHA_K[i] + W[i & 15];
        const u32 S0 = lop3_xor3(ROR32(a, 2), ROR32(a, 13), ROR32(a, 22));
        const u32 t2 = S0 + lop3_mj(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    W[0] = 0x6a09e667u + a; W[1] = 0xbb67ae85u + b; W[2] = 0x3c6ef372u + c; W[3] = 0xa54ff53au + d;
    W[4] = 0x510e527fu + e; W[5] = 0x9b05688cu + f; W[6] = 0x1f83d9abu + g; W[7] = 0x5be0cd19u + h;
    W[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) W[i] = 0u;
    W[15] = 256u;

    a = 0x6a09e667u; b = 0xbb67ae85u; c = 0x3c6ef372u; d = 0xa54ff53au;
    e = 0x510e527fu; f = 0x9b05688cu; g = 0x1f83d9abu; h = 0x5be0cd19u;
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            const u32 w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            const u32 s0 = lop3_xor3(ROR32(w15, 7), ROR32(w15, 18), w15 >> 3);
            const u32 s1 = lop3_xor3(ROR32(w2, 17), ROR32(w2, 19), w2 >> 10);
            W[i & 15] = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
        }
        const u32 S1 = lop3_xor3(ROR32(e, 6), ROR32(e, 11), ROR32(e, 25));
        const u32 t1 = h + S1 + lop3_ch(e, f, g) + SHA_K[i] + W[i & 15];
        const u32 S0 = lop3_xor3(ROR32(a, 2), ROR32(a, 13), ROR32(a, 22));
        const u32 t2 = S0 + lop3_mj(a, b, c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    return 0x6a09e667u + a;
}

__device__ __noinline__ u32 sha256d_chk_ni(u32 ab0, u32 ab1, u32 ab2, u32 ab3, u32 ab4) {
    return sha256d_chk_inline(ab0, ab1, ab2, ab3, ab4);
}

/* ---------- fast suffix test (division-free linear combination + single Barrett reduction) ---------- */

#if SUFFIX_LEN >= 6
__device__ __forceinline__ u64 fast_mod_v(u32 ab0, u32 ab1, u32 ab2, u32 ab3, u32 ab4) {
    const u64 m0 = (u64)ab0 * C5_LO;
    const u64 m1 = (u64)ab1 * C4_LO;
    const u64 m2 = (u64)ab2 * C3_LO;
    const u64 m3 = (u64)ab3 * C2_LO;
    const u64 m4 = (u64)ab4 * C1_LO;
    const u64 t0 = (u64)C0_LO + (u32)m0 + (u32)m1 + (u32)m2 + (u32)m3 + (u32)m4;
    const u64 t1 = (u64)C0_HI + (t0 >> 32)
                 + (m0 >> 32) + (m1 >> 32) + (m2 >> 32) + (m3 >> 32) + (m4 >> 32)
                 + (u64)ab0 * C5_HI + (u64)ab1 * C4_HI + (u64)ab2 * C3_HI + (u64)ab3 * C2_HI + (u64)ab4 * C1_HI;
    const u64 lo = (t1 << 32) | (u32)t0;
    const u64 hi = t1 >> 32;
    const u64 s_top = (hi << (65 - MOD_SHIFT)) | (lo >> (MOD_SHIFT - 1));
    const u64 q = __umul64hi(s_top, MOD_RECIP);
    u64 r = lo - q * MODM;
    if (r >= MODM) r -= MODM;
    return r;
}
#endif

/* Per-launch constant search context (double-buffered slots 0/1) - keeps target & output pointers
   out of live registers across the 24 Keccak rounds inside check_point6. */
struct SearchCtx {
    u64 target;
    u32 *hit_cnt;
    u32 *hit_tid;
    u64 *hit_off;
};
__device__ __constant__ SearchCtx G_CTX[2];

#if USE_WARP_BATCH
/* Shared-memory per-warp candidate queue (SoA, 0 bank conflict, 1 KB / warp). */
#define WARPS_PER_BLOCK (BLOCK / 32)
struct WarpQueue {
    u32 ab0[WARPS_PER_BLOCK][32];
    u32 ab1[WARPS_PER_BLOCK][32];
    u32 ab2[WARPS_PER_BLOCK][32];
    u32 ab3[WARPS_PER_BLOCK][32];
    u32 ab4[WARPS_PER_BLOCK][32];
    u32 exp_chk[WARPS_PER_BLOCK][32];
    u64 meta[WARPS_PER_BLOCK][32];  /* bits 60..62 = sym, bits 54..58 = orig lane, bits 0..53 = off */
};
__shared__ WarpQueue g_wq;

__device__ __forceinline__ void warp_flush(u32 warp, u32 lane, u32 count, u32 buf_id) {
    if (lane < count) {
        const u32 a0 = g_wq.ab0[warp][lane];
        const u32 a1 = g_wq.ab1[warp][lane];
        const u32 a2 = g_wq.ab2[warp][lane];
        const u32 a3 = g_wq.ab3[warp][lane];
        const u32 a4 = g_wq.ab4[warp][lane];
        const u32 ec = g_wq.exp_chk[warp][lane];
        if (sha256d_chk_ni(a0, a1, a2, a3, a4) == ec) {
            const u64 mt = g_wq.meta[warp][lane];
            const u32 warp_base_tid = (blockIdx.x * BLOCK + threadIdx.x) & ~31u;
            const u32 orig_tid = warp_base_tid | (u32)((mt >> 54) & 31ULL);
            const u64 clean_poff = mt & ~(31ULL << 54);
            const u32 slot = atomicAdd(G_CTX[buf_id].hit_cnt, 1u);
            if (slot < HIT_CAP) { G_CTX[buf_id].hit_tid[slot] = orig_tid; G_CTX[buf_id].hit_off[slot] = clean_poff; }
        }
    }
}
#endif

__device__ __forceinline__ u32 test_one(
    u32 xl0, u32 xl1, u32 xl2, u32 xl3, u32 xh0, u32 xh1, u32 xh2, u32 xh3,
    u32 yl0, u32 yl1, u32 yl2, u32 yl3, u32 yh0, u32 yh1, u32 yh2, u32 yh3,
    u64 packed_off, u32 qlen_buf
) {
    u32 ab0, ab1, ab2, ab3, ab4;
    keccak_h160_be(xl0, xl1, xl2, xl3, xh0, xh1, xh2, xh3,
                   yl0, yl1, yl2, yl3, yh0, yh1, yh2, yh3,
                   ab0, ab1, ab2, ab3, ab4);
    const u32 buf_id = qlen_buf >> 31;
    const u64 target = G_CTX[buf_id].target;
#if SUFFIX_LEN <= 5
    const u32 chk = sha256d_chk_inline(ab0, ab1, ab2, ab3, ab4);
    const u64 sum = C0_FULL + (u64)ab0 * C5_FULL + (u64)ab1 * C4_FULL
                  + (u64)ab2 * C3_FULL + (u64)ab3 * C2_FULL + (u64)ab4 * C1_FULL + (u64)chk;
    if ((sum % MODM) == target) {
        const u32 tid = blockIdx.x * BLOCK + threadIdx.x;
        const u32 slot = atomicAdd(G_CTX[buf_id].hit_cnt, 1u);
        if (slot < HIT_CAP) { G_CTX[buf_id].hit_tid[slot] = tid; G_CTX[buf_id].hit_off[slot] = packed_off; }
    }
    return qlen_buf;
#elif USE_WARP_BATCH
    const u64 vm = fast_mod_v(ab0, ab1, ab2, ab3, ab4);
    const u64 D = (target >= vm) ? (target - vm) : (target + MODM - vm);
    const int pass = ((D >> 32) == 0ULL);
    const u32 mask = __ballot_sync(0xFFFFFFFFu, pass);
    if (mask) {
        const u32 warp = threadIdx.x >> 5;
        const u32 lane = threadIdx.x & 31u;
        const u32 n = __popc(mask);
        const u32 rank = __popc(mask & ((1u << lane) - 1u));
        u32 qlen = qlen_buf & 0x7FFFFFFFu;
        const u64 mt = packed_off | ((u64)lane << 54);
        if (qlen + n < 32u) {
            if (pass) {
                const u32 idx = qlen + rank;
                g_wq.ab0[warp][idx] = ab0; g_wq.ab1[warp][idx] = ab1; g_wq.ab2[warp][idx] = ab2;
                g_wq.ab3[warp][idx] = ab3; g_wq.ab4[warp][idx] = ab4;
                g_wq.exp_chk[warp][idx] = (u32)D; g_wq.meta[warp][idx] = mt;
            }
            qlen += n;
        } else {
            const u32 space = 32u - qlen;
            if (pass && rank < space) {
                const u32 idx = qlen + rank;
                g_wq.ab0[warp][idx] = ab0; g_wq.ab1[warp][idx] = ab1; g_wq.ab2[warp][idx] = ab2;
                g_wq.ab3[warp][idx] = ab3; g_wq.ab4[warp][idx] = ab4;
                g_wq.exp_chk[warp][idx] = (u32)D; g_wq.meta[warp][idx] = mt;
            }
            __syncwarp();
            warp_flush(warp, lane, 32u, buf_id);
            __syncwarp();
            if (pass && rank >= space) {
                const u32 idx = rank - space;
                g_wq.ab0[warp][idx] = ab0; g_wq.ab1[warp][idx] = ab1; g_wq.ab2[warp][idx] = ab2;
                g_wq.ab3[warp][idx] = ab3; g_wq.ab4[warp][idx] = ab4;
                g_wq.exp_chk[warp][idx] = (u32)D; g_wq.meta[warp][idx] = mt;
            }
            qlen = qlen + n - 32u;
        }
        qlen_buf = (buf_id << 31) | qlen;
    }
    return qlen_buf;
#else
    const u64 vm = fast_mod_v(ab0, ab1, ab2, ab3, ab4);
    const u64 D = (target >= vm) ? (target - vm) : (target + MODM - vm);
    if ((D >> 32) == 0ULL) {
        if (sha256d_chk_ni(ab0, ab1, ab2, ab3, ab4) == (u32)D) {
            const u32 tid = blockIdx.x * BLOCK + threadIdx.x;
            const u32 slot = atomicAdd(G_CTX[buf_id].hit_cnt, 1u);
            if (slot < HIT_CAP) { G_CTX[buf_id].hit_tid[slot] = tid; G_CTX[buf_id].hit_off[slot] = packed_off; }
        }
    }
    return qlen_buf;
#endif
}

/* Check all 6 GLV/negation symmetric public keys of affine point (x, y) at offset cur_off.
   Only 4 live outer registers across the 24 Keccak rounds (s, cur_off[2], qlen_buf) -> 0 Keccak spill at 80 regs!
   Vectorized uint4 staging (128-bit LD.L/ST.L) cuts local-memory staging overhead by 4x. */
__device__ __noinline__ u32 check_point6(
    const u32 *x, const u32 *y, u64 cur_off, u32 qlen_buf
) {
    uint4 xs[3][2], ys[2][2];
    {
        u32 bx[8], b2x[8], ny[8];
        fe_mul(bx, x, BETA_C);
        fe_add(b2x, x, bx);
        fe_neg_nz(b2x, b2x);
        fe_neg_nz(ny, y);

        xs[0][0] = make_uint4(bswap32(x[7]),   bswap32(x[5]),   bswap32(x[3]),   bswap32(x[1]));
        xs[0][1] = make_uint4(bswap32(x[6]),   bswap32(x[4]),   bswap32(x[2]),   bswap32(x[0]));
        xs[1][0] = make_uint4(bswap32(bx[7]),  bswap32(bx[5]),  bswap32(bx[3]),  bswap32(bx[1]));
        xs[1][1] = make_uint4(bswap32(bx[6]),  bswap32(bx[4]),  bswap32(bx[2]),  bswap32(bx[0]));
        xs[2][0] = make_uint4(bswap32(b2x[7]), bswap32(b2x[5]), bswap32(b2x[3]), bswap32(b2x[1]));
        xs[2][1] = make_uint4(bswap32(b2x[6]), bswap32(b2x[4]), bswap32(b2x[2]), bswap32(b2x[0]));

        ys[0][0] = make_uint4(bswap32(y[7]),   bswap32(y[5]),   bswap32(y[3]),   bswap32(y[1]));
        ys[0][1] = make_uint4(bswap32(y[6]),   bswap32(y[4]),   bswap32(y[2]),   bswap32(y[0]));
        ys[1][0] = make_uint4(bswap32(ny[7]),  bswap32(ny[5]),  bswap32(ny[3]),  bswap32(ny[1]));
        ys[1][1] = make_uint4(bswap32(ny[6]),  bswap32(ny[4]),  bswap32(ny[2]),  bswap32(ny[0]));
    }

    #pragma unroll 1
    for (u32 s = 0u; s < 6u; s++) {
        const u32 xi = s >> 1;
        const u32 yi = s & 1u;
        const uint4 xl = xs[xi][0], xh = xs[xi][1];
        const uint4 yl = ys[yi][0], yh = ys[yi][1];
        qlen_buf = test_one(
            xl.x, xl.y, xl.z, xl.w, xh.x, xh.y, xh.z, xh.w,
            yl.x, yl.y, yl.z, yl.w, yh.x, yh.y, yh.z, yh.w,
            ((u64)s << 60) | cur_off, qlen_buf);
    }
    return qlen_buf;
}

/* ---------- group table: 1*G .. (GROUP-1)*G, and STEP*G = (2*GROUP-1)*G at index GROUP-1 ---------- */
@@GTAB@@

/* ---------- main search kernel ----------
   Each iteration checks center B (offset off) + symmetric pairs B ± d*G (d = 1..GROUP-1) using ONE shared
   Montgomery inversion (including the advance B_next = B + STEP*G at j = GROUP-1), each expanded 6-way via GLV+neg.
   Total checked per thread per iteration: 6 * (2*GROUP - 1) disjoint keys. */
extern "C" __global__ void __launch_bounds__(BLOCK, MINB)
tv_search(u32 *sx, u32 *sy, u64 *soff, u32 nthr, u32 iters, u32 buf_id)
{
    const u32 tid = blockIdx.x * BLOCK + threadIdx.x;
    if (tid >= nthr) return;

    u32 qlen_buf = buf_id << 31;

    u32 bx[8], by[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { bx[i] = sx[i * nthr + tid]; by[i] = sy[i * nthr + tid]; }
    u64 off = soff[tid];
    uint4 pre[GROUP - 1][2];
    u32 nbx[8], nby[8];

    #pragma unroll 1
    for (u32 it = 0; it < iters; it++) {
        /* 1. Check current center point B (zero EC cost) */
        qlen_buf = check_point6(bx, by, off, qlen_buf);

        /* 2. Prefix products of dx[j] = GTAB[j].x - bx for j = 0 .. GROUP-1 */
        u32 acc[8];
        {
            u32 gx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) gx[i] = GTAB[i];
            fe_sub(acc, gx, bx);
            pre[0][0] = make_uint4(acc[0], acc[1], acc[2], acc[3]);
            pre[0][1] = make_uint4(acc[4], acc[5], acc[6], acc[7]);
        }
        #pragma unroll 1
        for (int j = 1; j < GROUP; j++) {
            u32 gx[8], dx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) gx[i] = GTAB[j * 16 + i];
            fe_sub(dx, gx, bx);
            fe_mul(acc, acc, dx);
            if (j < GROUP - 1) {
                pre[j][0] = make_uint4(acc[0], acc[1], acc[2], acc[3]);
                pre[j][1] = make_uint4(acc[4], acc[5], acc[6], acc[7]);
            }
        }
        u32 inv[8];
        fe_inv(inv, acc);

        /* 3. j = GROUP-1: advance next center nb = B + STEP*G (not checked here; checked as center next iter) */
        {
            const int j = GROUP - 1;
            u32 gx[8], gy[8], idx[8], dx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) { gx[i] = GTAB[j * 16 + i]; gy[i] = GTAB[j * 16 + 8 + i]; }
            const uint4 p0 = pre[j - 1][0], p1 = pre[j - 1][1];
            const u32 pm[8] = { p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w };
            fe_mul(idx, inv, pm);
            fe_sub(dx, gx, bx);
            fe_mul(inv, inv, dx);

            u32 dy[8], lam[8], tmp[8];
            fe_sub(dy, gy, by);
            fe_mul(lam, dy, idx);
            fe_sqr(nbx, lam);
            fe_sub(nbx, nbx, bx);
            fe_sub(nbx, nbx, gx);
            fe_sub(tmp, bx, nbx);
            fe_mul(nby, lam, tmp);
            fe_sub(nby, nby, by);
        }

        /* 4. j = GROUP-2 .. 0: symmetric pair B + (j+1)*G and B - (j+1)*G sharing idx */
        #pragma unroll 1
        for (int j = GROUP - 2; j >= 0; j--) {
            u32 gx[8], gy[8], idx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) { gx[i] = GTAB[j * 16 + i]; gy[i] = GTAB[j * 16 + 8 + i]; }
            if (j > 0) {
                u32 dx[8];
                const uint4 p0 = pre[j - 1][0], p1 = pre[j - 1][1];
                const u32 pm[8] = { p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w };
                fe_mul(idx, inv, pm);
                fe_sub(dx, gx, bx);
                fe_mul(inv, inv, dx);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) idx[i] = inv[i];
            }

            u32 sumx[8], dy[8], lam[8], x3[8], y3[8], tmp[8];
            fe_add(sumx, bx, gx);

            /* B + (j+1)*G */
            fe_sub(dy, gy, by);
            fe_mul(lam, dy, idx);
            fe_sqr(x3, lam);
            fe_sub(x3, x3, sumx);
            fe_sub(tmp, bx, x3);
            fe_mul(y3, lam, tmp);
            fe_sub(y3, y3, by);
            qlen_buf = check_point6(x3, y3, off + (u64)(j + 1), qlen_buf);

            /* B - (j+1)*G = (bx, by) + (gx, -gy) */
            u32 ngy[8];
            fe_neg_nz(ngy, gy);
            fe_sub(dy, ngy, by);
            fe_mul(lam, dy, idx);
            fe_sqr(x3, lam);
            fe_sub(x3, x3, sumx);
            fe_sub(tmp, bx, x3);
            fe_mul(y3, lam, tmp);
            fe_sub(y3, y3, by);
            qlen_buf = check_point6(x3, y3, off - (u64)(j + 1), qlen_buf);
        }

        #pragma unroll
        for (int i = 0; i < 8; i++) { bx[i] = nbx[i]; by[i] = nby[i]; }
        off += (u64)(2 * GROUP - 1);
    }

#if USE_WARP_BATCH
    const u32 rem = qlen_buf & 0x7FFFFFFFu;
    if (rem > 0u) {
        __syncwarp();
        warp_flush(threadIdx.x >> 5, threadIdx.x & 31u, rem, buf_id);
    }
#endif

    #pragma unroll
    for (int i = 0; i < 8; i++) { sx[i * nthr + tid] = bx[i]; sy[i * nthr + tid] = by[i]; }
    soff[tid] = off;
}

#if USE_PTX
/* self-check of all PTX field primitives (mul, sqr, reduce, add, sub, neg) against portable C */
extern "C" __global__ void tv_mulcheck(u32 seed, u32 iters, u32 *bad)
{
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 s = seed ^ (tid * 0x9E3779B9u) ^ 0xA5A5A5A5u;
    if (s == 0u) s = 1u;
    u32 a[8], b[8], t1[16], t2[16], r1[8], r2[8];
    for (u32 it = 0; it < iters; it++) {
        #pragma unroll 1
        for (int i = 0; i < 8; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            u32 m = s & 7u;
            a[i] = (m == 0u) ? 0u : ((m == 1u) ? 0xFFFFFFFFu : ((m == 2u) ? 0x80000000u : s * 2654435761u));
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            m = s & 7u;
            b[i] = (m == 0u) ? 0u : ((m == 1u) ? 0xFFFFFFFFu : ((m == 2u) ? 0x7FFFFFFFu : s * 2246822519u));
        }
        /* 1. 256x256 mul + reduce */
        mul_wide_c(t1, a, b);
        mul_wide_ptx(t2, a, b);
        u32 diff = 0u;
        #pragma unroll
        for (int i = 0; i < 16; i++) diff |= (t1[i] ^ t2[i]);
        fe_reduce_c(r1, t1);
        fe_reduce_ptx(r2, t2);
        #pragma unroll
        for (int i = 0; i < 8; i++) diff |= (r1[i] ^ r2[i]);

        /* 2. 256-bit sqr + reduce */
        mul_wide_c(t1, a, a);
        sqr_wide_ptx(t2, a);
        #pragma unroll
        for (int i = 0; i < 16; i++) diff |= (t1[i] ^ t2[i]);
        fe_reduce_c(r1, t1);
        fe_reduce_ptx(r2, t2);
        #pragma unroll
        for (int i = 0; i < 8; i++) diff |= (r1[i] ^ r2[i]);

        /* 3. add / sub / neg on canonical field elements (< p) */
        u32 fa[8], fb[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) { fa[i] = r1[i]; }
        mul_wide_c(t1, b, b);
        fe_reduce_c(fb, t1);
        fe_add_c(r1, fa, fb);
        fe_add_ptx(r2, fa, fb);
        #pragma unroll
        for (int i = 0; i < 8; i++) diff |= (r1[i] ^ r2[i]);

        fe_sub_c(r1, fa, fb);
        fe_sub_ptx(r2, fa, fb);
        #pragma unroll
        for (int i = 0; i < 8; i++) diff |= (r1[i] ^ r2[i]);

        if ((fa[0] | fa[1] | fa[2] | fa[3] | fa[4] | fa[5] | fa[6] | fa[7]) != 0u) {
            fe_neg_nz_c(r1, fa);
            fe_neg_nz_ptx(r2, fa);
            #pragma unroll
            for (int i = 0; i < 8; i++) diff |= (r1[i] ^ r2[i]);
        }

        /* 4. rol64_1 dual-issue ADD.CC/ADDC check */
        u32 rl, rh;
        rol64_1(rl, rh, a[0], a[1]);
        const u64 w64 = ((u64)a[1] << 32) | (u64)a[0];
        const u64 exp64 = (w64 << 1) | (w64 >> 63);
        diff |= (rl ^ (u32)exp64) | (rh ^ (u32)(exp64 >> 32));

        if (diff) atomicAdd(bad, 1u);
    }
}
#endif
'''


def gen_ptx_mul_rows():
    """Return list of asm lines per row for the PTX 8x8 limb multiplier."""
    rows = []
    for i in range(8):
        a = []
        a.append("mad.lo.cc.u32 %0, %9, %17, %0;")
        for j in range(1, 8):
            a.append("madc.lo.cc.u32 %%%d, %%%d, %%17, %%%d;" % (j, 9 + j, j))
        a.append("addc.u32 %8, %8, 0;")
        a.append("mad.hi.cc.u32 %1, %9, %17, %1;")
        for j in range(1, 7):
            a.append("madc.hi.cc.u32 %%%d, %%%d, %%17, %%%d;" % (1 + j, 9 + j, 1 + j))
        a.append("madc.hi.u32 %8, %16, %17, %8;")
        rows.append(a)
    return rows


def gen_ptx_mul_src():
    L = []
    L.append("#if USE_PTX")
    L.append("/* 8x8 limb schoolbook multiplier using PTX carry chains (mad.lo.cc / madc.hi.cc), 2 chains per row */")
    L.append("__device__ __forceinline__ void mul_wide_ptx(u32 *t, const u32 *a, const u32 *b) {")
    L.append("    const u32 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];")
    L.append("    u32 " + ", ".join("t%d = 0u" % i for i in range(16)) + ";")
    for i, rows in enumerate(gen_ptx_mul_rows()):
        outs = ", ".join('"+r"(t%d)' % (i + k) for k in range(9))
        ins = ", ".join(['"r"(a%d)' % j for j in range(8)] + ['"r"(b[%d])' % i])
        L.append("    asm(")
        for ln in rows:
            L.append('        "%s\\n\\t"' % ln)
        L.append("        : %s" % outs)
        L.append("        : %s);" % ins)
    L.append("    " + " ".join("t[%d] = t%d;" % (i, i) for i in range(16)))
    L.append("}")
    L.append("#endif")
    return "\n".join(L)


def gen_ptx_sqr_src():
    """Generate dedicated 256-bit -> 512-bit PTX squaring (94 PTX ops vs 136 for general mul)."""
    L = []
    L.append("#if USE_PTX")
    L.append("__device__ __forceinline__ void sqr_wide_ptx(u32 *t, const u32 *a) {")
    L.append("    const u32 a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];")
    L.append("    u32 t0, " + ", ".join("t%d = 0u" % i for i in range(1, 15)) + ", t15;")
    # Step 1: off-diagonal products a[i]*a[j] for 0 <= i < j <= 7
    for i in range(7):
        cnt = 7 - i
        # outputs: t[2*i+1 .. i+8] (cnt + 1 registers)
        outs = ", ".join('"+r"(t%d)' % (2 * i + 1 + k) for k in range(cnt + 1))
        # inputs: a[i], then a[i+1 .. 7]
        ai_op = cnt + 1
        aj_base = cnt + 2
        ins = ", ".join(['"r"(a%d)' % i] + ['"r"(a%d)' % j for j in range(i + 1, 8)])
        rows = []
        if cnt == 1:
            # i = 6, j = 7: single product a6*a7 into (t13, t14)
            rows.append("mad.lo.cc.u32 %%0, %%%d, %%%d, %%0;" % (ai_op, aj_base))
            rows.append("madc.hi.u32 %%1, %%%d, %%%d, %%1;" % (ai_op, aj_base))
        else:
            # lo chain: t[2*i+1 + k] += lo(a[i] * a[i+1+k]) for k = 0..cnt-1, carry into t[i+8] (%cnt)
            rows.append("mad.lo.cc.u32 %%0, %%%d, %%%d, %%0;" % (ai_op, aj_base))
            for k in range(1, cnt):
                rows.append("madc.lo.cc.u32 %%%d, %%%d, %%%d, %%%d;" % (k, ai_op, aj_base + k, k))
            rows.append("addc.u32 %%%d, %%%d, 0;" % (cnt, cnt))
            # hi chain: t[2*i+2 + k] += hi(a[i] * a[i+1+k]) for k = 0..cnt-1
            rows.append("mad.hi.cc.u32 %%1, %%%d, %%%d, %%1;" % (ai_op, aj_base))
            for k in range(1, cnt - 1):
                rows.append("madc.hi.cc.u32 %%%d, %%%d, %%%d, %%%d;" % (k + 1, ai_op, aj_base + k, k + 1))
            rows.append("madc.hi.u32 %%%d, %%%d, %%%d, %%%d;" % (cnt, ai_op, aj_base + cnt - 1, cnt))
        L.append("    asm(")
        for ln in rows:
            L.append('        "%s\\n\\t"' % ln)
        L.append("        : %s" % outs)
        L.append("        : %s);" % ins)
    # Step 2: double off-diagonal sum (t[1..14] <<= 1, carry into t15)
    L.append("    asm(")
    L.append('        "add.cc.u32 %0, %0, %0;\\n\\t"')
    for k in range(1, 14):
        L.append('        "addc.cc.u32 %%%d, %%%d, %%%d;\\n\\t"' % (k, k, k))
    L.append('        "addc.u32 %14, 0, 0;\\n\\t"')
    L.append("        : " + ", ".join('"+r"(t%d)' % k for k in range(1, 15)) + ', "=&r"(t15));')
    # Step 3: add diagonal squares a[0]^2 .. a[7]^2
    L.append("    asm(")
    L.append('        "mul.lo.u32 %0, %16, %16;\\n\\t"')
    L.append('        "mad.hi.cc.u32 %1, %16, %16, %1;\\n\\t"')
    for k in range(1, 7):
        L.append('        "madc.lo.cc.u32 %%%d, %%%d, %%%d, %%%d;\\n\\t"' % (2 * k, 16 + k, 16 + k, 2 * k))
        L.append('        "madc.hi.cc.u32 %%%d, %%%d, %%%d, %%%d;\\n\\t"' % (2 * k + 1, 16 + k, 16 + k, 2 * k + 1))
    L.append('        "madc.lo.cc.u32 %14, %23, %23, %14;\\n\\t"')
    L.append('        "madc.hi.u32 %15, %23, %23, %15;\\n\\t"')
    L.append('        : "=&r"(t0), ' + ", ".join('"+r"(t%d)' % k for k in range(1, 16)))
    L.append("        : " + ", ".join('"r"(a%d)' % k for k in range(8)) + ");")
    L.append("    " + " ".join("t[%d] = t%d;" % (i, i) for i in range(16)))
    L.append("}")
    L.append("#endif")
    return "\n".join(L)


_ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
_PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]


def _keccak_pi_rho_maps():
    src_to_dst = {0: (0, 0)}
    cur = 1
    for i in range(24):
        dst = _PILN[i]
        src_to_dst[cur] = (dst, _ROTC[i])
        cur = dst
    dst_to_src = {dst: (src, rot) for src, (dst, rot) in src_to_dst.items()}
    return src_to_dst, dst_to_src


def _rol32_pair_expr(lo, hi, rot):
    """Return C expressions (out_lo, out_hi) for 64-bit ROL((lo, hi), rot) using byte_perm / funnelshift."""
    if rot == 0:
        return lo, hi
    if rot == 8:
        return "__byte_perm(%s, %s, 0x6543)" % (hi, lo), "__byte_perm(%s, %s, 0x6543)" % (lo, hi)
    if rot == 56:
        return "__byte_perm(%s, %s, 0x4321)" % (lo, hi), "__byte_perm(%s, %s, 0x4321)" % (hi, lo)
    if rot < 32:
        return "__funnelshift_l(%s, %s, %d)" % (hi, lo, rot), "__funnelshift_l(%s, %s, %d)" % (lo, hi, rot)
    if rot == 32:
        return hi, lo
    s = rot - 32
    return "__funnelshift_l(%s, %s, %d)" % (lo, hi, s), "__funnelshift_l(%s, %s, %d)" % (hi, lo, s)


def gen_keccak_src(kunroll):
    """Generate in-place 32-bit lop3 + funnelshift + dual-issue rol64_1 Keccak-256 specialized for
    64-byte pubkey -> 20-byte BE TRON address."""
    src_to_dst, dst_to_src = _keccak_pi_rho_maps()
    L = []
    L.append("__device__ __forceinline__ void keccak_h160_be(")
    L.append("    u32 x0l, u32 x1l, u32 x2l, u32 x3l, u32 x0h, u32 x1h, u32 x2h, u32 x3h,")
    L.append("    u32 y0l, u32 y1l, u32 y2l, u32 y3l, u32 y0h, u32 y1h, u32 y2h, u32 y3h,")
    L.append("    u32 &ab0, u32 &ab1, u32 &ab2, u32 &ab3, u32 &ab4)")
    L.append("{")
    L.append("    u32 sl[25], sh[25];")
    # ---- Round 0 (specialized for 64-byte input + 0x01 at lane 8 + 0x80.. at lane 16) ----
    L.append("    {")
    L.append("        const u32 c0l = x0l ^ y1l, c0h = x0h ^ y1h;")
    L.append("        const u32 c1l = x1l ^ y2l, c1h = x1h ^ y2h ^ 0x80000000u;")
    L.append("        const u32 c2l = x2l ^ y3l, c2h = x2h ^ y3h;")
    L.append("        const u32 c3l = x3l ^ 1u,  c3h = x3h;")
    L.append("        const u32 c4l = y0l,       c4h = y0h;")
    L.append("        u32 r0l, r0h, r1l, r1h, r2l, r2h, r3l, r3h, r4l, r4h;")
    L.append("        rol64_1(r0l, r0h, c1l, c1h); const u32 d0l = c4l ^ r0l, d0h = c4h ^ r0h;")
    L.append("        rol64_1(r1l, r1h, c2l, c2h); const u32 d1l = c0l ^ r1l, d1h = c0h ^ r1h;")
    L.append("        rol64_1(r2l, r2h, c3l, c3h); const u32 d2l = c1l ^ r2l, d2h = c1h ^ r2h;")
    L.append("        rol64_1(r3l, r3h, c4l, c4h); const u32 d3l = c2l ^ r3l, d3h = c2h ^ r3h;")
    L.append("        rol64_1(r4l, r4h, c0l, c0h); const u32 d4l = c3l ^ r4l, d4h = c3h ^ r4h;")
    in_names = [
        ("x0l", "x0h"), ("x1l", "x1h"), ("x2l", "x2h"), ("x3l", "x3h"),
        ("y0l", "y0h"), ("y1l", "y1h"), ("y2l", "y2h"), ("y3l", "y3h"),
    ]
    for dst in range(25):
        src, rot = dst_to_src[dst]
        col = src % 5
        if src < 8:
            al = "(%s ^ d%dl)" % (in_names[src][0], col)
            ah = "(%s ^ d%dh)" % (in_names[src][1], col)
        elif src == 8:
            al = "(1u ^ d3l)"
            ah = "d3h"
        elif src == 16:
            al = "d1l"
            ah = "(0x80000000u ^ d1h)"
        else:
            al = "d%dl" % col
            ah = "d%dh" % col
        if rot == 1:
            L.append("        rol64_1(sl[%d], sh[%d], %s, %s);" % (dst, dst, al, ah))
        else:
            rl, rh = _rol32_pair_expr(al, ah, rot)
            L.append("        sl[%d] = %s; sh[%d] = %s;" % (dst, rl, dst, rh))
    for row in range(0, 25, 5):
        L.append("        {")
        L.append("            const u32 a0l = sl[%d], a1l = sl[%d], a2l = sl[%d], a3l = sl[%d], a4l = sl[%d];" % (
            row, row + 1, row + 2, row + 3, row + 4))
        L.append("            const u32 a0h = sh[%d], a1h = sh[%d], a2h = sh[%d], a3h = sh[%d], a4h = sh[%d];" % (
            row, row + 1, row + 2, row + 3, row + 4))
        for k in range(5):
            k1 = (k + 1) % 5
            k2 = (k + 2) % 5
            L.append("            sl[%d] = lop3_chi(a%dl, a%dl, a%dl); sh[%d] = lop3_chi(a%dh, a%dh, a%dh);" % (
                row + k, k, k1, k2, row + k, k, k1, k2))
        L.append("        }")
    L.append("        sl[0] ^= 1u;")
    L.append("    }")

    # ---- Rounds 1 .. 22 ----
    L.append("    #pragma unroll %d" % int(kunroll))
    L.append("    for (int r = 1; r < 23; r++) {")
    L.append("        {")
    for col in range(5):
        L.append("            const u32 c%dl = lop3_xor3(lop3_xor3(sl[%d], sl[%d], sl[%d]), sl[%d], sl[%d]);" % (
            col, col, col + 5, col + 10, col + 15, col + 20))
        L.append("            const u32 c%dh = lop3_xor3(lop3_xor3(sh[%d], sh[%d], sh[%d]), sh[%d], sh[%d]);" % (
            col, col, col + 5, col + 10, col + 15, col + 20))
    for col in range(5):
        p = (col + 4) % 5
        n = (col + 1) % 5
        L.append("            u32 rl%d, rh%d; rol64_1(rl%d, rh%d, c%dl, c%dh);" % (
            col, col, col, col, n, n))
        for y in range(5):
            idx = col + 5 * y
            L.append("            sl[%d] = lop3_xor3(sl[%d], c%dl, rl%d); sh[%d] = lop3_xor3(sh[%d], c%dh, rh%d);" % (
                idx, idx, p, col, idx, idx, p, col))
    L.append("        }")
    # In-place cycle rho + pi
    L.append("        {")
    L.append("            u32 tl = sl[1], th = sh[1];")
    for i in range(24):
        j = _PILN[i]
        rot = _ROTC[i]
        if rot == 1:
            L.append("            { const u32 ql = sl[%d], qh = sh[%d]; rol64_1(sl[%d], sh[%d], tl, th); tl = ql; th = qh; }" % (
                j, j, j, j))
        else:
            rl, rh = _rol32_pair_expr("tl", "th", rot)
            L.append("            { const u32 ql = sl[%d], qh = sh[%d]; sl[%d] = %s; sh[%d] = %s; tl = ql; th = qh; }" % (
                j, j, j, rl, j, rh))
    L.append("        }")
    # Row-by-row in-place chi
    for row in range(0, 25, 5):
        L.append("        {")
        L.append("            const u32 a0l = sl[%d], a1l = sl[%d], a2l = sl[%d], a3l = sl[%d], a4l = sl[%d];" % (
            row, row + 1, row + 2, row + 3, row + 4))
        L.append("            const u32 a0h = sh[%d], a1h = sh[%d], a2h = sh[%d], a3h = sh[%d], a4h = sh[%d];" % (
            row, row + 1, row + 2, row + 3, row + 4))
        for k in range(5):
            k1 = (k + 1) % 5
            k2 = (k + 2) % 5
            L.append("            sl[%d] = lop3_chi(a%dl, a%dl, a%dl); sh[%d] = lop3_chi(a%dh, a%dh, a%dh);" % (
                row + k, k, k1, k2, row + k, k, k1, k2))
        L.append("        }")
    L.append("        sl[0] ^= KRC_LO[r]; sh[0] ^= KRC_HI[r];")
    L.append("    }")

    # ---- Round 23 (specialized: only row 0 lanes -> digest bytes 12..31) ----
    L.append("    {")
    for col in range(5):
        L.append("        const u32 c%dl = lop3_xor3(lop3_xor3(sl[%d], sl[%d], sl[%d]), sl[%d], sl[%d]);" % (
            col, col, col + 5, col + 10, col + 15, col + 20))
        L.append("        const u32 c%dh = lop3_xor3(lop3_xor3(sh[%d], sh[%d], sh[%d]), sh[%d], sh[%d]);" % (
            col, col, col + 5, col + 10, col + 15, col + 20))
    for dst in range(5):
        src, rot = dst_to_src[dst]
        col = src % 5
        p = (col + 4) % 5
        n = (col + 1) % 5
        L.append("        u32 tl%d, th%d; rol64_1(tl%d, th%d, c%dl, c%dh);" % (dst, dst, dst, dst, n, n))
        al = "lop3_xor3(sl[%d], c%dl, tl%d)" % (src, p, dst)
        ah = "lop3_xor3(sh[%d], c%dh, th%d)" % (src, p, dst)
        rl, rh = _rol32_pair_expr(al, ah, rot)
        if dst == 1:
            L.append("        const u32 r1h = %s;" % rh)
        else:
            L.append("        const u32 r%dl = %s, r%dh = %s;" % (dst, rl, dst, rh))
    L.append("        ab0 = bswap32(lop3_chi(r1h, r2h, r3h));")
    L.append("        ab1 = bswap32(lop3_chi(r2l, r3l, r4l));")
    L.append("        ab2 = bswap32(lop3_chi(r2h, r3h, r4h));")
    L.append("        ab3 = bswap32(lop3_chi(r3l, r4l, r0l));")
    L.append("        ab4 = bswap32(lop3_chi(r3h, r4h, r0h));")
    L.append("    }")
    L.append("}")
    return "\n".join(L)


def build_source(suffix_len, group, block, minb, hit_cap, use_ptx, kunroll, gtab_points, warp_batch=None):
    """gtab_points: list of (x, y) ints for 1*G .. (GROUP-1)*G, and (2*GROUP-1)*G at index GROUP-1."""
    if not (1 <= suffix_len <= MAX_SUFFIX):
        raise ValueError("suffix_len 必须在 1..%d 之间" % MAX_SUFFIX)
    if not (2 <= group <= 512):
        raise ValueError("group 必须在 2..512 之间（当前 %s）" % group)
    if block < 32 or block > 1024 or (block % 32) != 0:
        raise ValueError("block 必须为 32..1024 之间且是 32 的倍数（当前 %s）" % block)
    if minb < 1:
        raise ValueError("minb 必须 >= 1（当前 %s）" % minb)
    if kunroll < 1:
        raise ValueError("kunroll 必须 >= 1（当前 %s）" % kunroll)
    if warp_batch is None:
        warp_batch = (suffix_len in (6, 7))
    words = []
    for (x, y) in gtab_points:
        for v in (x, y):
            for k in range(8):
                words.append("0x%08xu" % ((v >> (32 * k)) & 0xFFFFFFFF))
    lines = []
    for i in range(0, len(words), 8):
        lines.append("    " + ",".join(words[i:i + 8]))
    gtab = "__device__ __constant__ u32 GTAB[%d] = {\n%s };" % (len(words), ",\n".join(lines))

    M = 58 ** suffix_len
    C0 = (0x41 << 192) % M
    C1 = (1 << 32) % M
    C2 = (1 << 64) % M
    C3 = (1 << 96) % M
    C4 = (1 << 128) % M
    C5 = (1 << 160) % M
    pre_lines = [
        "#define SUFFIX_LEN %d" % suffix_len,
        "#define MODM %dULL" % M,
        "#define GROUP %d" % group,
        "#define BLOCK %d" % block,
        "#define MINB %d" % minb,
        "#define HIT_CAP %d" % hit_cap,
        "#define USE_PTX %d" % (1 if use_ptx else 0),
        "#define USE_WARP_BATCH %d" % (1 if (warp_batch and suffix_len >= 6) else 0),
    ]
    if suffix_len <= 5:
        for idx, val in enumerate((C0, C1, C2, C3, C4, C5)):
            pre_lines.append("#define C%d_FULL %dULL" % (idx, val))
    else:
        s_shift = M.bit_length() - 1
        recip = (1 << (s_shift + 63)) // M
        pre_lines.append("#define MOD_SHIFT %d" % s_shift)
        pre_lines.append("#define MOD_RECIP %dULL" % recip)
        for idx, val in enumerate((C0, C1, C2, C3, C4, C5)):
            pre_lines.append("#define C%d_LO %du" % (idx, val & 0xFFFFFFFF))
            pre_lines.append("#define C%d_HI %du" % (idx, val >> 32))
    pre = "\n".join(pre_lines) + "\n"

    krc_lo = ", ".join("0x%08xu" % (v & 0xFFFFFFFF) for v in _RC)
    krc_hi = ", ".join("0x%08xu" % (v >> 32) for v in _RC)
    krc = "__device__ __constant__ u32 KRC_LO[24] = { %s };\n__device__ __constant__ u32 KRC_HI[24] = { %s };" % (
        krc_lo, krc_hi)

    src = CUDA_SRC
    src = src.replace("@@PTXMUL@@", gen_ptx_mul_src())
    src = src.replace("@@PTXSQR@@", gen_ptx_sqr_src())
    src = src.replace("@@KRC@@", krc)
    src = src.replace("@@KECCAK_FN@@", gen_keccak_src(kunroll))
    src = src.replace("@@GTAB@@", gtab)
    return pre + src


# =====================================================================================
#  Part 3 : helpers + GPU runner (CuPy / NVRTC)
# =====================================================================================

HIT_CAP = 4096                 # max hits stored per kernel launch
STRIDE_BITS = 48               # thread t starts at key  k0 + t * 2^48  (disjoint key ranges)
MAX_SUFFIX = 10

DEFAULT_CFG = dict(group=96, block=256, minb=3, kunroll=1, ptx=True)

TUNE_GRID = [
    dict(group=96,  block=256, minb=3, kunroll=1, ptx=True),
    dict(group=64,  block=256, minb=3, kunroll=1, ptx=True),
    dict(group=128, block=256, minb=3, kunroll=1, ptx=True),
    dict(group=96,  block=256, minb=2, kunroll=1, ptx=True),
    dict(group=64,  block=256, minb=3, kunroll=2, ptx=True),
    dict(group=64,  block=128, minb=5, kunroll=1, ptx=True),
    dict(group=64,  block=128, minb=6, kunroll=1, ptx=True),
    dict(group=64,  block=256, minb=3, kunroll=1, ptx=False),
]


class GpuError(Exception):
    pass


def suffix_value(s):
    v = 0
    for ch in s:
        v = v * 58 + B58.index(ch)
    return v


def random_key():
    """Uniform 256-bit private key from the OS CSPRNG (kept away from 0 and N edges)."""
    lo = 1 << 40
    hi = N - (1 << 80)
    while True:
        k = int.from_bytes(secrets.token_bytes(32), 'big')
        if lo <= k < hi:
            return k


def gtab_points(group):
    """Return [1*G, 2*G, ..., (group-1)*G, (2*group-1)*G]."""
    pts, cur = [], None
    for _ in range(group - 1):
        cur = ec_add(cur, G)
        pts.append(cur)
    pts.append(ec_mul(2 * group - 1))
    return pts


def seed_points(k0, count, init_offset=0):
    """P_t = (k0 + init_offset + t * 2^STRIDE_BITS) * G  for t = 0..count-1 using batched doubling."""
    if count <= 0:
        return []
    step = ec_mul(1 << STRIDE_BITS)
    pts = [ec_mul((k0 + init_offset) % N)]
    while len(pts) < count:
        m = min(len(pts), count - len(pts))
        pts.extend(ec_batch_add_const(pts[:m], step))
        if len(pts) < count:
            step = ec_add(step, step)
    return pts


def fmt_num(x):
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= div:
            return "%.2f%s" % (x / div, unit)
    return "%.0f" % x


def fmt_dur(s):
    if s < 1:
        return "<1秒"
    if s < 90:
        return "%.0f秒" % s
    if s < 5400:
        return "%.1f分钟" % (s / 60)
    if s < 172800:
        return "%.1f小时" % (s / 3600)
    if s < 86400 * 400:
        return "%.1f天" % (s / 86400)
    return "%.1f年" % (s / 86400 / 365)


class GpuSearcher:
    def __init__(self, K, cfg, device=0, verbose=False, allow_ptx_fallback=True):
        try:
            import cupy as cp
            import numpy as np
        except Exception as e:  # noqa
            raise GpuError("无法导入 CuPy：%s\n请先安装：  pip install cupy-cuda12x   (需要 NVIDIA 驱动，Blackwell 显卡需 CUDA 12.8+ 的 NVRTC)" % e)
        self.cp, self.np = cp, np
        self.K = K
        self.cfg = dict(cfg)
        self.verbose = verbose
        try:
            ndev = cp.cuda.runtime.getDeviceCount()
        except Exception as e:  # noqa
            raise GpuError("找不到可用的 CUDA 设备：%s" % e)
        if ndev == 0 or device >= ndev:
            raise GpuError("没有找到编号为 %d 的 NVIDIA GPU" % device)
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        nm = props['name']
        self.gpu_name = nm.decode() if isinstance(nm, (bytes, bytearray)) else str(nm)
        self.sm_count = int(props['multiProcessorCount'])
        self.cc = (int(props['major']), int(props['minor']))
        self.device = device
        self.k0 = None
        self.T = 0
        self.step = 2 * self.cfg['group'] - 1
        self.keys_per_iter = self.step * 6
        self._build(allow_ptx_fallback)

    # ---------- compile ----------
    def _compile(self):
        cp, cfg = self.cp, self.cfg
        src = build_source(
            self.K, cfg['group'], cfg['block'], cfg['minb'], HIT_CAP, cfg['ptx'], cfg['kunroll'],
            gtab_points(cfg['group']),
            warp_batch=cfg.get('warp_batch'),
        )
        mod = cp.RawModule(code=src, options=('--std=c++14',), backend='nvrtc')
        self.mod = mod
        self.kern = mod.get_function('tv_search')
        self.attrs = dict(self.kern.attributes)          # forces compilation, raises on error
        self.mulcheck = mod.get_function('tv_mulcheck') if cfg['ptx'] else None
        self.blocks = self.sm_count * cfg['minb']
        self.T = self.blocks * cfg['block']

    def _check_ptx(self):
        cp, np = self.cp, self.np
        bad = cp.zeros(1, dtype=cp.uint32)
        self.mulcheck((64,), (128,), (np.uint32(secrets.randbits(32)), np.uint32(500), bad))
        cp.cuda.Stream.null.synchronize()
        return int(bad.get()[0]) == 0

    def _build(self, allow_fallback):
        try:
            self._compile()
            if self.cfg['ptx'] and not self._check_ptx():
                raise GpuError("PTX 域运算自检未通过")
        except ValueError as e:
            raise GpuError("参数配置无效：%s" % e)
        except Exception as e:  # noqa
            if self.cfg['ptx'] and allow_fallback:
                print("  [提示] PTX 加速路径不可用（%s），自动改用通用实现。" % str(e).strip().splitlines()[0][:200])
                self.cfg['ptx'] = False
                try:
                    self._compile()
                except Exception as e2:  # noqa
                    raise GpuError("GPU 内核编译失败：\n%s" % str(e2)[:3000])
            else:
                raise GpuError("GPU 内核编译/自检失败：\n%s" % str(e)[:3000])
        # buffers
        cp = self.cp
        self.hit_cnt = [cp.zeros(1, dtype=cp.uint32) for _ in range(2)]
        self.hit_tid = [cp.zeros(HIT_CAP, dtype=cp.uint32) for _ in range(2)]
        self.hit_off = [cp.zeros(HIT_CAP, dtype=cp.uint64) for _ in range(2)]
        self.g_ctx_ptr = self.mod.get_global('G_CTX')
        self.copy_stream = cp.cuda.Stream(non_blocking=True)
        if self.verbose:
            print("  内核资源: 寄存器/线程=%s  本地内存/线程=%s B  共享内存/块=%s B  最大线程/块=%s" % (
                self.attrs.get('num_regs'), self.attrs.get('local_size_bytes'),
                self.attrs.get('shared_size_bytes'), self.attrs.get('max_threads_per_block')))

    def _set_ctx(self, target):
        """Upload double-buffered SearchCtx[2] to __constant__ G_CTX before launching kernels."""
        cp, np = self.cp, self.np
        t = int(target) & 0xFFFFFFFFFFFFFFFF
        host = np.array([
            t, self.hit_cnt[0].data.ptr, self.hit_tid[0].data.ptr, self.hit_off[0].data.ptr,
            t, self.hit_cnt[1].data.ptr, self.hit_tid[1].data.ptr, self.hit_off[1].data.ptr,
        ], dtype=np.uint64)
        cp.cuda.runtime.memcpy(self.g_ctx_ptr.ptr, host.ctypes.data, host.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    # ---------- state ----------
    def seed(self, k0, nthreads=None, points=None):
        cp, np = self.cp, self.np
        T = nthreads or self.T
        if T <= 0 or (T % 32) != 0:
            raise ValueError("nthreads 必须为 32 的正整数倍（Warp 同步要求），当前为 %s" % T)
        init_off = self.cfg['group']  # center B_0 = k_t + (M + 1) = k_t + group -> covers [1 .. 2*group-1]
        if points is None:
            points = seed_points(k0, T, init_offset=init_off)
        xs = b''.join(p[0].to_bytes(32, 'little') for p in points[:T])
        ys = b''.join(p[1].to_bytes(32, 'little') for p in points[:T])
        ax = np.frombuffer(xs, dtype='<u4').reshape(T, 8).T.copy().reshape(-1)
        ay = np.frombuffer(ys, dtype='<u4').reshape(T, 8).T.copy().reshape(-1)
        self.sx = cp.asarray(ax.astype(np.uint32, copy=False))
        self.sy = cp.asarray(ay.astype(np.uint32, copy=False))
        self.so = cp.full(T, init_off, dtype=cp.uint64)
        self.k0 = k0
        self.T_active = T

    def _launch(self, b, iters):
        np = self.np
        block = self.cfg['block']
        grid = (self.T_active + block - 1) // block
        self.kern((grid,), (block,),
                  (self.sx, self.sy, self.so, np.uint32(self.T_active), np.uint32(iters), np.uint32(b)))

    def _read_hits(self, b):
        cnt = int(self.hit_cnt[b].get(stream=self.copy_stream)[0])
        n = min(cnt, HIT_CAP)
        if n == 0:
            return cnt, []
        tids = self.hit_tid[b][:n].get(stream=self.copy_stream)
        offs = self.hit_off[b][:n].get(stream=self.copy_stream)
        return cnt, [(int(t), int(o)) for t, o in zip(tids, offs)]

    def key_of(self, tid, packed_off):
        sym = packed_off >> 60
        off = packed_off & ((1 << 60) - 1)
        base_k = (self.k0 + tid * (1 << STRIDE_BITS) + off) % N
        return (base_k * SYM_MULTS[sym]) % N

    def run_sync(self, iters, target):
        self._set_ctx(target)
        self.hit_cnt[0].fill(0)
        self._launch(0, iters)
        self.cp.cuda.Stream.null.synchronize()
        return self._read_hits(0)

    # ---------- known-answer test of the *compiled* kernel (covers all 6 GLV/neg orbits & +, -, center) ----------
    def kat(self, rounds=2, sym=None, offset_kind=None):
        step = self.step
        g = self.cfg['group']
        if self.K <= 2:
            rounds = 1
        if sym is None:
            sym = secrets.randbelow(6)
        r_idx = secrets.randbelow(rounds)
        if offset_kind == 'center':
            exp_off = r_idx * step + g
        elif offset_kind == 'plus':
            exp_off = r_idx * step + g + secrets.randbelow(g - 1) + 1
        elif offset_kind == 'minus':
            exp_off = r_idx * step + secrets.randbelow(g - 1) + 1
        else:
            exp_off = secrets.randbelow(rounds * step) + 1

        lo = 1 << 40
        hi = N - (1 << 80)
        while True:
            kk = random_key()
            base_k = (kk * SYM_INV_MULTS[sym]) % N
            if lo <= base_k < hi:
                break
        addr = tron_address(kk)
        suffix = addr[-self.K:]
        target = suffix_value(suffix)
        k0 = base_k - exp_off
        # For tiny suffixes (K<=2), run a single full warp (32 threads) so hits stay far below HIT_CAP
        nthr = 32 if self.K <= 2 else min(self.T, self.cfg['block'] * 4)
        self.seed(k0, nthreads=nthr)
        cnt, hits = self.run_sync(rounds, target)
        exp_packed = (sym << 60) | exp_off
        ok = (0, exp_packed) in hits
        if ok:
            ok = (self.key_of(0, exp_packed) == kk)
        for tid, poff in hits[:48]:
            if not tron_address(self.key_of(tid, poff)).endswith(suffix):
                ok = False
        return ok

    # ---------- main search (pipelined) ----------
    def search(self, suffix, target, max_seconds=None, stop_on_hit=True, slice_s=0.25, progress=None, k0=None):
        cp = self.cp
        kpi = self.keys_per_iter
        if k0 is None:
            k0 = random_key()
        self.seed(k0)
        self._set_ctx(target)
        T = self.T_active
        stats = dict(total=0, gpu_s=0.0, speed=0.0, bad_hits=0, launches=0, _n=0, _t=0.0)
        inflight = []
        nl = [0]

        def enqueue(iters):
            b = nl[0] % 2
            nl[0] += 1
            self.hit_cnt[b].fill(0)
            s_ev, e_ev = cp.cuda.Event(), cp.cuda.Event()
            s_ev.record()
            self._launch(b, iters)
            e_ev.record()
            inflight.append((b, s_ev, e_ev, iters))

        t_wall0 = time.perf_counter()
        last_print = 0.0
        enqueue(1)
        enqueue(2)
        result = None
        try:
            while True:
                b, s_ev, e_ev, it = inflight.pop(0)
                while not e_ev.done:
                    time.sleep(0.001)
                sec = max(cp.cuda.get_elapsed_time(s_ev, e_ev) / 1000.0, 1e-6)
                stats['total'] += T * kpi * it
                stats['gpu_s'] += sec
                stats['launches'] += 1
                if stats['launches'] > 3:
                    stats['_n'] += T * kpi * it
                    stats['_t'] += sec
                    stats['speed'] = stats['_n'] / stats['_t']
                if stop_on_hit:
                    cnt, hits = self._read_hits(b)
                    if cnt:
                        good = None
                        for tid, poff in hits[:32]:
                            key = self.key_of(tid, poff)
                            addr = tron_address(key)          # independent CPU re-derivation
                            if addr.endswith(suffix):
                                good = (key, addr)
                                break
                            stats['bad_hits'] += 1
                        if good:
                            result = dict(key=good[0], addr=good[1])
                            break
                        if stats['bad_hits'] >= 64:
                            raise GpuError("GPU 返回的命中无法通过 CPU 校验，已停止（请运行 --selftest 并检查驱动/显卡）")
                now = time.perf_counter()
                if max_seconds is not None and now - t_wall0 >= max_seconds:
                    break
                enqueue(int(max(1, min(it * slice_s / sec, it * 4))))
                if progress and now - last_print >= 0.5:
                    last_print = now
                    progress(stats, now - t_wall0)
        finally:
            cp.cuda.Stream.null.synchronize()
        stats['wall_s'] = time.perf_counter() - t_wall0
        return result, stats


# =====================================================================================
#  Part 4 : command line
# =====================================================================================

def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def tune_key(K):
    return "v2:k<=5" if K <= 5 else ("v2:k6-7" if K <= 7 else "v2:k>=8")


def load_tune(gpu_name, K):
    try:
        with open(os.path.join(script_dir(), "tron_vanity_tune.json"), "r", encoding="utf-8") as f:
            return json.load(f).get(gpu_name, {}).get(tune_key(K))
    except Exception:  # noqa
        return None


def save_tune(gpu_name, K, cfg, speed):
    path = os.path.join(script_dir(), "tron_vanity_tune.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:  # noqa
        data = {}
    entry = dict(cfg)
    entry["speed"] = speed
    data.setdefault(gpu_name, {})[tune_key(K)] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_suffix(s):
    s = s.strip()
    if not s:
        raise ValueError("后缀不能为空")
    if len(s) > MAX_SUFFIX:
        raise ValueError("后缀最长支持 %d 位" % MAX_SUFFIX)
    bad = [c for c in s if c not in B58]
    if bad:
        raise ValueError("后缀含有非法字符 %s ：Base58 字母表不含 0 (零)、O (大写欧)、I (大写艾)、l (小写L)，且区分大小写" % " ".join(sorted(set(bad))))
    return s


def make_progress(K, suffix):
    expected = 58.0 ** K

    def cb(stats, wall):
        sp = stats["speed"]
        tried = stats["total"]
        p = 1.0 - math.exp(-tried / expected)
        eta = ("平均约需 %s" % fmt_dur(expected / sp)) if sp > 0 else "测速中"
        sys.stdout.write("\r  速度 %7.1f MH/s | 已尝试 %8s | 用时 %-9s | 已累计命中概率 %5.1f%% | %s        " % (
            sp / 1e6, fmt_num(tried), fmt_dur(wall), p * 100, eta))
        sys.stdout.flush()
    return cb


def save_result(path, key, addr):
    hexaddr = "41" + base58_decode_hex(addr)
    line = "时间: %s\n地址: %s\n地址(hex): %s\n私钥(hex): %064x\n\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), addr, hexaddr, key)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def base58_decode_hex(addr):
    n = 0
    for ch in addr:
        n = n * 58 + B58.index(ch)
    raw = n.to_bytes(25, 'big')
    return raw[1:21].hex()


def print_found(key, addr, tried, secs):
    print("\n")
    print("=" * 66)
    print("  找到了！  地址: %s" % addr)
    print("  私钥(hex): %064x" % key)
    print("  已用独立的纯 Python CPU 代码重新推导并校验：地址后缀一致 [OK]")
    print("  本次共尝试 %s 个地址，用时 %s" % (fmt_num(tried), fmt_dur(secs)))
    print("=" * 66)


def pick_cfg(args, gpu_name_hint, K):
    cfg = dict(DEFAULT_CFG)
    src = "默认"
    if gpu_name_hint:
        cached = load_tune(gpu_name_hint, K)
        if cached:
            for k in ("group", "block", "minb", "kunroll", "ptx"):
                if k in cached:
                    cfg[k] = cached[k]
            src = "调优缓存"
    for k, a in (("group", args.group), ("block", args.block), ("minb", args.minb), ("kunroll", args.kunroll)):
        if a is not None:
            cfg[k] = a
            src = "命令行"
    if args.no_ptx:
        cfg["ptx"] = False
    return cfg, src


def open_gpu(args, K):
    """Create the searcher; the GPU name is needed first to look at the tuning cache -> two-step."""
    try:
        import cupy as cp  # noqa
        import numpy  # noqa
    except Exception as e:  # noqa
        raise GpuError("无法导入 CuPy：%s\n请先安装：  pip install \"cupy-cuda12x[ctk]\"   (需要 NVIDIA 驱动；Blackwell 显卡需 CUDA 12.8+ 的 NVRTC)\n"
                       "若提示找不到 nvrtc / cudart 的 DLL，请再安装 CUDA Toolkit 12.8 或更新版本。" % e)
    try:
        props = cp.cuda.runtime.getDeviceProperties(args.device)
        nm = props['name']
        name = nm.decode() if isinstance(nm, (bytes, bytearray)) else str(nm)
    except Exception:  # noqa
        name = None
    cfg, src = pick_cfg(args, name, K)
    print("正在编译 GPU 内核（首次约十几秒到一分钟，同一位数的后缀之后会命中缓存）……")
    t0 = time.perf_counter()
    s = GpuSearcher(K, cfg, device=args.device, verbose=args.verbose)
    print("GPU: %s | SM 数量 %d | 计算能力 %d.%d | 并行线程 %d | 参数来源: %s" % (
        s.gpu_name, s.sm_count, s.cc[0], s.cc[1], s.T, src))
    print("配置: group=%d block=%d minb=%d kunroll=%d ptx=%s | 编译耗时 %.1fs" % (
        s.cfg['group'], s.cfg['block'], s.cfg['minb'], s.cfg['kunroll'], s.cfg['ptx'], time.perf_counter() - t0))
    return s


def do_selftest(args):
    print("[1/4] CPU 参考实现 ……", end=" ")
    assert keccak256(b'').hex() == 'c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470'
    assert tron_address(1) == 'TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC', "已知私钥=1 的地址不匹配"
    # Verify GLV endomorphism identity in CPU reference
    gx, gy = G
    assert ec_mul(LAM) == ((BETA * gx) % P, gy)
    assert ec_mul(LAM2) == ((P - (gx + BETA * gx) % P) % P, gy)
    print("通过（私钥=1 → TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC，GLV 恒等式 OK）")
    ok_all = True
    for step, K, label in (
        ("[2/4]", 3, "完整校验和"),
        ("[3/4]", 6, "Warp 聚合预过滤"),
        ("[4/4]", 8, "直通预过滤"),
    ):
        print("%s GPU 已知答案测试，后缀长度 %d（%s 路径，覆盖全部 6 路轨道 + 正/负/中心点）……" % (step, K, label))
        s = GpuSearcher(K, pick_cfg(args, None, K)[0], device=args.device, verbose=args.verbose)
        kinds = ('center', 'plus', 'minus', 'center', 'plus', 'minus')
        ok = all(s.kat(sym=sym, offset_kind=kinds[sym]) for sym in range(6))
        print("      %s" % ("通过 [OK]" if ok else "失败 [失败]"))
        ok_all &= ok
    print("全部通过，可以放心使用。" if ok_all else "有测试失败：请把上面的输出发给我。")
    return 0 if ok_all else 1


def do_tune(args, K):
    results = []
    print("开始自动调优（后缀长度 %d），共 %d 组配置，每组先编译再测速约 %.0f 秒……" % (K, len(TUNE_GRID), args.tune_seconds))
    gpu_name = None
    for i, cfg in enumerate(TUNE_GRID):
        tag = "group=%d block=%d minb=%d kunroll=%d ptx=%s" % (cfg['group'], cfg['block'], cfg['minb'], cfg['kunroll'], cfg['ptx'])
        try:
            s = GpuSearcher(K, cfg, device=args.device, verbose=False, allow_ptx_fallback=False)
            gpu_name = s.gpu_name
            if not s.kat():
                print("  [%d/%d] %s -> 已知答案测试失败，跳过" % (i + 1, len(TUNE_GRID), tag))
                continue
            _, st = s.search("", 0, max_seconds=args.tune_seconds, stop_on_hit=False)
            sp = st["speed"]
            print("  [%d/%d] %s -> %.1f MH/s  (寄存器 %s, 本地内存 %s B)" % (
                i + 1, len(TUNE_GRID), tag, sp / 1e6, s.attrs.get('num_regs'), s.attrs.get('local_size_bytes')))
            results.append((sp, cfg))
        except Exception as e:  # noqa
            print("  [%d/%d] %s -> 跳过：%s" % (i + 1, len(TUNE_GRID), tag, str(e).strip().splitlines()[0][:120]))
    if not results:
        print("没有可用的配置。")
        return 1
    results.sort(key=lambda r: -r[0])
    best_speed, best = results[0]
    save_tune(gpu_name, K, best, best_speed)
    print("\n最佳配置: %s -> %.1f MH/s，已写入 tron_vanity_tune.json，之后同长度后缀会自动使用。" % (best, best_speed / 1e6))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="TRON(波场)靓号地址生成器：GPU 搜索自定义后缀。示例： python tron_vanity_gpu.py 8888")
    ap.add_argument("suffix", nargs="?", help="想要的地址后缀（Base58 字符，区分大小写，最长 %d 位）" % MAX_SUFFIX)
    ap.add_argument("-n", "--count", type=int, default=1, help="要找几个地址（每个都用全新的随机起点，彼此独立），默认 1")
    ap.add_argument("-o", "--out", default=os.path.join(script_dir(), "tron_vanity_found.txt"), help="结果保存文件")
    ap.add_argument("--device", type=int, default=0, help="GPU 编号，默认 0")
    ap.add_argument("--bench", type=float, nargs="?", const=10.0, help="只测速（默认 10 秒），不保存任何结果")
    ap.add_argument("--tune", action="store_true", help="自动调优并缓存最佳参数（耗时几分钟，只需做一次）")
    ap.add_argument("--tune-seconds", type=float, default=3.0, help="调优时每组配置的测速时长")
    ap.add_argument("--selftest", action="store_true", help="运行自检（CPU 参考向量 + GPU 已知答案测试）")
    ap.add_argument("--group", type=int, help="每次批量求逆覆盖的对称点数（2..512，默认 %d）" % DEFAULT_CFG["group"])
    ap.add_argument("--block", type=int, help="每个 block 的线程数")
    ap.add_argument("--minb", type=int, help="每个 SM 驻留的 block 数（也决定寄存器上限）")
    ap.add_argument("--kunroll", type=int, help="Keccak 轮循环展开倍数")
    ap.add_argument("--no-ptx", action="store_true", help="不使用 PTX 进位链/指令级优化（用通用 C 实现）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    interactive = False
    if args.selftest:
        return do_selftest(args)

    suffix = args.suffix
    if suffix is None and args.bench is None:
        interactive = True
        print("TRON 靓号地址生成器（GPU）。Base58 字母表：123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        suffix = input("请输入想要的地址后缀（如 8888）： ")
    try:
        if args.bench is not None and suffix is None:
            suffix = "A" * 8
        suffix = validate_suffix(suffix)
    except ValueError as e:
        print("错误：%s" % e)
        return 2
    K = len(suffix)
    target = suffix_value(suffix)

    try:
        if args.tune:
            return do_tune(args, K)
        s = open_gpu(args, K)
        print("正在做内核已知答案自检 ……", end=" ")
        if not all(s.kat() for _ in range(2)):
            print("失败 [失败]\n内核在这块显卡/驱动上算出了错误结果，已中止（不会给出任何结果）。请尝试 --no-ptx，或运行 --selftest。")
            return 1
        print("通过 [OK]")

        if args.bench is not None:
            print("测速 %.0f 秒（后缀长度 %d）……" % (args.bench, K))
            _, st = s.search("", 0, max_seconds=args.bench, stop_on_hit=False, progress=make_progress(K, suffix))
            print("\n平均速度: %.1f MH/s（%s 个地址/秒）" % (st["speed"] / 1e6, fmt_num(st["speed"])))
            return 0

        expected = 58.0 ** K
        print("\n目标后缀: %s（%d 位）| 平均需要尝试约 %s 个地址（58^%d）" % (suffix, K, fmt_num(expected), K))
        if K >= 8:
            print("提示：%d 位后缀非常难，通常需要数小时到数天以上。" % K)
        found = 0
        for idx in range(args.count):
            if args.count > 1:
                print("\n-- 第 %d / %d 个 --" % (idx + 1, args.count))
            res, st = s.search(suffix, target, progress=make_progress(K, suffix))
            if res is None:
                break
            print_found(res["key"], res["addr"], st["total"], st["wall_s"])
            save_result(args.out, res["key"], res["addr"])
            print("  已追加保存到: %s" % args.out)
            found += 1
        return 0
    except KeyboardInterrupt:
        print("\n已手动停止。")
        return 130
    except GpuError as e:
        print("\n错误：%s" % e)
        return 1
    finally:
        if interactive:
            try:
                input("\n按回车键退出 ……")
            except Exception:  # noqa
                pass


if __name__ == "__main__":
    sys.exit(main())
