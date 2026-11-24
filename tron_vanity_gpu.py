#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRON (波场) 靓号地址生成器 —— NVIDIA GPU 版，自定义「后缀」，针对 RTX 50 系 (Blackwell) 优化
=====================================================================================

用法
----
    pip install "cupy-cuda12x[ctk]"          # 一次性安装（需要 NVIDIA 驱动 570+；RTX 5060 Ti 需要 CUDA 12.8+ 的 NVRTC）
    python tron_vanity_gpu.py 8888            # 找以 8888 结尾的地址
    python tron_vanity_gpu.py Abc -n 3        # 找 3 个以 Abc 结尾的地址（每个都用全新的随机起点）
    python tron_vanity_gpu.py --selftest      # 自检（CPU 参考向量 + GPU 已知答案测试）
    python tron_vanity_gpu.py --bench         # 只测速
    python tron_vanity_gpu.py 8888 --tune     # 自动调优并缓存最佳参数（可选，只需做一次）

原理
----
  * 每个 GPU 线程从一个随机私钥对应的公钥点出发，用「点加 G 的倍数 + 批量(Montgomery)求逆」连续枚举 128 个相邻私钥，
    摊销后每个候选地址只需约 6 次域乘法，而不是一次完整的标量乘。
  * 后缀的 Base58 数位只取决于 N mod 58^k（N 为 25 字节的 0x41|hash160|校验和），GPU 上直接用 64 位取模判断，不做完整 Base58 编码。
  * 后缀 >= 7 位时：先用 Keccak 结果算出「校验和应该是多少」，只有这个值落在 32 位范围内才继续算两次 SHA-256（约 99.8% 的候选可跳过）。
  * 起始私钥来自操作系统 CSPRNG（secrets.token_bytes，256 位），不是 profanity 那种 32 位种子；
    每找到一个地址就重新取随机起点，所以同一次运行里得到的多个地址之间互相独立。
  * GPU 报告的每一个命中，都会用独立的纯 Python CPU 代码从私钥重新推导地址并校验后才显示。

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
// ============ TRON vanity (suffix) search kernel ============
typedef unsigned int u32;
typedef unsigned long long u64;

#define Q5 656356768ULL            /* 58^5 */

__device__ __forceinline__ u32 bswap32(u32 x) {
#ifdef HOST_EMU
    return __builtin_bswap32(x);
#else
    return __byte_perm(x, 0, 0x0123);
#endif
}

/* ---------- secp256k1 field: p = 2^256 - 2^32 - 977, 8 x 32-bit limbs (little endian) ---------- */

__device__ __forceinline__ void fe_add(u32 *r, const u32 *a, const u32 *b) {
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

__device__ __forceinline__ void fe_sub(u32 *r, const u32 *a, const u32 *b) {
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

/* 256x256 -> 512 bit product, portable version (IMAD.WIDE friendly) */
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

@@PTXMUL@@

__device__ __forceinline__ void fe_reduce(u32 *r, const u32 *t) {
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

__device__ __forceinline__ void fe_mul(u32 *r, const u32 *a, const u32 *b) {
    u32 t[16];
#if USE_PTX
    mul_wide_ptx(t, a, b);
#else
    mul_wide_c(t, a, b);
#endif
    fe_reduce(r, t);
}

__device__ __noinline__ void fe_mul_ni(u32 *r, const u32 *a, const u32 *b) { fe_mul(r, a, b); }

__device__ __noinline__ void fe_sqrn(u32 *r, const u32 *a, int n) {
    #pragma unroll 1
    for (int i = 0; i < 8; i++) r[i] = a[i];
    #pragma unroll 1
    for (int i = 0; i < n; i++) fe_mul(r, r, r);
}

/* r = a^(p-2) = a^-1 (a != 0).  addition chain: 255 squarings + 15 multiplications */
__device__ __noinline__ void fe_inv(u32 *r, const u32 *a) {
    u32 x2[8], x3[8], x6[8], x9[8], x11[8], x22[8], x44[8], x88[8], x176[8], x220[8], x223[8], t1[8];
    fe_sqrn(x2, a, 1);     fe_mul_ni(x2, x2, a);
    fe_sqrn(x3, x2, 1);    fe_mul_ni(x3, x3, a);
    fe_sqrn(x6, x3, 3);    fe_mul_ni(x6, x6, x3);
    fe_sqrn(x9, x6, 3);    fe_mul_ni(x9, x9, x3);
    fe_sqrn(x11, x9, 2);   fe_mul_ni(x11, x11, x2);
    fe_sqrn(x22, x11, 11); fe_mul_ni(x22, x22, x11);
    fe_sqrn(x44, x22, 22); fe_mul_ni(x44, x44, x22);
    fe_sqrn(x88, x44, 44); fe_mul_ni(x88, x88, x44);
    fe_sqrn(x176, x88, 88);  fe_mul_ni(x176, x176, x88);
    fe_sqrn(x220, x176, 44); fe_mul_ni(x220, x220, x44);
    fe_sqrn(x223, x220, 3);  fe_mul_ni(x223, x223, x3);
    fe_sqrn(t1, x223, 23); fe_mul_ni(t1, t1, x22);
    fe_sqrn(t1, t1, 5);    fe_mul_ni(t1, t1, a);
    fe_sqrn(t1, t1, 3);    fe_mul_ni(t1, t1, x2);
    fe_sqrn(t1, t1, 2);    fe_mul_ni(r, a, t1);
}

/* ---------- Keccak-256 (original Keccak padding, as used by Ethereum / TRON) ---------- */

__device__ __constant__ u64 KRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

__device__ __forceinline__ void keccakf(u64 *st) {
    #pragma unroll @@KUNROLL@@
    for (int r = 0; r < 24; r++) {
        u64 bc0, bc1, bc2, bc3, bc4, t;
        bc0 = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc1 = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc2 = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc3 = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc4 = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];
        t = bc4 ^ ROL64(bc1, 1); st[0] ^= t; st[5] ^= t; st[10] ^= t; st[15] ^= t; st[20] ^= t;
        t = bc0 ^ ROL64(bc2, 1); st[1] ^= t; st[6] ^= t; st[11] ^= t; st[16] ^= t; st[21] ^= t;
        t = bc1 ^ ROL64(bc3, 1); st[2] ^= t; st[7] ^= t; st[12] ^= t; st[17] ^= t; st[22] ^= t;
        t = bc2 ^ ROL64(bc4, 1); st[3] ^= t; st[8] ^= t; st[13] ^= t; st[18] ^= t; st[23] ^= t;
        t = bc3 ^ ROL64(bc0, 1); st[4] ^= t; st[9] ^= t; st[14] ^= t; st[19] ^= t; st[24] ^= t;
        /* rho + pi */
        t = st[1];
        @@RHOPI@@
        /* chi */
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            u64 a0 = st[j], a1 = st[j + 1], a2 = st[j + 2], a3 = st[j + 3], a4 = st[j + 4];
            st[j]     = a0 ^ (~a1 & a2);
            st[j + 1] = a1 ^ (~a2 & a3);
            st[j + 2] = a2 ^ (~a3 & a4);
            st[j + 3] = a3 ^ (~a4 & a0);
            st[j + 4] = a4 ^ (~a0 & a1);
        }
        st[0] ^= KRC[r];
    }
}

/* x,y: public key coordinates (8 LE limbs each) -> 5 words w[] holding the 20 address bytes (LE byte order per word) */
__device__ __forceinline__ void keccak_h160(const u32 *x, const u32 *y, u32 *w) {
    u64 st[25];
    st[0] = ((u64)bswap32(x[6]) << 32) | bswap32(x[7]);
    st[1] = ((u64)bswap32(x[4]) << 32) | bswap32(x[5]);
    st[2] = ((u64)bswap32(x[2]) << 32) | bswap32(x[3]);
    st[3] = ((u64)bswap32(x[0]) << 32) | bswap32(x[1]);
    st[4] = ((u64)bswap32(y[6]) << 32) | bswap32(y[7]);
    st[5] = ((u64)bswap32(y[4]) << 32) | bswap32(y[5]);
    st[6] = ((u64)bswap32(y[2]) << 32) | bswap32(y[3]);
    st[7] = ((u64)bswap32(y[0]) << 32) | bswap32(y[1]);
    st[8] = 1ULL;
    #pragma unroll
    for (int i = 9; i < 16; i++) st[i] = 0ULL;
    st[16] = 0x8000000000000000ULL;
    #pragma unroll
    for (int i = 17; i < 25; i++) st[i] = 0ULL;
    keccakf(st);
    w[0] = (u32)(st[1] >> 32);            /* digest bytes 12..15 */
    w[1] = (u32)st[2];  w[2] = (u32)(st[2] >> 32);
    w[3] = (u32)st[3];  w[4] = (u32)(st[3] >> 32);
}

/* ---------- SHA-256 (single block) ---------- */

__device__ __constant__ u32 SHA_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u };

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

__device__ __forceinline__ void sha256_block(u32 *st, u32 *W) {
    st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;
    st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;
    u32 a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            const u32 w15 = W[(i - 15) & 15], w2 = W[(i - 2) & 15];
            const u32 s0 = ROR32(w15, 7) ^ ROR32(w15, 18) ^ (w15 >> 3);
            const u32 s1 = ROR32(w2, 17) ^ ROR32(w2, 19) ^ (w2 >> 10);
            W[i & 15] = W[i & 15] + s0 + W[(i - 7) & 15] + s1;
        }
        const u32 S1 = ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25);
        const u32 ch = (e & f) ^ (~e & g);
        const u32 t1 = h + S1 + ch + SHA_K[i] + W[i & 15];
        const u32 S0 = ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22);
        const u32 mj = (a & b) ^ (a & c) ^ (b & c);
        const u32 t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

/* ab0..ab4: the 20 address bytes as 5 big-endian words.  returns the 4 checksum bytes as a big-endian u32 */
__device__ __forceinline__ u32 sha256d_chk(u32 ab0, u32 ab1, u32 ab2, u32 ab3, u32 ab4) {
    u32 W[16], st[8], st2[8];
    W[0] = 0x41000000u | (ab0 >> 8);
    W[1] = (ab0 << 24) | (ab1 >> 8);
    W[2] = (ab1 << 24) | (ab2 >> 8);
    W[3] = (ab2 << 24) | (ab3 >> 8);
    W[4] = (ab3 << 24) | (ab4 >> 8);
    W[5] = (ab4 << 24) | 0x00800000u;
    #pragma unroll
    for (int i = 6; i < 15; i++) W[i] = 0u;
    W[15] = 168u;
    sha256_block(st, W);
    #pragma unroll
    for (int i = 0; i < 8; i++) W[i] = st[i];
    W[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) W[i] = 0u;
    W[15] = 256u;
    sha256_block(st2, W);
    return st2[0];
}

/* ---------- suffix test ----------
   address = Base58( 0x41 || h160 || chk ),  N = that 25-byte number.  Suffix of length K matches  <=>  N mod 58^K == target. */

__device__ __forceinline__ int check_candidate(const u32 *x, const u32 *y, u64 target) {
    u32 w[5];
    keccak_h160(x, y, w);
    const u32 ab0 = bswap32(w[0]), ab1 = bswap32(w[1]), ab2 = bswap32(w[2]), ab3 = bswap32(w[3]), ab4 = bswap32(w[4]);
#if SUFFIX_LEN <= 5
    const u32 chk = sha256d_chk(ab0, ab1, ab2, ab3, ab4);
    u64 r = 0x41ULL;
    r = ((r << 32) | ab0) % Q5;
    r = ((r << 32) | ab1) % Q5;
    r = ((r << 32) | ab2) % Q5;
    r = ((r << 32) | ab3) % Q5;
    r = ((r << 32) | ab4) % Q5;
    r = ((r << 32) | chk) % Q5;
    return (r % MODM) == target;
#else
    /* v = N - chk.  Need  chk == (target - v) mod 58^K ; since 58^K > 2^32 this is only possible when that value < 2^32,
       so SHA-256 is only evaluated for the (very few) candidates that survive. */
    u64 cur, qq, r = 0x41ULL;
    u32 q5, q4, q3, q2, q1, q0;
    cur = (r << 32) | ab0;  qq = cur / Q5;  r = cur - qq * Q5;  q5 = (u32)qq;
    cur = (r << 32) | ab1;  qq = cur / Q5;  r = cur - qq * Q5;  q4 = (u32)qq;
    cur = (r << 32) | ab2;  qq = cur / Q5;  r = cur - qq * Q5;  q3 = (u32)qq;
    cur = (r << 32) | ab3;  qq = cur / Q5;  r = cur - qq * Q5;  q2 = (u32)qq;
    cur = (r << 32) | ab4;  qq = cur / Q5;  r = cur - qq * Q5;  q1 = (u32)qq;
    cur = (r << 32);        qq = cur / Q5;  r = cur - qq * Q5;  q0 = (u32)qq;
    const u64 r1 = r;
    u64 r2 = q5;
    r2 = ((r2 << 32) | q4) % Q5;
    r2 = ((r2 << 32) | q3) % Q5;
    r2 = ((r2 << 32) | q2) % Q5;
    r2 = ((r2 << 32) | q1) % Q5;
    r2 = ((r2 << 32) | q0) % Q5;
    u64 vm = (r2 * Q5 + r1) % MODM;
    const u64 D = (target >= vm) ? (target - vm) : (target + MODM - vm);
    if (D >> 32) return 0;
    return sha256d_chk(ab0, ab1, ab2, ab3, ab4) == (u32)D;
#endif
}

/* ---------- group table: (j+1)*G for j = 0..GROUP-1, affine, x limbs then y limbs ---------- */
@@GTAB@@

/* ---------- main search kernel ----------
   Every thread owns a base point B (= key kb*G) and walks  B + (j+1)*G  using one shared (Montgomery) inversion per GROUP points. */
extern "C" __global__ void __launch_bounds__(BLOCK, MINB)
tv_search(u32 *sx, u32 *sy, u64 *soff, u32 nthr, u32 iters, u64 target,
          u32 *hit_cnt, u32 *hit_tid, u64 *hit_off)
{
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthr) return;

    u32 bx[8], by[8], nbx[8], nby[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { bx[i] = sx[i * nthr + tid]; by[i] = sy[i * nthr + tid]; nbx[i] = 0; nby[i] = 0; }
    u64 off = soff[tid];
    u32 pre[GROUP][8];

    #pragma unroll 1
    for (u32 it = 0; it < iters; it++) {
        u32 acc[8];
        {
            u32 gx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) gx[i] = GTAB[i];
            fe_sub(acc, gx, bx);
            #pragma unroll
            for (int i = 0; i < 8; i++) pre[0][i] = acc[i];
        }
        #pragma unroll 1
        for (int j = 1; j < GROUP; j++) {
            u32 gx[8], dx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) gx[i] = GTAB[j * 16 + i];
            fe_sub(dx, gx, bx);
            fe_mul(acc, acc, dx);
            #pragma unroll
            for (int i = 0; i < 8; i++) pre[j][i] = acc[i];
        }
        u32 inv[8];
        fe_inv(inv, acc);

        #pragma unroll 1
        for (int j = GROUP - 1; j >= 0; j--) {
            u32 gx[8], gy[8], idx[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) { gx[i] = GTAB[j * 16 + i]; gy[i] = GTAB[j * 16 + 8 + i]; }
            if (j > 0) {
                u32 dx[8];
                fe_mul(idx, inv, pre[j - 1]);
                fe_sub(dx, gx, bx);
                fe_mul(inv, inv, dx);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) idx[i] = inv[i];
            }
            u32 dy[8], lam[8], x3[8], y3[8], tmp[8];
            fe_sub(dy, gy, by);
            fe_mul(lam, dy, idx);
            fe_mul(x3, lam, lam);
            fe_sub(x3, x3, bx);
            fe_sub(x3, x3, gx);
            fe_sub(tmp, bx, x3);
            fe_mul(y3, lam, tmp);
            fe_sub(y3, y3, by);

            if (check_candidate(x3, y3, target)) {
                const u32 slot = atomicAdd(hit_cnt, 1u);
                if (slot < HIT_CAP) { hit_tid[slot] = tid; hit_off[slot] = off + (u64)(j + 1); }
            }
            if (j == GROUP - 1) {
                #pragma unroll
                for (int i = 0; i < 8; i++) { nbx[i] = x3[i]; nby[i] = y3[i]; }
            }
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) { bx[i] = nbx[i]; by[i] = nby[i]; }
        off += (u64)GROUP;
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) { sx[i * nthr + tid] = bx[i]; sy[i * nthr + tid] = by[i]; }
    soff[tid] = off;
}

#if USE_PTX
/* self-check of the PTX multiplier against the portable one (random + carry-heavy operands) */
extern "C" __global__ void tv_mulcheck(u32 seed, u32 iters, u32 *bad)
{
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 s = seed ^ (tid * 0x9E3779B9u) ^ 0xA5A5A5A5u;
    if (s == 0u) s = 1u;
    u32 a[8], b[8], t1[16], t2[16];
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
        mul_wide_c(t1, a, b);
        mul_wide_ptx(t2, a, b);
        u32 diff = 0u;
        #pragma unroll
        for (int i = 0; i < 16; i++) diff |= (t1[i] ^ t2[i]);
        if (diff) atomicAdd(bad, 1u);
    }
}
#endif
'''


def gen_ptx_mul_rows():
    """Return list of (asm_lines, window_indices) per row for the PTX 8x8 limb multiplier (also used by the unit test)."""
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


_ROTC = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44]
_PILN = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1]


def gen_rhopi():
    out = []
    for i in range(24):
        j = _PILN[i]
        out.append("{ u64 tmp = st[%d]; st[%d] = ROL64(t, %d); t = tmp; }" % (j, j, _ROTC[i]))
    return "\n        ".join(out)


def build_source(suffix_len, group, block, minb, hit_cap, use_ptx, kunroll, gtab_points):
    """gtab_points: list of (x, y) ints for 1*G .. GROUP*G"""
    words = []
    for (x, y) in gtab_points:
        for v in (x, y):
            for k in range(8):
                words.append("0x%08xu" % ((v >> (32 * k)) & 0xFFFFFFFF))
    lines = []
    for i in range(0, len(words), 8):
        lines.append("    " + ",".join(words[i:i + 8]))
    gtab = "__device__ __constant__ u32 GTAB[%d] = {\n%s };" % (len(words), ",\n".join(lines))
    pre = ("#define SUFFIX_LEN %d\n#define MODM %dULL\n#define GROUP %d\n#define BLOCK %d\n#define MINB %d\n"
           "#define HIT_CAP %d\n#define USE_PTX %d\n") % (suffix_len, 58 ** suffix_len, group, block, minb, hit_cap,
                                                         1 if use_ptx else 0)
    src = CUDA_SRC
    src = src.replace("@@PTXMUL@@", gen_ptx_mul_src())
    src = src.replace("@@KUNROLL@@", str(int(kunroll)))
    src = src.replace("@@RHOPI@@", gen_rhopi())
    src = src.replace("@@GTAB@@", gtab)
    return pre + src


# =====================================================================================
#  Part 3 : helpers + GPU runner (CuPy / NVRTC)
# =====================================================================================

HIT_CAP = 4096                 # max hits stored per kernel launch
STRIDE_BITS = 48               # thread t starts at key  k0 + t * 2^48  (disjoint key ranges)
MAX_SUFFIX = 10

DEFAULT_CFG = dict(group=128, block=128, minb=4, kunroll=2, ptx=True)

TUNE_GRID = [
    dict(group=128, block=128, minb=4, kunroll=2, ptx=True),
    dict(group=128, block=128, minb=4, kunroll=2, ptx=False),
    dict(group=64,  block=128, minb=4, kunroll=2, ptx=True),
    dict(group=256, block=128, minb=4, kunroll=2, ptx=True),
    dict(group=128, block=128, minb=3, kunroll=2, ptx=True),
    dict(group=128, block=128, minb=4, kunroll=4, ptx=True),
    dict(group=128, block=256, minb=2, kunroll=2, ptx=True),
]


class GpuError(Exception):
    pass


def suffix_value(s):
    v = 0
    for ch in s:
        v = v * 58 + B58.index(ch)
    return v


def random_key():
    """Uniform 256-bit private key from the OS CSPRNG (kept away from the group order's edge)."""
    while True:
        k = int.from_bytes(secrets.token_bytes(32), 'big')
        if 1 <= k < N - (1 << 80):
            return k


def gtab_points(n):
    pts, cur = [], None
    for _ in range(n):
        cur = ec_add(cur, G)
        pts.append(cur)
    return pts


def seed_points(k0, count):
    """P_t = (k0 + t * 2^STRIDE_BITS) * G  for t = 0..count-1 (one scalar mult + count-1 point additions)."""
    D = ec_mul(1 << STRIDE_BITS)
    cur = ec_mul(k0)
    pts = [cur]
    for _ in range(count - 1):
        cur = ec_add(cur, D)
        pts.append(cur)
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
        self._build(allow_ptx_fallback)

    # ---------- compile ----------
    def _compile(self):
        cp, cfg = self.cp, self.cfg
        src = build_source(self.K, cfg['group'], cfg['block'], cfg['minb'], HIT_CAP, cfg['ptx'], cfg['kunroll'],
                           gtab_points(cfg['group']))
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
        self.mulcheck((64,), (128,), (np.uint32(secrets.randbits(32)), np.uint32(1500), bad))
        cp.cuda.Stream.null.synchronize()
        return int(bad.get()[0]) == 0

    def _build(self, allow_fallback):
        try:
            self._compile()
            if self.cfg['ptx'] and not self._check_ptx():
                raise GpuError("PTX 乘法自检未通过")
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
        # hit read-back goes through a separate non-blocking stream, so it never waits for the kernel that is already queued behind it
        self.copy_stream = cp.cuda.Stream(non_blocking=True)
        if self.verbose:
            print("  内核资源: 寄存器/线程=%s  本地内存/线程=%s B  最大线程/块=%s" % (
                self.attrs.get('num_regs'), self.attrs.get('local_size_bytes'), self.attrs.get('max_threads_per_block')))

    # ---------- state ----------
    def seed(self, k0, nthreads=None, points=None):
        cp, np = self.cp, self.np
        T = nthreads or self.T
        if points is None:
            points = seed_points(k0, T)
        xs = b''.join(p[0].to_bytes(32, 'little') for p in points[:T])
        ys = b''.join(p[1].to_bytes(32, 'little') for p in points[:T])
        ax = np.frombuffer(xs, dtype='<u4').reshape(T, 8).T.copy().reshape(-1)
        ay = np.frombuffer(ys, dtype='<u4').reshape(T, 8).T.copy().reshape(-1)
        self.sx = cp.asarray(ax.astype(np.uint32, copy=False))
        self.sy = cp.asarray(ay.astype(np.uint32, copy=False))
        self.so = cp.zeros(T, dtype=cp.uint64)
        self.k0 = k0
        self.T_active = T

    def _launch(self, b, iters, target):
        np = self.np
        block = self.cfg['block']
        grid = (self.T_active + block - 1) // block
        self.kern((grid,), (block,),
                  (self.sx, self.sy, self.so, np.uint32(self.T_active), np.uint32(iters), np.uint64(target),
                   self.hit_cnt[b], self.hit_tid[b], self.hit_off[b]))

    def _read_hits(self, b):
        cnt = int(self.hit_cnt[b].get(stream=self.copy_stream)[0])
        n = min(cnt, HIT_CAP)
        if n == 0:
            return cnt, []
        tids = self.hit_tid[b][:n].get(stream=self.copy_stream)
        offs = self.hit_off[b][:n].get(stream=self.copy_stream)
        return cnt, [(int(t), int(o)) for t, o in zip(tids, offs)]

    def key_of(self, tid, off):
        return (self.k0 + tid * (1 << STRIDE_BITS) + off) % N

    def run_sync(self, iters, target):
        self.hit_cnt[0].fill(0)
        self._launch(0, iters, target)
        self.cp.cuda.Stream.null.synchronize()
        return self._read_hits(0)

    # ---------- known-answer test of the *compiled* kernel ----------
    def kat(self, rounds=2):
        group = self.cfg['group']
        kk = random_key()
        addr = tron_address(kk)
        suffix = addr[-self.K:]
        target = suffix_value(suffix)
        exp_off = secrets.randbelow(rounds * group) + 1
        k0 = kk - exp_off
        nthr = self.cfg['block'] if self.K <= 2 else self.T          # tiny suffixes => very many hits, keep the buffer from overflowing
        self.seed(k0, nthreads=nthr)
        cnt, hits = self.run_sync(rounds, target)
        ok = (0, exp_off) in hits
        for tid, off in hits[:48]:
            if not tron_address(self.key_of(tid, off)).endswith(suffix):
                ok = False
        return ok

    # ---------- main search (pipelined: the next kernel is always queued before we look at the previous result) ----------
    def search(self, suffix, target, max_seconds=None, stop_on_hit=True, slice_s=0.25, progress=None, k0=None):
        cp = self.cp
        group = self.cfg['group']
        if k0 is None:
            k0 = random_key()
        self.seed(k0)
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
            self._launch(b, iters, target)
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
                stats['total'] += T * group * it
                stats['gpu_s'] += sec
                stats['launches'] += 1
                if stats['launches'] > 3:          # the first (tiny, warm-up) launches are not counted in the speed estimate
                    stats['_n'] += T * group * it
                    stats['_t'] += sec
                    stats['speed'] = stats['_n'] / stats['_t']
                if stop_on_hit:
                    cnt, hits = self._read_hits(b)
                    if cnt:
                        good = None
                        for tid, off in hits[:32]:
                            key = self.key_of(tid, off)
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
    return "k<=5" if K <= 5 else ("k6" if K == 6 else "k>=7")


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
    print("[1/3] CPU 参考实现 ……", end=" ")
    assert keccak256(b'').hex() == 'c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470'
    assert tron_address(1) == 'TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC', "已知私钥=1 的地址不匹配"
    print("通过（私钥=1 → TMVQGm1qAQYVdetCeGRRkTWYYrLXuHK2HC）")
    ok_all = True
    for step, K in (("[2/3]", 3), ("[3/3]", 7)):
        print("%s GPU 已知答案测试，后缀长度 %d（%s 路径）……" % (step, K, "完整校验和" if K <= 5 else "预过滤"))
        s = GpuSearcher(K, pick_cfg(args, None, K)[0], device=args.device, verbose=args.verbose)
        ok = all(s.kat() for _ in range(3))
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
    ap.add_argument("--group", type=int, help="每次批量求逆覆盖的点数（调优参数）")
    ap.add_argument("--block", type=int, help="每个 block 的线程数")
    ap.add_argument("--minb", type=int, help="每个 SM 驻留的 block 数（也决定寄存器上限）")
    ap.add_argument("--kunroll", type=int, help="Keccak 轮循环展开倍数")
    ap.add_argument("--no-ptx", action="store_true", help="不使用 PTX 进位链乘法（用通用 C 实现）")
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
