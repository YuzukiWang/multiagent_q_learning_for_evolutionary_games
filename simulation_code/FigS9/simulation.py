

import os
import math
import json
import numpy as np
from time import time
import multiprocessing as mp

# =========================
# Configuration
# =========================
population_size = 1
mentor_instances = 1
mentor_strategy = list(range(14648))   # 0..120
max_num_strategy = 14649             # 0..121
s = 20.0                             # selection intensity
threshold_ratio = 0.8                # 活跃数 M < threshold_ratio*S 时，切至子集计算

# 文件路径 (按需修改)
strategy_txt_path = r'strategy0729.txt'
payoff_npy_path   = r'payoff_matrix_FULL.npy'

# =========================
# Load data
# =========================
if os.path.isfile(strategy_txt_path):
    with open(strategy_txt_path, 'r', encoding='utf-8') as f:
        strategy00 = json.load(f)
    strategy00_dict = {item[0]: item[1] for item in strategy00}
else:
    strategy00_dict = {}

# payoff = np.load(payoff_npy_path).astype(np.float32)  # (S, S)
# diag_pay = np.diag(payoff).astype(np.float32)
# S = payoff.shape[0]                                    # 实际策略总数(=maxS)

# dot 分块（活跃列分块大小，显存/内存吃紧可调小）
ACTIVE_COL_CHUNK = 2048

rng = np.random.default_rng()

FULL_NPY_PATH = r"payoff_matrix_FULL.npy"
# ================ 载入大矩阵 ================
A_orig = np.load(FULL_NPY_PATH, mmap_mode="r")   # memmap 读，省内存
S_orig = A_orig.shape[0]
focus_size = 8                                  # MTBR+7 自定义策略数量

# 默认“焦点索引”为原矩阵末尾 8 个（[S-8,...,S-1]）
focus_idx_orig = np.arange(S_orig - focus_size, S_orig, dtype=np.int32)
other_idx_orig = np.arange(0, S_orig - focus_size, dtype=np.int32)
perm = np.concatenate([focus_idx_orig, other_idx_orig]).astype(np.int32)

# 逆置换：new_idx -> old_idx ；old_idx -> new_idx（方便映射）
inv_perm = np.empty_like(perm)
inv_perm[perm] = np.arange(S_orig, dtype=np.int32)

# 重排后的矩阵（用切片视图避免复制；mmap + 切片 @ 仍高效）
# 注意：A_reordered 是“视图式”的高级索引，后续只读用于 @
A = A_orig[np.ix_(perm, perm)]
S = A.shape[0]
diag_pay = np.asarray(np.diag(A), dtype=np.float32)



S = A.shape[0]                              # 策略总数（14649）
diag_pay = np.asarray(np.diag(A), dtype=np.float32)

# # ================ 初始化种群 ================
# use_all_strategies_as_seed = True
# if use_all_strategies_as_seed:
#     # 每个策略 1 个体（N = S）；大规模时启动更慢，但演示最简单
#     types_cur = np.arange(S, dtype=np.int32)
# else:
#     # ——按你旧代码的风格初始化——
#     # 例：保留 1 个 Q-agent（索引无所谓，因我们用的是预生成矩阵，不再用 Q 的即时策略）
#     #    再加上一些你想种下的“导师策略集合”
#     # 建议：显式列出你希望出现的初始策略索引，例如常见 M1 或自定义策略的下标
#     seed_indices = [
#         # 举例：ALLC、ALLD、TFT、MTBR、某些自定义……
#         # 你可以用你保存的 strategy_index_FULL.csv 来查对应索引
#         # 下面只是演示：假设 0,1,2 是你想要的 M1，IDX_MTBR=S-8 是 MTBR（如果你按我之前定义），
#         # 请务必替换成真实索引！
#         0, 1, 2, S-8  # <-- 改成你的真实种子
#     ]
#     # 每个种子给若干个体：
#     per_seed = 10
#     types_cur = np.repeat(np.array(seed_indices, dtype=np.int32), per_seed)
    
# N = types_cur.size
# counts = np.bincount(types_cur, minlength=S).astype(np.int64)
# # ================ 结果数组（按需） ================

# max_episode = 100000

SPARSE_OUT = "avg_num_sparse.npy"
# SPARSE_STEPS_OUT = "record_steps.npy"

# avg_num_strategy = np.zeros((max_episode, S), dtype=np.float32)
# avg_reward_globle = np.zeros((max_episode,), dtype=np.float32)

# =========================
# Init population
# =========================
# N = population_size + mentor_instances * len(mentor_strategy)
# types_init = np.empty(N, dtype=np.int32)
# if population_size >= 1:
#     types_init[:population_size] = S - 1   # Q-agent 用最后一个编号
# if N - population_size > 0:
#     types_init[population_size:] = np.array(mentor_strategy, dtype=np.int32).repeat(mentor_instances)

# =========================
# Results
# =========================

# avg_num_strategy = np.zeros((max_episode, S), dtype=np.float32)
# avg_reward_globle = np.zeros((max_episode,), dtype=np.float32)

# repeattimes = 1
# i_num_strategy = np.zeros((repeattimes, max_episode, S), dtype=np.int32)
# i_reward_globle = np.zeros((repeattimes, max_episode), dtype=np.float32)

# =========================
# Mutation (vectorized)
# =========================

from numpy.lib.format import open_memmap

def build_record_steps(max_ep: int) -> np.ndarray:
    steps = []
    # 0-99 每代
    steps.extend(range(0, min(100, max_ep)))
    # 100-1000 每10
    if max_ep > 100:
        hi = min(1000, max_ep-1)
        steps.extend(range(100, hi+1, 10))
    # 1000-10000 每100
    if max_ep > 1000:
        hi = min(10_000, max_ep-1)
        steps.extend(range(1000, hi+1, 100))
    # 10000-100000 每1000
    if max_ep > 10_000:
        hi = max_ep - 1 
        steps.extend(range(10_000, hi+1, 1000))
    # 去重并排序（避免边界重叠时重复）
    steps = np.unique(np.array(steps, dtype=np.int64))
    return steps



def compute_dot_from_counts(counts: np.ndarray) -> np.ndarray:
    """
    dot = A @ counts，但只对活跃列计算：
    dot = sum_k counts[k] * A[:, k]
    以列分块（ACTIVE_COL_CHUNK）累加，复杂度 O(S·M)
    """
    active = np.flatnonzero(counts)
    if active.size == 0:
        return np.zeros(S, dtype=np.float32)
    dot = np.zeros(S, dtype=np.float32)
    # 分块累加
    for i0 in range(0, active.size, ACTIVE_COL_CHUNK):
        cols = active[i0:i0+ACTIVE_COL_CHUNK]
        # A[:, cols] 形状 (S, len(cols))，右乘 counts[cols] 得 (S,)
        dot += A[:, cols] @ counts[cols].astype(np.float32)
    return dot

def maybe_apply_mutation(types, counts, iepisode,
                         start_gen = 2000, event_rate = 0.1, share = 0.001, rng = rng):
    """事件级突变（整批向量化）：触发时，随机 m 个体=ceil(share*N) 改为同一 target"""
    if iepisode < start_gen:
        return
    if rng.random() >= event_rate:
        return
    N_local = types.size
    m = max(1, math.ceil(share * N_local))
    if rng.random() < 0.01:
        target = 0
    else:
        target = rng.integers(0, S)
    idx = rng.choice(N_local, size=m, replace=False)
    old = types[idx]
    if np.any(old != target):
        types[idx] = target
        # counts 增量更新
        dec = np.bincount(old, minlength=S).astype(np.int64)
        inc = np.zeros(S, dtype=np.int64)
        inc[target] = m
        counts -= dec
        counts += inc
        
        
def run_one(args):
    """
    单次实验，返回稀疏结果 (steps, S)
    - 这里我只返回稀疏结果（avg_num_sparse），
      避免巨大的全轨迹矩阵
    """
    seed, max_episode, record_steps = args
    rng = np.random.default_rng(seed)
    
    # 初始化
    N = S
    types_cur = np.arange(S, dtype=np.int32)
    counts = np.bincount(types_cur, minlength=S).astype(np.int64)

    # 稀疏采样点
    record_steps = build_record_steps(max_episode)
    sparse_result = np.zeros((len(record_steps), S), dtype=np.float32)

    # 演化
    step_to_row = {int(t): i for i, t in enumerate(record_steps)}
    for t in range(max_episode):
        dot = compute_dot_from_counts(counts)
        avg_by_type = (dot - diag_pay) / (N - 1)

        opp = rng.integers(0, N, size=N, dtype=np.int32)
        ti = types_cur[opp]
        tj = types_cur
        ri = avg_by_type[ti]
        rj = avg_by_type[tj]
        diff = s * (rj - ri)
        diff = np.clip(diff, -60.0, 60.0)
        p_ji = 1.0 / (np.exp(diff) + 1.0)
        flip = rng.random(N) < p_ji
        new_types = np.where(flip, ti, tj)

        types_cur = new_types
        counts = np.bincount(types_cur, minlength=S).astype(np.int64)

        maybe_apply_mutation(types_cur, counts, iepisode=t, rng=rng)

        row_id = step_to_row.get(t)
        if row_id is not None:
            sparse_result[row_id, :] = counts.astype(np.float32)

    return sparse_result

# ========== 多进程入口 ==========
if __name__ == "__main__":
    max_episode = 1_000_000

    repeats = 100

    workers = min(repeats, mp.cpu_count())  # 并行核数
    record_steps = build_record_steps(max_episode)

    start = time()
    args_list = [(seed, max_episode, record_steps) for seed in range(repeats)]
    K, S_local = len(record_steps), A.shape[0]
    sum_results = np.zeros((K, S_local), dtype=np.float64)

    with mp.Pool(processes=workers) as pool:
        for r in pool.imap_unordered(run_one, args_list):
            sum_results += r  # r: (K,S) float32

    avg_results = (sum_results / repeats).astype(np.float32)

    np.save("avg_num_sparse_mean.npy", avg_results)
    np.save("record_steps.npy", build_record_steps(max_episode))

    print(f"全部完成，用时 {time()-start:.2f} 秒")

