# build_weighted_adj_corr.py
from pathlib import Path
import numpy as np

data_dir = Path("/mnt/c/새 폴더/res")
A_norm_path = data_dir / "adjacency_norm.npy"
Y_path      = data_dir / "Y_samples.npy"   # (S, N, T_out, 2)  [traffic, speed]

out_path = data_dir / "adjacency_corr_norm.npy"

def sym_norm(A: np.ndarray) -> np.ndarray:
    # add self-loop
    A2 = A.copy()
    np.fill_diagonal(A2, A2.diagonal() + 1.0)
    deg = A2.sum(axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    D = np.diag(deg_inv_sqrt)
    return D @ A2 @ D

def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a @ b) / denom)

# 1) 그래프 구조(간선)만 추출
A_norm = np.load(A_norm_path)  # (N,N)
N = A_norm.shape[0]

# 구조만 필요하니 >0 으로 edge 추출 (self-loop 제외)
edge_i, edge_j = np.where(A_norm > 0)
mask = edge_i != edge_j
edge_i, edge_j = edge_i[mask], edge_j[mask]

# 중복 제거(undirected 가정)
edges = set()
for i, j in zip(edge_i, edge_j):
    if i < j:
        edges.add((i, j))
edges = list(edges)
print("num edges:", len(edges))

# 2) train 시계열로 node별 series 만들기 (여기선 간단히 전체 S를 씀; 엄밀히는 train split만 쓰는 게 좋음)
Y = np.load(Y_path)  # (S,N,T_out,2)
traffic = Y[..., 0].reshape(Y.shape[0], Y.shape[1], -1)[:, :, 0]  # (S, N) (T_out=1 가정)
speed   = Y[..., 1].reshape(Y.shape[0], Y.shape[1], -1)[:, :, 0]  # (S, N)

# (선택) 전역 패턴 제거(잔차)
traffic = traffic - traffic.mean(axis=1, keepdims=True)
speed   = speed   - speed.mean(axis=1, keepdims=True)

# 3) 간선별 corr 가중치 생성
A_w = np.zeros((N, N), dtype=np.float32)

for (i, j) in edges:
    c_tr = corr_1d(traffic[:, i], traffic[:, j])
    c_sp = corr_1d(speed[:, i],   speed[:, j])

    # 음수는 0으로 컷 (GCN에서 음수 가중치 섞이면 불안정할 때 많음)
    w = max(0.0, 0.5 * (c_tr + c_sp))
    A_w[i, j] = w
    A_w[j, i] = w

A_corr_norm = sym_norm(A_w)
np.save(out_path, A_corr_norm)
print("saved:", out_path, A_corr_norm.shape)
