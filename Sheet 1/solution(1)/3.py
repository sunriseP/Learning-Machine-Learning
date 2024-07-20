import numpy as np

def func_L1norm(v1, v2):
    return np.sum(np.abs(v1-v2))

def func_L2norm(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def func_cosine_sim(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return np.dot(v1.T, v2) / (norm_v1 * norm_v2)

v1 = np.array([0.2, 0.1, 0.4, -0.4])
v2 = np.array([-0.1, -0.1, 0.8, 0.5])
print(f'res: {func_cosine_sim(v1, v2)}')