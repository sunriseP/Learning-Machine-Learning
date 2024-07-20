import numpy as np

# 3(b)
p = np.array([0.05, 0.20, 0.40, 0.20, 0.15]).T
q = np.array([0.10, 0.60, 0.05, 0.15, 0.10]).T

def d_kl(p: np.ndarray, q: np.ndarray):
  return sum(p*np.log(p/q))

print(f"D_KL(p||q): {d_kl(p, q)}")
print(f"D_KL(q||p): {d_kl(q, p)}")

