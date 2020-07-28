"""Softmax."""

# scores = [3.0, 1.0, 0.2, 0.5, 4.2, -1.3]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=0)

# print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-3.0, 7.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 1.3 * np.ones_like(x), x/1.1 + 1])

print(scores)

plt.grid()
plt.xlabel("x")
plt.ylabel("Pravděpodobnost = Softmax(x)")
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
