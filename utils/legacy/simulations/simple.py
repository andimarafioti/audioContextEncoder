import numpy as np
import matplotlib.pyplot as plt

fs = 16000
time = np.arange(0, 0.005, 1/fs)
plt.plot(np.sin(2 * np.pi *440 * time , dtype=np.float32) + np.random.normal(0, 0.1, len(time)))
plt.show()