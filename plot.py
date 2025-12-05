import numpy as np
import matplotlib.pyplot as plt

# 取一个故障样本
sample = train_data[0, :, 0]  # 假设train_data是(样本数, 500, 1)的数组
fft_result = np.fft.fft(sample)
frequencies = np.fft.fftfreq(len(sample))
amplitudes = np.abs(fft_result)

# 绘制前100个频率的振幅（排除直流分量）
plt.plot(frequencies[1:100], amplitudes[1:100])
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.title("FFT of CWRU Vibration Signal")
plt.show()