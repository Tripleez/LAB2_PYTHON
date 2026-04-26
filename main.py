import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, ifft
import time

start_time = time.time()

variant = 29
filename = f"{variant}.wav"

print(f"=== Вариант {variant} ===")
print(f"Частота дискретизации по заданию: 16000 Гц")
print(f"Тип спектрального анализа: Кепстр (ОДПФ[ln(|ДПФ|²)])")

try:
    sample_rate, audio_data = wavfile.read(filename)
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1).astype(np.int16)
except Exception as e:
    print(f"Ошибка: {e}")
    exit(1)

print(f"Файл загружен: {sample_rate} Гц, {len(audio_data)} отсчетов")

# Простая проверка ввода
while True:
    user_input = input(f"Введите количество отсчетов (1-{len(audio_data)}): ")
    if user_input.isdigit():
        n_samples = int(user_input)
        if 1 <= n_samples <= len(audio_data):
            break
    print("Ошибка ввода")

samples_to_plot = audio_data[:n_samples]
time_axis = np.arange(n_samples) / sample_rate

plt.figure(figsize=(14, 10))

# График 1
plt.subplot(2, 2, 1)
plt.plot(time_axis, samples_to_plot, 'b--', linewidth=0.8)
plt.title(f'Дискретные отсчеты (первые {n_samples})')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)

# График 2
plt.subplot(2, 2, 2)
full_time = np.arange(len(audio_data)) / sample_rate
plt.plot(full_time, audio_data, 'g-', linewidth=0.5)
plt.title('Осциллограмма')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid(True)

# График 3: Кепстр
N = len(audio_data)
fft_result = fft(audio_data)
magnitude_sq = np.maximum(np.abs(fft_result) ** 2, 1e-10)
cepstrum = np.real(ifft(np.log(magnitude_sq)))
freqs = np.fft.fftfreq(N, 1/sample_rate)[:N//2]
cepstrum_pos = cepstrum[:N//2]

plt.subplot(2, 2, 3)
plt.plot(freqs, cepstrum_pos, 'r-', linewidth=0.7)
plt.title('Кепстр сигнала')
plt.xlabel('Частота, Гц')
plt.ylabel('Амплитуда кепстра')
plt.grid(True)
plt.xlim(0, sample_rate/2)

# График 4: Гистограмма
plt.subplot(2, 2, 4)
n_bins = min(int(np.ceil(np.log2(len(audio_data)) + 1)), 100)
plt.hist(audio_data, bins=n_bins, color='purple', alpha=0.7, edgecolor='black')
plt.title('Гистограмма амплитуд')
plt.xlabel('Амплитуда')
plt.ylabel('Частота, Гц')
plt.grid(True)

plt.tight_layout()
plt.suptitle(f'Анализ сигнала (вариант {variant})', fontsize=14)
plt.show()

#Время выполнения
print(f"\nВремя выполнения: {time.time() - start_time:.4f} секунд")
