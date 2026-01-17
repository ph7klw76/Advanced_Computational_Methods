
## Resonance Frequency

<img width="563" height="1000" alt="image" src="https://github.com/user-attachments/assets/a83981e7-1acc-47db-a3a0-b478e4be742d" /> 
<img width="750" height="1000" alt="image" src="https://github.com/user-attachments/assets/15e99fe4-8dd0-4836-bbce-694df9d778e3" />


How to teach lab during the conditional restricted movement order? I created a python template that can make use of the student's microphone and computer to study the resonant frequency and damping constant. Pictures are sent by the students doing lab at home.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.optimize import curve_fit


import sounddevice as sd

from skimage import util  


def record_from_mic(filename="mic_recording.wav", duration=5.0, rate=44100, channels=1):
    print(f"Recording {duration:.1f}s from microphone at {rate} Hz ...")
    audio = sd.rec(int(duration * rate), samplerate=rate, channels=channels, dtype=np.float32)
    sd.wait()
    print("Done recording.")

    # Convert float32 [-1,1] to int16 for WAV saving
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    wavfile.write(filename, rate, audio_int16)
    print(f"Saved: {filename}")
    return filename


def drawgraph(func, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    popt, pcov = curve_fit(func, x, y, maxfev=20000)
    predicted = func(x, *popt)
    r = np.corrcoef(y, predicted)
    r2 = r[0, 1] ** 2

    print("coefficient of determination (R^2):", r2)
    for idx, p in enumerate(popt, start=1):
        print(p, f"  parameter {idx}")
    return popt, r2


def analyze_wav(wav_path):
    rate, audio = wavfile.read(wav_path)

    # Ensure float for processing
    audio = audio.astype(np.float32)

    # If stereo, convert to mono
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    N = audio.shape[0]
    L = N / rate
    print(f"Audio length: {L:.2f} seconds")

    f, ax = plt.subplots(2, 2, figsize=(12, 8))

    # --- Waveform ---
    t = np.arange(N) / rate
    ax[0][0].plot(t, audio)
    ax[0][0].set_xlabel("Time (s)")
    ax[0][0].set_ylabel("Amplitude (a.u.)")
    ax[0][0].set_title("Recorded waveform")

    # --- Spectrum (windowed slices like your approach) ---
    NN = 1
    M = 1024 * NN
    step = 100

    if N < M:
        raise ValueError(f"Recording too short for window size M={M}. Record longer or reduce M.")

    slices = util.view_as_windows(audio, window_shape=(M,), step=step)
    print(f"Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}")

    win = np.hanning(M)
    slices = slices * win
    slices = slices.T

    # Use rFFT for real signals; gives correct half-spectrum
    spectrum = np.fft.rfft(slices, axis=0)
    S = np.abs(spectrum)
    Z = np.sum(S, axis=1)

    freqs = np.fft.rfftfreq(M, d=1.0 / rate)

    ax[0][1].plot(freqs, Z)
    ax[0][1].set_xscale("log")
    ax[0][1].set_yscale("log")
    ax[0][1].set_xlabel("Frequency (Hz)")
    ax[0][1].set_ylabel("Amplitude (a.u.)")
    ax[0][1].set_title("Summed magnitude spectrum")

    # --- Decay fit region (you can edit these times) ---
    def datapoint(tsec):
        return int(tsec * rate)

    # Pick a region inside the recording
    # (Change these to match where your sound "rings down")
    t1 = min(0.5, L * 0.2)
    t2 = min(2.5, L * 0.8)

    x1 = datapoint(t1)
    x2 = datapoint(t2)

    amplitude = []
    number = 200

    # Robust peak envelope sampling
    x0 = None
    y0 = None

    for i in range((x2 - x1) // number):
        seg = audio[x1 + i * number: x1 + (i + 1) * number]
        y = np.max(np.abs(seg))  # use abs peak for mic signals

        if i == 0:
            x0 = x1 + i * number
            y0 = y if y != 0 else 1.0
            x = 0.0
            y_norm = 1.0
        else:
            # time offset (seconds)
            x = ((x1 + i * number) - x0) / rate
            y_norm = y / y0

        amplitude.append([x, y_norm])

    x = [pt[0] for pt in amplitude]
    y = [pt[1] for pt in amplitude]

    def func(t, k):
        return np.exp(-k * t)

    if len(x) > 3:
        popt, r2 = drawgraph(func, x, y)
        newx = np.linspace(0, max(x), 200)

        ax[1][0].scatter(x, y, label="envelope points")
        ax[1][0].plot(newx, func(newx, *popt), color="red", label="exp fit")
        ax[1][0].set_ylim(0, max(y) * 1.1)
        ax[1][0].set_xlabel("Time (s)")
        ax[1][0].set_ylabel("Normalized amplitude (a.u.)")
        ax[1][0].set_title(f"Decay fit: k={popt[0]:.4g}, R^2={r2:.3f}")
        ax[1][0].legend()
    else:
        ax[1][0].text(0.1, 0.5, "Not enough points for decay fit.\nRecord longer or adjust t1/t2.",
                      transform=ax[1][0].transAxes)

    # --- Zoomed waveform region (like your last plot) ---
    ax[1][1].plot(t, audio)
    ax[1][1].set_xlabel("Time (s)")
    ax[1][1].set_ylabel("Amplitude (a.u.)")
    ax[1][1].set_xlim(t1, t2)
    ax[1][1].set_title("Waveform (zoomed region)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Record first, then analyze
    wav_path = record_from_mic(filename="mic_recording.wav", duration=5.0, rate=44100, channels=1)
    analyze_wav(wav_path)
```
