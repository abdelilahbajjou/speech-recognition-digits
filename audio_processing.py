import numpy as np
import librosa
import wave
import pyaudio

# Function to load and preprocess audio files using MFCC
def load_and_preprocess_audio(file_path, sr=16000, dt=1.0):
    audio, _ = librosa.load(file_path, sr=sr)
    target_length = int(sr * dt)
    if len(audio) < target_length:
        pad_width = target_length - len(audio)
        audio = np.pad(audio, (0, pad_width), 'constant')
    else:
        audio = audio[:target_length]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, win_length=400, hop_length=160)
    return audio, mfcc.T

# Function to record audio using PyAudio and plot waveform
def record_audio(filename="recorded_audio.wav", duration=3, sr=16000, plot_ax=None, root=None):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=1024)
    print("Recording...")
    frames = []
    audio_data = np.array([])

    for _ in range(0, int(sr / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
        audio_chunk = np.frombuffer(data, dtype=np.int16)
        audio_data = np.concatenate((audio_data, audio_chunk))

        if plot_ax is not None:
            plot_ax.clear()
            plot_ax.plot(audio_data, color='blue')
            plot_ax.set_title("Real-time Waveform")
            plot_ax.set_xlabel("Time (samples)")
            plot_ax.set_ylabel("Amplitude")
            plot_ax.figure.canvas.draw()
        if root:
            root.update()

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sr)
        wf.writeframes(b''.join(frames))

    return filename
