import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from audio_processing import record_audio, load_and_preprocess_audio
import threading
import numpy as np

# Function to update the GUI with the predicted digit
def on_record_button_click(model, mfcc_mean, mfcc_std, result_label, plot_ax, root, stop_event):
    stop_event.clear()
    recording_thread = threading.Thread(target=record_audio_and_update_plot,
                                        args=(plot_ax, root, stop_event, model, mfcc_mean, mfcc_std, result_label))
    recording_thread.start()

# Function to stop recording and predict
def record_audio_and_update_plot(plot_ax, root, stop_event, model, mfcc_mean, mfcc_std, result_label):
    record_audio(filename="recorded_audio.wav", duration=2, plot_ax=plot_ax, root=root)
    audio, mfcc = load_and_preprocess_audio("recorded_audio.wav")
    processed_audio = (mfcc - mfcc_mean) / mfcc_std
    processed_audio = np.expand_dims(processed_audio, axis=0)
    prediction = model.predict(processed_audio)
    predicted_digit = np.argmax(prediction)
    result_label.config(text=f"Predicted digit: {predicted_digit}")

# Function to handle file upload
def on_upload_button_click(model, mfcc_mean, mfcc_std, result_label):
    file_path = filedialog.askopenfilename(title="Select an Audio File", filetypes=[("WAV files", "*.wav")])
    if file_path:
        audio, mfcc = load_and_preprocess_audio(file_path)
        processed_audio = (mfcc - mfcc_mean) / mfcc_std
        processed_audio = np.expand_dims(processed_audio, axis=0)
        prediction = model.predict(processed_audio)
        predicted_digit = np.argmax(prediction)
        result_label.config(text=f"Predicted digit: {predicted_digit}")
    else:
        messagebox.showerror("Error", "No file selected!")

# Function to initialize the GUI
def initialize_gui(model, mfcc_mean, mfcc_std):
    root = tk.Tk()
    root.title("Digit Prediction from Audio")
    root.geometry("600x500")

    instruction_label = tk.Label(root, text="Click to Record or Upload Your Voice", font=("Arial", 16), bg="#f0f0f0")
    instruction_label.pack(pady=20)

    result_label = tk.Label(root, text="Predicted digit: ", font=("Arial", 18), bg="#f0f0f0")
    result_label.pack(pady=20)

    stop_event = threading.Event()

    record_button = tk.Button(root, text="Record", font=("Arial", 14),
                               command=lambda: on_record_button_click(model, mfcc_mean, mfcc_std, result_label, plot_ax, root, stop_event),
                               bg="#4CAF50", fg="white")
    record_button.pack(pady=10)

    upload_button = tk.Button(root, text="Upload File", font=("Arial", 14),
                               command=lambda: on_upload_button_click(model, mfcc_mean, mfcc_std, result_label),
                               bg="#2196F3", fg="white")
    upload_button.pack(pady=10)

    fig, plot_ax = plt.subplots(figsize=(5, 3))
    plot_ax.set_title("Real-time Waveform")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(pady=20)
    canvas.draw()

    root.mainloop()
