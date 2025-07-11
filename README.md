# Speech Recognition for Digit Classification 🎤🔢

A deep learning application that recognizes spoken digits (0-9) using LSTM neural networks and MFCC feature extraction.

## 🎯 Features

- **Real-time audio recording** and digit prediction
- **File upload** functionality for audio files
- **LSTM-based neural network** for sequence classification
- **MFCC feature extraction** for audio preprocessing
- **Interactive GUI** with real-time waveform visualization
- **Bidirectional LSTM** for improved accuracy

## 🖥️ Demo

📹 **[Watch Demo Video](https://drive.google.com/drive/u/1/folders/1Ewm1Tq8ZM3AfP0V6E_4fexOgv6ekF4d8)**

## 📊 Performance

- **Accuracy**: 95%
- **Model Architecture**: Bidirectional LSTM with 64 units
- **Features**: 13 MFCC coefficients

## 🚀 Quick Start

### Prerequisites

- Python 3.8-3.12 (Python 3.10 recommended)
- Microphone for recording
- Audio files in WAV format (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abdelilahbajjou/speech-recognition-digits.git
   cd speech-recognition-digits
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset** (see Dataset section below)

5. **Run the application**
   ```bash
   python main.py
   ```

## 📁 Dataset

### Option 1: Download Full Dataset
1. Download the dataset from: [Google Drive Link](https://drive.google.com/file/d/1cqtSSpbizNVbypmqs80ecyXWy89jhc28/view?usp=sharing)
2. Extract the zip file to the project directory
3. Ensure the structure matches:
   ```
   Dataset/
   ├── d0/  # Digit 0 audio files
   ├── d1/  # Digit 1 audio files
   ├── ...
   └── d9/  # Digit 9 audio files
   ```

### Option 2: Use Sample Dataset
- The repository includes sample audio files in `dataset/sample_audio/`
- You can test the application with these samples
- Add your own recordings to improve the model

### Dataset Structure
- **10 classes**: Digits 0-9
- **File format**: WAV files (16kHz sampling rate)
- **Duration**: 1-2 seconds per audio file
- **Total files**: [NUMBER] audio samples

## 🛠️ Usage

### Training the Model
The model trains automatically when you run `main.py`. The training process:
1. Loads audio files from the dataset
2. Extracts MFCC features
3. Trains the LSTM model
4. Saves normalization parameters

### Using the GUI
1. **Record Audio**: Click "Record" and speak a digit (0-9)
2. **Upload File**: Click "Upload File" to test existing audio files
3. **View Results**: The predicted digit appears on screen
4. **Real-time Visualization**: Watch the audio waveform as you record

## 🏗️ Architecture

### Model Architecture
```
Input (None, 13) -> Bidirectional LSTM (64) -> Flatten -> Dense (64) -> Dropout (0.5) -> Dense (10)
```

### Key Components
- **MFCC Extraction**: 13 coefficients with librosa
- **Normalization**: Z-score normalization using training statistics
- **LSTM**: Bidirectional LSTM for sequence modeling
- **Regularization**: Dropout layer to prevent overfitting

## 📝 Files Description

- `main.py`: Main application entry point
- `model.py`: Neural network model definition and training
- `gui.py`: Tkinter-based graphical user interface
- `audio_processing.py`: Audio preprocessing and MFCC extraction
- `requirements.txt`: Python dependencies
- `docs/academic_report.pdf`: Detailed academic report (French)

## 🔧 Troubleshooting

### Common Issues

1. **TensorFlow Installation Error**
   - Ensure you're using Python 3.8-3.12
   - Use virtual environment
   - Try: `pip install tensorflow==2.15.0`

2. **PyAudio Installation Error**
   ```bash
   # Windows
   pip install pipwin
   pipwin install pyaudio
   ```

3. **Audio Recording Issues**
   - Check microphone permissions
   - Ensure microphone is not used by other applications
   - Try different audio devices

4. **Dataset Not Found**
   - Verify dataset folder structure
   - Check file paths in `main.py`
   - Ensure audio files are in WAV format

## 📚 Dependencies

- `tensorflow`: Deep learning framework
- `librosa`: Audio processing library
- `scikit-learn`: Machine learning utilities
- `matplotlib`: Plotting and visualization
- `numpy`: Numerical computing
- `pyaudio`: Audio recording interface
- `tkinter`: GUI framework (usually included with Python)

## 🎓 Academic Context

This project was developed as part of a speech recognition course at Faculty of Sciences Dhar El Mahraz –FSDM-. The complete academic report (in French) is available in `docs/academic_report.pdf`.

### Methodology
1. **Data Collection**: Audio recordings of spoken digits
2. **Feature Extraction**: MFCC coefficients for audio representation
3. **Model Training**: LSTM neural network with bidirectional processing
4. **Evaluation**: Cross-validation and performance metrics
5. **Application**: Real-time GUI for practical usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Faculty of Sciences Dhar El Mahraz –FSDM- for the academic framework
- Course instructors for guidance
- Open source libraries that made this project possible

## 📞 Contact

- **Author**: Abdelilah Bajjou
- **Email**: abdelilahbajjou@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/abdelilah-bajjou-454222246/
- **Academic Report**: Available in `docs/academic_report.pdf`

---

⭐ **Star this repository if you found it helpful!**
