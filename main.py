from model import main as train_model
from gui import initialize_gui

if __name__ == "__main__":
    data_folder = r"..\Dataset"
    model, mfcc_mean, mfcc_std = train_model(data_folder)  # Train and get MFCC stats
    initialize_gui(model, mfcc_mean, mfcc_std)  # Start GUI
