import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from PyQt5 import QtCore, QtGui, QtWidgets
import speech_recognition as sr
import pyaudio
import sounddevice as sd
import pyttsx3
from scipy.io.wavfile import write
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import time


# Specify the path to the root folder containing 24 class folders
data_root_folder = "Data"

# Function to extract MFCCs from an audio file
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCCs
    return mfccs.flatten()  # Flatten the MFCCs to use as features

# Extract features and labels from the folders
X = []
y = []

for class_folder in os.listdir(data_root_folder):
    class_folder_path = os.path.join(data_root_folder, class_folder)
    
    if os.path.isdir(class_folder_path):
        for file_name in os.listdir(class_folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(class_folder_path, file_name)
                mfcc_features = extract_mfcc(file_path)
                
                X.append(mfcc_features)
                y.append(class_folder)  # Assuming the folder name is the label

# Convert class labels to numerical values
label_mapping = {class_label: idx for idx, class_label in enumerate(set(y))}
y = np.array([label_mapping[class_label] for class_label in y])
print(label_mapping)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions on the test set
predictions = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Additional metrics and details
print(classification_report(y_test, predictions))




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(934, 752)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.spectrogram_widget = QtWidgets.QWidget(self.centralwidget)
        self.spectrogram_widget.setGeometry(QtCore.QRect(20, 150, 391, 321))
        self.spectrogram_widget.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.spectrogram_widget.setObjectName("spectrogram_widget")

       # Create a Figure and an Axes for the spectrogram
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        
        # Set layout for the spectrogram_widget
        layout = QtWidgets.QVBoxLayout(self.spectrogram_widget)
        layout.addWidget(self.canvas)


        self.progressBar_9 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_9.setGeometry(QtCore.QRect(570, 240, 351, 23))
        self.progressBar_9.setProperty("value", 24)
        self.progressBar_9.setObjectName("progressBar_9")
        self.progressBar_2_word = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2_word.setGeometry(QtCore.QRect(570, 330, 351, 23))
        self.progressBar_2_word.setProperty("value", 24)
        self.progressBar_2_word.setObjectName("progressBar_2_word")
        self.progressBar_3_word = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3_word.setGeometry(QtCore.QRect(570, 420, 351, 23))
        self.progressBar_3_word.setProperty("value", 24)
        self.progressBar_3_word.setObjectName("progressBar_3_word")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(450, 240, 111, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 330, 101, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(450, 420, 101, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(40, 530, 55, 16))
        self.label_4.setObjectName("label_4")
        self.progressBar_8 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_8.setGeometry(QtCore.QRect(790, 620, 118, 23))
        self.progressBar_8.setProperty("value", 24)
        self.progressBar_8.setObjectName("progressBar_8")
        self.progressBar_5 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_5.setGeometry(QtCore.QRect(560, 530, 118, 23))
        self.progressBar_5.setProperty("value", 24)
        self.progressBar_5.setObjectName("progressBar_5")
        self.progressBar_6 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_6.setGeometry(QtCore.QRect(560, 620, 118, 23))
        self.progressBar_6.setProperty("value", 24)
        self.progressBar_6.setObjectName("progressBar_6")
        self.progressBar_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3.setGeometry(QtCore.QRect(340, 530, 118, 23))
        self.progressBar_3.setProperty("value", 24)
        self.progressBar_3.setObjectName("progressBar_3")
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setGeometry(QtCore.QRect(790, 530, 118, 23))
        self.progressBar_7.setProperty("value", 24)
        self.progressBar_7.setObjectName("progressBar_7")
        self.progressBar_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_4.setGeometry(QtCore.QRect(340, 620, 118, 23))
        self.progressBar_4.setProperty("value", 24)
        self.progressBar_4.setObjectName("progressBar_4")
        self.progressBar_1 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_1.setGeometry(QtCore.QRect(110, 530, 118, 23))
        self.progressBar_1.setProperty("value", 24)
        self.progressBar_1.setObjectName("progressBar_1")
        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setGeometry(QtCore.QRect(110, 620, 118, 23))
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 620, 55, 16))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(270, 530, 55, 16))
        self.label_6.setObjectName("label_6")
        self.securityvoicefingerprint_radio_button = QtWidgets.QRadioButton(self.centralwidget)
        self.securityvoicefingerprint_radio_button.setGeometry(QtCore.QRect(50, 120, 181, 20))
        self.securityvoicefingerprint_radio_button.setObjectName("securityvoicefingerprint_radio_button")
        self.securityvoicecode_radio_button = QtWidgets.QRadioButton(self.centralwidget)
        self.securityvoicecode_radio_button.setGeometry(QtCore.QRect(50, 90, 171, 20))
        self.securityvoicecode_radio_button.setObjectName("securityvoicecode_radio_button")
        self.display_text_box = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.display_text_box.setGeometry(QtCore.QRect(590, 100, 301, 41))
        self.display_text_box.setObjectName("display_text_box")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(270, 620, 55, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(490, 530, 55, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(490, 620, 55, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(720, 530, 55, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(720, 620, 55, 16))
        self.label_11.setObjectName("label_11")
        self.record_button = QtWidgets.QPushButton(self.centralwidget)
        self.record_button.setGeometry(QtCore.QRect(250, 100, 121, 41))
        self.record_button.setObjectName("record_button")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(500, 100, 81, 31))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(410, 20, 171, 51))
        self.label_13.setObjectName("label_13")
        self.access_text_button = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.access_text_button.setGeometry(QtCore.QRect(590, 160, 301, 41))
        self.access_text_button.setObjectName("access_text_button")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(520, 170, 81, 31))
        self.label_14.setObjectName("label_14")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 934, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Open Middle Door"))
        self.label_2.setText(_translate("MainWindow", "Unlock the gate "))
        self.label_3.setText(_translate("MainWindow", "Grant me access"))
        self.label_4.setText(_translate("MainWindow", "Person1"))
        self.label_5.setText(_translate("MainWindow", "Person 2"))
        self.label_6.setText(_translate("MainWindow", "Person 3"))
        self.securityvoicefingerprint_radio_button.setText(_translate("MainWindow", "Security voice fingerprint "))
        self.securityvoicecode_radio_button.setText(_translate("MainWindow", "Security voice code"))
        self.label_7.setText(_translate("MainWindow", "Person 4"))
        self.label_8.setText(_translate("MainWindow", "Person 5"))
        self.label_9.setText(_translate("MainWindow", "Person 6"))
        self.label_10.setText(_translate("MainWindow", "Person 7"))
        self.label_11.setText(_translate("MainWindow", "Person 8"))
        self.record_button.setText(_translate("MainWindow", "Record"))
        self.label_12.setText(_translate("MainWindow", "Display text"))
        self.label_13.setText(_translate("MainWindow", "Security Voice Code Access"))
        self.label_14.setText(_translate("MainWindow", "Access"))

        self.record_button.clicked.connect(self.record_and_predict)

    def record_and_predict(self):
        # Record audio for 3 seconds
        fs = 44100  # Sample rate
        duration = 3  # Duration in seconds
        recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save the recorded audio to a .wav file
        timestamp = int(time.time())
        filename = f"recorded_{timestamp}.wav"
        write(filename, fs, recording)

        # Plot the spectrogram of the recorded audio
        self.plot_spectrogram(filename)

        # Extract MFCC features from the recorded audio
        mfcc_features = extract_mfcc(filename)

        # Use the trained RandomForestClassifier for prediction
        prediction_proba = rf_model.predict_proba([mfcc_features])[0]
        print("Prediction Probabilities:", prediction_proba)

        # Update progress bars or take any action based on the prediction probabilities
        # For example, update progress bars with the class probabilities
        for i, prob in enumerate(prediction_proba):
            getattr(self, f"progressBar_{i+1}").setValue(int((prob * 100)+20))

    
    def plot_spectrogram(self, filename):
        # Load the audio file
        fs, audio = wavfile.read(filename)

        # Plot the spectrogram
        self.ax.clear()
        self.ax.specgram(audio, Fs=fs, cmap='viridis', aspect='auto')  # Remove [:, 0]
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        self.fig.tight_layout()
        self.canvas.draw()
    def clear_spectrogram_widget(self):
        self.ax.clear()
        self.canvas.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())