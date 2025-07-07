import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define base directory
DATA_DIR = 'data'
LABELS = ['TB', 'Pneumonia', 'Normal']

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=5, offset=0.5)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data():
    features, labels = [], []

    for label in LABELS:
        folder = os.path.join(DATA_DIR, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)
                mfcc = extract_features(file_path)
                if mfcc is not None:
                    features.append(mfcc)
                    labels.append(label)
    
    return np.array(features), np.array(labels)

def train_and_save_model():
    print("[INFO] Loading and extracting features...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("[INFO] Saving model to 'cough_classifier.pkl'...")
    joblib.dump(clf, "cough_classifier.pkl")

if __name__ == "__main__":
    train_and_save_model()
