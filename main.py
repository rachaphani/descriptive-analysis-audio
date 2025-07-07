import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import os

# Load or train a dummy classifier for TB, Pneumonia
def train_dummy_model():
    # Simulated training with dummy data
    X = np.random.rand(100, 40)  # 100 samples, 40 MFCCs
    y = np.random.choice(['TB', 'Pneumonia', 'Normal'], size=100)
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Step 1: Extract MFCC Features
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Step 2: Classify Audio (TB / Pneumonia / Normal)
def classify_disease(model, features):
    prediction = model.predict([features])[0]
    return prediction

# Step 3: Use HuggingFace to Generate a Descriptive Report
def generate_descriptive_analysis(disease_label):
    summarizer = pipeline("text2text-generation", model="google/flan-t5-xl")
    prompt = f"""Generate a medical explanation for a diagnosis of {disease_label} based on cough audio analysis. 
    Include causes, symptoms, and possible treatments in layman's terms."""
    
    result = summarizer(prompt, max_length=300, do_sample=True)[0]['generated_text']
    return result

# Step 4: Main Pipeline
def analyze_audio(audio_path):
    print("🔍 Extracting features from audio...")
    features = extract_audio_features(audio_path)

    print("🧠 Classifying disease...")
    model = train_dummy_model()
    label = classify_disease(model, features)

    print(f"✅ Prediction: {label}")

    print("📄 Generating descriptive analysis using GenAI...")
    report = generate_descriptive_analysis(label)

    print("\n📋 Final Report:")
    print(report)

# Example usage:
if __name__ == "__main__":
    audio_path = "normal-4.wav" 
    if os.path.exists(audio_path):
        analyze_audio(audio_path)
    else:
        print(f"Audio file '{audio_path}' not found. Please provide a .wav file.")
