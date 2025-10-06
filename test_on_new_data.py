# test_on_new_data.py
import os
import numpy as np
import librosa
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shutil
import datetime

def extract_mfcc(audio_path, sr=22050, n_mfcc=8):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        y_clean, _ = librosa.effects.trim(y, top_db=20)
        if len(y_clean) == 0:
            y_clean = y
        mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
        return mfccs.T
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def calculate_confidence(scores):
    """Safely calculate confidence scores from log-likelihoods."""
    if not scores or len(scores) == 0:
        return 0.5
    
    if len(scores) == 1:
        return 0.7
    
    log_likelihoods = list(scores.values())
    max_log_likelihood = max(log_likelihoods)
    
    if max_log_likelihood < -1000:
        return 0.5
    
    shifted_log_likelihoods = [ll - max_log_likelihood for ll in log_likelihoods]
    
    try:
        probabilities = [np.exp(ll) for ll in shifted_log_likelihoods]
        total = sum(probabilities)
        
        if total == 0 or np.isnan(total):
            return 0.5
            
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx] / total
        return max(0.1, min(0.99, confidence))
        
    except (FloatingPointError, ValueError):
        return 0.5

def predict_audio_class(audio_path, hmm_male, hmm_female):
    """Predict gender from audio using HMMs."""
    features = extract_mfcc(audio_path)
    if features is None or len(features) < 5:
        return 'unknown', 0.5
    
    try:
        scores = {}
        if hmm_male:
            scores['male'] = hmm_male.score(features)
        if hmm_female:
            scores['female'] = hmm_female.score(features)
        
        if not scores:
            return 'unknown', 0.5
        
        # Determine prediction
        best_label = max(scores.items(), key=lambda x: x[1])[0]
        confidence = calculate_confidence(scores)
        
        return best_label, confidence
        
    except Exception as e:
        print(f"Error in audio prediction: {e}")
        return 'unknown', 0.5

def manual_correction(audio_path, predicted_label, true_label, weight):
    """Handle wrong predictions and collect corrected data."""
    print(f"\n❌ Prediction Error Detected!")
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Predicted: {predicted_label}, Actual: {true_label}")
    
    # Ask user if they want to correct and collect this sample
    response = input("Add this sample to training data? (y/n): ").lower()
    
    if response == 'y':
        # Create corrected directory structure if it doesn't exist
        corrected_audio_dir = os.path.join("audio/corrected_data", true_label)
        os.makedirs(corrected_audio_dir, exist_ok=True)
        
        # Copy audio file to appropriate class folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"corrected_{true_label}_{timestamp}.wav"
        destination_path = os.path.join(corrected_audio_dir, new_filename)
        
        shutil.copy2(audio_path, destination_path)
        
        # Also add to weight CSV if available
        corrected_weight_file = "data/corrected_weights.csv"
        os.makedirs('data', exist_ok=True)
        
        with open(corrected_weight_file, 'a') as f:
            f.write(f"{weight},{true_label}\n")
        
        print(f"✓ Added to training data: {destination_path}")
        return True
    
    return False

def analyze_error(audio_path, true_label, predicted_label):
    """Analyze why a prediction might be wrong."""
    try:
        y, sr = librosa.load(audio_path)
        
        duration = len(y) / sr
        max_amplitude = np.max(np.abs(y))
        rms_energy = np.sqrt(np.mean(y**2))
        
        print(f"\n🔍 Error Analysis for {os.path.basename(audio_path)}:")
        print(f"True: {true_label}, Predicted: {predicted_label}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Max Amplitude: {max_amplitude:.3f}")
        print(f"RMS Energy: {rms_energy:.3f}")
        
        if duration < 0.5:
            print("⚠️  Audio might be too short for reliable analysis")
        if max_amplitude < 0.1:
            print("⚠️  Low audio volume - might be noisy recording")
        if rms_energy < 0.01:
            print("⚠️  Very low energy - possibly silent or noisy")
            
        return {
            'duration': duration,
            'max_amplitude': max_amplitude,
            'rms_energy': rms_energy
        }
    except:
        print("Could not analyze audio file")
        return {}

def test_on_new_data():
    # Load trained models
    try:
        rf_model = joblib.load('models/lovebird_rf_model.pkl')
        hmm_male = joblib.load('models/hmm_male.pkl')
        hmm_female = joblib.load('models/hmm_female.pkl')
        print("Models loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please train the models first using train_rf.py and train_hmm.py")
        return

    # Configuration
    test_data_path = "audio/new_test_data"
    test_metadata_file = os.path.join(test_data_path, "test_metadata.csv")

    if not os.path.exists(test_metadata_file):
        print(f"Error: Test metadata file {test_metadata_file} not found!")
        return

    # Load test metadata
    test_df = pd.read_csv(test_metadata_file)
    results = []

    print(f"\nTesting on {len(test_df)} samples from {test_data_path}...")
    print("-" * 60)

    for index, row in test_df.iterrows():
        audio_file = os.path.join(test_data_path, row['filename'])
        true_sex = row['true_sex']
        weight = row['weight']
        
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file {audio_file} not found! Skipping...")
            continue
        
        # Get individual predictions
        try:
            rf_pred = 'male' if rf_model.predict([[weight]])[0] == 1 else 'female'
            rf_proba = rf_model.predict_proba([[weight]])[0]
            rf_confidence = rf_proba[1] if rf_pred == 'male' else rf_proba[0]
        except:
            rf_pred = 'unknown'
            rf_confidence = 0.5
        
        hmm_pred, hmm_confidence = predict_audio_class(audio_file, hmm_male, hmm_female)
        
        # Fuse predictions (simple weighted average)
        if rf_pred != 'unknown' and hmm_pred != 'unknown':
            alpha, beta = 0.5, 0.5  # Equal weights
            fused_male_prob = (alpha * (1 if rf_pred == 'male' else 0) + 
                              beta * (hmm_confidence if hmm_pred == 'male' else (1 - hmm_confidence)))
            fused_pred = 'male' if fused_male_prob > 0.5 else 'female'
            fused_confidence = fused_male_prob if fused_pred == 'male' else (1 - fused_male_prob)
        else:
            # Fallback to whichever prediction is available
            if rf_pred != 'unknown':
                fused_pred, fused_confidence = rf_pred, rf_confidence
            elif hmm_pred != 'unknown':
                fused_pred, fused_confidence = hmm_pred, hmm_confidence
            else:
                fused_pred, fused_confidence = 'unknown', 0.5
        
        results.append({
            'file': row['filename'],
            'true_sex': true_sex,
            'weight': weight,
            'rf_pred': rf_pred,
            'rf_confidence': rf_confidence,
            'hmm_pred': hmm_pred,
            'hmm_confidence': hmm_confidence,
            'fused_pred': fused_pred,
            'fused_confidence': fused_confidence,
            'correct': fused_pred == true_sex if fused_pred != 'unknown' else False
        })
        
        # Print individual result
        status = "✓" if fused_pred == true_sex else "✗" if fused_pred != 'unknown' else "?"
        print(f"{status} {row['filename']}: True={true_sex}, "
              f"RF={rf_pred}({rf_confidence:.2f}), HMM={hmm_pred}({hmm_confidence:.2f}), "
              f"Fused={fused_pred}({fused_confidence:.2f})")
        
        # Handle errors
        if fused_pred != true_sex and fused_pred != 'unknown':
            analyze_error(audio_file, true_sex, fused_pred)
            manual_correction(audio_file, fused_pred, true_sex, weight)

    # Calculate overall metrics
    valid_results = [r for r in results if r['fused_pred'] != 'unknown']
    
    if valid_results:
        true_labels = [r['true_sex'] for r in valid_results]
        fused_preds = [r['fused_pred'] for r in valid_results]
        rf_preds = [r['rf_pred'] for r in valid_results if r['rf_pred'] != 'unknown']
        hmm_preds = [r['hmm_pred'] for r in valid_results if r['hmm_pred'] != 'unknown']

        print("\n" + "="*60)
        print("FINAL EVALUATION ON NEW DATA")
        print("="*60)

        if rf_preds:
            rf_accuracy = accuracy_score([r['true_sex'] for r in valid_results if r['rf_pred'] != 'unknown'], 
                                       [r for r in rf_preds])
            print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({len(rf_preds)} samples)")
        
        if hmm_preds:
            hmm_accuracy = accuracy_score([r['true_sex'] for r in valid_results if r['hmm_pred'] != 'unknown'], 
                                        [r for r in hmm_preds])
            print(f"HMM Accuracy: {hmm_accuracy:.4f} ({len(hmm_preds)} samples)")
        
        fused_accuracy = accuracy_score(true_labels, fused_preds)
        print(f"Fused Model Accuracy: {fused_accuracy:.4f} ({len(valid_results)} samples)")

        # Detailed reports
        print(f"\nFused Model Classification Report:")
        print(classification_report(true_labels, fused_preds, target_names=['female', 'male']))

        print(f"Fused Model Confusion Matrix:")
        print(confusion_matrix(true_labels, fused_preds))

    # Save detailed results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('new_data_test_results.csv', index=False)
    print(f"\nDetailed results saved to 'new_data_test_results.csv'")

if __name__ == "__main__":
    test_on_new_data()