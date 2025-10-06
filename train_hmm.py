# train_hmm.py
import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import warnings

def extract_mfcc(audio_path, sr=22050, n_mfcc=8):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        # Remove silence
        y_clean, _ = librosa.effects.trim(y, top_db=20)
        if len(y_clean) == 0:
            y_clean = y  # Use original if trimming removed everything
            
        mfccs = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=256)
        return mfccs.T
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def train_hmm_for_class(audio_files, n_components=2):
    """Train an HMM on a set of audio files for one class."""
    features_list = []
    valid_files = []
    
    for file_path in audio_files:
        mfccs = extract_mfcc(file_path)
        if mfccs is not None and len(mfccs) > 5:  # Minimum 5 frames
            features_list.append(mfccs)
            valid_files.append(file_path)
    
    if not features_list:
        print("No valid features found for this class!")
        return None, []
    
    lengths = [len(f) for f in features_list]
    X_train = np.vstack(features_list)
    
    try:
        model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=50,
            random_state=42,
            tol=1e-3  # Added tolerance to help convergence
        )
        model.fit(X_train, lengths=lengths)
        return model, valid_files
    except Exception as e:
        print(f"Error training HMM: {e}")
        return None, valid_files

def calculate_confidence(scores):
    """Safely calculate confidence scores from log-likelihoods."""
    if not scores or len(scores) == 0:
        return 0.5
    
    # Handle case where only one model is available
    if len(scores) == 1:
        return 0.7  # Default confidence for single model
    
    # Get the log-likelihood values
    log_likelihoods = list(scores.values())
    
    # Check for extreme values that would cause underflow
    max_log_likelihood = max(log_likelihoods)
    
    # If all scores are extremely low (very negative), return neutral confidence
    if max_log_likelihood < -1000:  # Very low likelihood threshold
        return 0.5
    
    # Subtract max value for numerical stability (prevents underflow)
    shifted_log_likelihoods = [ll - max_log_likelihood for ll in log_likelihoods]
    
    # Convert to probabilities
    try:
        probabilities = [np.exp(ll) for ll in shifted_log_likelihoods]
        total = sum(probabilities)
        
        # Avoid division by zero
        if total == 0 or np.isnan(total):
            return 0.5
            
        # Normalize and return confidence for the best model
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx] / total
        
        # Ensure confidence is within valid range
        return max(0.1, min(0.99, confidence))
        
    except (FloatingPointError, ValueError):
        return 0.5

def train_hmm_models():
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Configuration
    data_path = "audio/training_audio"
    classes = ['male', 'female']
    n_components = 2

    # Collect all audio files with their labels
    audio_files = []
    true_labels = []

    print("Scanning audio files...")
    for label in classes:
        class_path = os.path.join(data_path, label)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} does not exist!")
            continue
            
        for file_name in os.listdir(class_path):
            if file_name.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(class_path, file_name)
                audio_files.append(file_path)
                true_labels.append(label)

    if not audio_files:
        print("No audio files found! Please check your directory structure.")
        return None

    print(f"Found {len(audio_files)} audio files")
    class_distribution = pd.Series(true_labels).value_counts().to_dict()
    print(f"Class distribution: {class_distribution}")

    # Check if we have enough data for each class
    for label, count in class_distribution.items():
        if count < 2:
            print(f"Warning: Only {count} samples for {label}. Need at least 2 for HMM training.")

    # Use Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    predictions = []
    actual_labels = []
    file_results = []
    confidence_scores = []

    print("\nPerforming Leave-One-Out Cross-Validation...")
    for train_index, test_index in loo.split(audio_files):
        test_file = audio_files[test_index[0]]
        true_label = true_labels[test_index[0]]
        
        print(f"Testing on {os.path.basename(test_file)} (true: {true_label})")
        
        # Split data
        train_files = [audio_files[i] for i in train_index]
        train_labels = [true_labels[i] for i in train_index]
        
        # Train HMMs for each class using only training data
        models = {}
        
        for label in classes:
            # Get training files for this specific class
            class_train_files = [f for f, l in zip(train_files, train_labels) if l == label]
            
            if len(class_train_files) >= 2:  # Need at least 2 samples to train HMM
                model, valid_files = train_hmm_for_class(class_train_files, n_components)
                if model is not None:
                    models[label] = model
                    print(f"  - Trained {label} HMM on {len(valid_files)} samples")
        
        if len(models) < 1:
            print(f"  - Could not train any models. Skipping...")
            predictions.append('unknown')
            confidence_scores.append(0.5)
            continue
        
        # Predict on test file
        test_features = extract_mfcc(test_file)
        if test_features is None or len(test_features) < 5:
            print(f"  - Could not extract features from test file. Skipping...")
            predictions.append('unknown')
            confidence_scores.append(0.5)
            continue
        
        # Score against each model
        scores = {}
        for label, model in models.items():
            try:
                score = model.score(test_features)
                scores[label] = score
            except Exception as e:
                print(f"  - Error scoring {label} model: {e}")
                scores[label] = -float('inf')
        
        if not scores:
            predictions.append('unknown')
            confidence_scores.append(0.5)
            continue
        
        # Determine prediction (use the model with highest likelihood)
        best_label = max(scores.items(), key=lambda x: x[1])[0]
        predictions.append(best_label)
        actual_labels.append(true_label)
        
        # Calculate confidence safely
        confidence = calculate_confidence(scores)
        confidence_scores.append(confidence)
        
        file_results.append({
            'filename': os.path.basename(test_file),
            'true_label': true_label,
            'predicted_label': best_label,
            'confidence': confidence,
            'correct': best_label == true_label,
            'log_likelihoods': scores
        })
        
        print(f"  - Predicted: {best_label}, Confidence: {confidence:.3f}")

    # Calculate metrics (only on valid predictions)
    valid_indices = [i for i, pred in enumerate(predictions) if pred != 'unknown']
    valid_predictions = [predictions[i] for i in valid_indices]
    valid_actual = [actual_labels[i] for i in valid_indices]

    if valid_predictions:
        accuracy = accuracy_score(valid_actual, valid_predictions)
        report = classification_report(valid_actual, valid_predictions, target_names=classes)
        conf_matrix = confusion_matrix(valid_actual, valid_predictions)
        
        print("\n" + "="*50)
        print("HMM MODEL EVALUATION (Leave-One-Out CV)")
        print("="*50)
        print(f"Total files: {len(audio_files)}")
        print(f"Valid predictions: {len(valid_predictions)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:\n{report}")
        print(f"Confusion Matrix:\n{conf_matrix}")
    else:
        print("No valid predictions were made!")
        accuracy = 0
        report = "No valid predictions"
        conf_matrix = "[]"

    # Save evaluation report
    os.makedirs('models', exist_ok=True)
    with open('models/hmm_model_evaluation.txt', 'w') as f:
        f.write("HMM Model Evaluation (Leave-One-Out CV)\n")
        f.write("="*50 + "\n")
        f.write(f"Total audio files: {len(audio_files)}\n")
        f.write(f"Valid predictions: {len(valid_predictions)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + "\n\n")
        f.write("Detailed Results:\n")
        for result in file_results:
            f.write(f"{result['filename']}: True={result['true_label']}, "
                   f"Pred={result['predicted_label']}, Conf={result['confidence']:.3f}, "
                   f"Correct={result['correct']}\n")

    # Train final models on ALL data for deployment
    print("\nTraining final models on entire dataset...")
    final_models = {}

    for label in classes:
        class_files = [f for f, l in zip(audio_files, true_labels) if l == label]
        if len(class_files) >= 2:
            model, valid_files = train_hmm_for_class(class_files, n_components)
            if model is not None:
                final_models[label] = model
                print(f"Final {label} HMM trained on {len(valid_files)} samples")
                
                # Test the model on its own training data
                try:
                    test_features = extract_mfcc(class_files[0])
                    if test_features is not None:
                        score = model.score(test_features)
                        print(f"  - Model score on sample: {score:.2f}")
                except:
                    pass
        else:
            print(f"Warning: Only {len(class_files)} samples for {label}, skipping final model")

    # Save the final trained models
    if 'male' in final_models:
        joblib.dump(final_models['male'], 'models/hmm_male.pkl')
        print("Final male model saved as 'models/hmm_male.pkl'")
    if 'female' in final_models:
        joblib.dump(final_models['female'], 'models/hmm_female.pkl')
        print("Final female model saved as 'models/hmm_female.pkl'")

    print("Evaluation report saved as 'models/hmm_model_evaluation.txt'")

    # Show summary
    print(f"\nSummary:")
    print(f"- Total audio files processed: {len(audio_files)}")
    print(f"- Successful predictions: {len(valid_predictions)}")
    print(f"- Overall accuracy: {accuracy:.3f}")
    
    # Additional diagnostics
    if file_results:
        avg_confidence = np.mean([r['confidence'] for r in file_results])
        print(f"- Average confidence: {avg_confidence:.3f}")
    
    return final_models

if __name__ == "__main__":
    train_hmm_models()