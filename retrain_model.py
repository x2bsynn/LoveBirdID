import os
import datetime
import shutil

def manual_correction(audio_path, predicted_label, true_label, weight):
    """Handle wrong predictions and collect corrected data."""
    print(f"\n❌ Prediction Error Detected!")
    print(f"File: {os.path.basename(audio_path)}")
    print(f"Predicted: {predicted_label}, Actual: {true_label}")
    
    # Validate weight input
    try:
        weight_val = float(weight)
        if weight_val <= 0 or weight_val > 100:  # Reasonable range for lovebirds
            print(f"⚠️  Warning: Weight {weight_val} seems unrealistic for lovebirds")
    except ValueError:
        print(f"⚠️  Error: Weight '{weight}' is not a valid number!")
        return False
    
    # Validate sex input
    if true_label not in ['male', 'female']:
        print(f"⚠️  Error: Sex must be 'male' or 'female', not '{true_label}'")
        return False
    
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