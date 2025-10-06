# validate_audio.py
import os
import librosa
import numpy as np
import pandas as pd

def validate_audio_file(file_path, min_duration=0.5, min_amplitude=0.05, min_energy=0.01):
    """Validate a single audio file."""
    try:
        y, sr = librosa.load(file_path)
        
        duration = len(y) / sr
        max_amplitude = np.max(np.abs(y))
        rms_energy = np.sqrt(np.mean(y**2))
        
        issues = []
        if duration < min_duration:
            issues.append(f"Too short ({duration:.2f}s < {min_duration}s)")
        if max_amplitude < min_amplitude:
            issues.append(f"Too quiet ({max_amplitude:.3f} < {min_amplitude})")
        if rms_energy < min_energy:
            issues.append(f"Low energy ({rms_energy:.3f} < {min_energy})")
            
        return {
            'duration': duration,
            'max_amplitude': max_amplitude,
            'rms_energy': rms_energy,
            'issues': issues,
            'valid': len(issues) == 0
        }
    except Exception as e:
        return {
            'duration': 0,
            'max_amplitude': 0,
            'rms_energy': 0,
            'issues': [f"Load error: {str(e)}"],
            'valid': False
        }

def validate_audio_directory(directory_path, min_duration=0.5, min_amplitude=0.05, min_energy=0.01):
    """Validate all audio files in a directory."""
    results = []
    problematic_files = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(root, file)
                result = validate_audio_file(file_path, min_duration, min_amplitude, min_energy)
                
                result['file_path'] = file_path
                result['filename'] = file
                result['directory'] = root
                
                results.append(result)
                
                if not result['valid']:
                    problematic_files.append((file_path, result['issues']))
    
    # Create summary report
    valid_count = sum(1 for r in results if r['valid'])
    total_count = len(results)
    
    print(f"\nAudio Validation Report for: {directory_path}")
    print("="*50)
    print(f"Total files: {total_count}")
    print(f"Valid files: {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"Problematic files: {total_count - valid_count}")
    
    if problematic_files:
        print("\nProblematic Files:")
        for file_path, issues in problematic_files:
            print(f"⚠️  {os.path.basename(file_path)}: {', '.join(issues)}")
    
    # Save detailed report to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv('audio_validation_report.csv', index=False)
        print(f"\nDetailed report saved to 'audio_validation_report.csv'")
    
    return results

if __name__ == "__main__":
    # Validate training data
    print("Validating training audio...")
    validate_audio_directory("audio/training_audio")
    
    # Validate test data
    print("\nValidating test audio...")
    validate_audio_directory("audio/new_test_data")