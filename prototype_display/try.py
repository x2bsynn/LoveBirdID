import os

path = "audio/corrected_data/male/corrected_male_20250921_191552.wav"
print("Exists?", os.path.exists(path))
print("Absolute path:", os.path.abspath(path))
