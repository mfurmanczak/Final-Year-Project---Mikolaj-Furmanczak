import os

import librosa
import numpy as np
import pandas as pd

from tqdm import tqdm
     

US8K_AUDIO_PATH = os.path.abspath('UrbanSound8K/audio/')
US8K_METADATA_PATH = os.path.abspath('UrbanSound8K/metadata/UrbanSound8K.csv')  


us8k_metadata_df = pd.read_csv(US8K_METADATA_PATH,
                               usecols=["slice_file_name", "fold", "classID"],
                               dtype={"fold": "uint8", "classID" : "uint8"})
HOP_LENGTH = 512 
WINDOW_LENGTH = 512 
N_MEL = 128             


def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                        sr=sampling_rate, 
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH, 
                                                        n_mels=N_MEL)

        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        
        melspectrogram_length = melspectrogram_db.shape[1]
        
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db, 
                                                        size=num_of_samples, 
                                                        axis=1, 
                                                        constant_values=(0, -80.0))
    except Exception as e:
        print("\nError encountered while parsing files\n>>", e)
        return None 
    
    return melspectrogram_db

SOUND_DURATION = 2.95  

features = []


for index, row in tqdm(us8k_metadata_df.iterrows(), total=len(us8k_metadata_df)):
    file_path = f'{US8K_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
    audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')
    
    melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)
    label = row["classID"]
    fold = row["fold"]
    
    features.append([melspectrogram, label, fold])


us8k_df = pd.DataFrame(features, columns=["melspectrogram", "label", "fold"])


WRITE_DATA = True

if WRITE_DATA:
  us8k_df.to_pickle("us8k_df.pkl")