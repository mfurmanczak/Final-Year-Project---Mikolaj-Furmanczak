import os
import time

import numpy as np
import pandas as pd
import librosa

import mutagen
import mutagen.wave
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def create_dataset_df(csv_file):
    dataset_df = pd.read_csv(csv_file)
    filepaths = []
    for i, row in dataset_df.iterrows():
        filepaths.append(os.path.join('UrbanSound8K/audio',
                         'fold'+str(row['fold']), row['slice_file_name']))
    dataset_df['filepath'] = filepaths
    return dataset_df


def get_audio_metadata_mutagen(filepath):
    metadata = {}
    f = mutagen.wave.WAVE(filepath)
    metadata['length'] = f.info.length
    metadata['bitrate'] = f.info.bitrate
    metadata['channels'] = f.info.channels
    metadata['sample_rate'] = f.info.sample_rate
    metadata['bits_per_sample'] = f.info.bits_per_sample
    return metadata


def compute_audio_statistics(dataset_df):
    metadata_dict = {'length': [], 'bitrate': [],
                     'channels': [], 'sample_rate': [], 'bits_per_sample': []}
    for filepath in dataset_df['filepath']:
        metadata = get_audio_metadata_mutagen(filepath)
        for key in metadata_dict.keys():
            metadata_dict[key].append(metadata[key])
    for key in metadata_dict.keys():
        dataset_df[key] = metadata_dict[key]

    return dataset_df


def class_count(dataset_df):
    values = dataset_df.groupby('class').slice_file_name.count()
    print(f"Class:{'':<20}{'Count: '}")
    for i, (index, value) in enumerate(values.items()):
        print(f"{index:<25} | {value:>6}")
        
    


def audio_statistics(dataset_df):
    dataset_df = dataset_df.drop(
        columns=['fold', 'slice_file_name', 'fsID', 'start', 'end'])
    audio_statistics_df = compute_audio_statistics(dataset_df)
    print(audio_statistics_df.describe())
    print(audio_statistics_df['sample_rate'].value_counts())
    print(audio_statistics_df['bits_per_sample'].value_counts())
    print(audio_statistics_df.groupby('class').describe())


def plot_waveform(dataset_df):
    random_samples = dataset_df.groupby('class').sample(1)
    audio_samples, labels = random_samples['filepath'].tolist(
    ), random_samples['class'].tolist()

    for i, filepath in enumerate(audio_samples):
        data, sr = librosa.load(filepath)
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(labels[i])
        plt.show()


def plot_mel_spectrogram(dataset_df):
    random_samples = dataset_df.groupby('class').sample(1)
    audio_samples, labels = random_samples['filepath'].tolist(
    ), random_samples['class'].tolist()

    HOP_LENGTH = 512
    WINDOW_LENGTH = 512
    N_MEL = 128
    for i, filepath in enumerate(audio_samples):
        audio_file, sample_rate = librosa.load(audio_samples[i])
        melspectrogram = librosa.feature.melspectrogram(y=audio_file,
                                                        sr=sample_rate,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH,
                                                        n_mels=N_MEL)

        librosa.display.specshow(melspectrogram, sr=sample_rate, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis='mel')
        plt.title(labels[i])
        plt.show()


def plot_mel_spectrogram_db(dataset_df):
    random_samples = dataset_df.groupby('class').sample(1)
    audio_samples, labels = random_samples['filepath'].tolist(
    ), random_samples['class'].tolist()

    # n_fft = 2048
    HOP_LENGTH = 512
    WINDOW_LENGTH = 512
    N_MEL = 128
    for i, filepath in enumerate(audio_samples):
        audio_file, sample_rate = librosa.load(audio_samples[i])
        melspectrogram = librosa.feature.melspectrogram(y=audio_file,
                                                        sr=sample_rate,
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH,
                                                        n_mels=N_MEL)

        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        librosa.display.specshow(melspectrogram_db, sr=sample_rate, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis='mel')
        plt.title(labels[i])
        plt.show()


def main():
    dataset_df = create_dataset_df('UrbanSound8K/metadata/UrbanSound8K.csv')
    compute_audio_statistics(dataset_df)
    class_count(dataset_df)
    audio_statistics(dataset_df)
    plot_waveform(dataset_df)
    plot_mel_spectrogram(dataset_df)
    plot_mel_spectrogram_db(dataset_df)


if __name__ == '__main__':
    main()