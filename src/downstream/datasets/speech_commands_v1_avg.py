import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets.data_utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa, signal_to_frame, get_avg_duration
from datasets.data_utils import DataUtils
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
duration = 1
print(duration,'duration')
class SpeechCommandsV1Train(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/speechv1/train/"
        self.uttr_labels= pd.read_csv(self.feat_root+"train_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'unknown': 0, 'down': 1, 'go': 2, 'silence': 3, 'on': 4, 'stop': 5, 'left': 6, 'no': 7,'up': 8, 'yes': 9, 'off': 10, 'right': 11}
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate) #load file
        wave_audio = torch.tensor(wave_audio) #convert into ttorch tensor
        wave_audio = extract_window(wave_audio,data_size=duration) #extract fixes size length
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec) #convert into logmel
        uttr_melspec=uttr_melspec.unsqueeze(0) #unsqueeze it

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec) #if tfms present, normalize it

        label = row['Label']

        return uttr_melspec, self.labels_dict[label] #return normalized

class SpeechCommandsV1Test(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root = "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/speechv1/train/"
        self.uttr_labels= pd.read_csv(self.feat_root+"test_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'unknown': 0, 'down': 1, 'go': 2, 'silence': 3, 'on': 4, 'stop': 5, 'left': 6, 'no': 7,'up': 8, 'yes': 9, 'off': 10, 'right': 11}
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate) #load file
        wave_audio = torch.tensor(wave_audio) #convert into ttorch tensor
        wave_audio = extract_window(wave_audio,data_size=duration) #extract fixes size length
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec) #convert into logmel
        uttr_melspec=uttr_melspec.unsqueeze(0) #unsqueeze it

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec) #if tfms present, normalize it

        label = row['Label']

        return uttr_melspec, self.labels_dict[label] #return normalized
