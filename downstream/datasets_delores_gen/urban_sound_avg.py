import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets_delores_gen.data_utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa, signal_to_frame, get_avg_duration
from datasets_delores_gen.data_utils import DataUtils
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
#random sample is taken from the whole audio frame
complete_data = pd.read_csv("/nlsasfs/home/nltm-pilot/ashishs/UrbanSound8K/audio/UrbanSound8K_final.csv")
duration = 4
print(duration,'duration')
test = complete_data.loc[complete_data['fold'] == 10]
train = pd.concat([complete_data,test]).drop_duplicates(keep=False)

#train, test = train_test_split(complete_data, test_size=0.2, random_state=1, stratify=complete_data['Label'])

class UrbanSoundTrain(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/UrbanSound8K/audio/"
        self.uttr_labels= train
        self.sample_rate = sample_rate
        self.no_of_classes = 10
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['spec_Norm'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec

        label = row['classID']

        return uttr_melspec, label #return normalized

class UrbanSoundTest(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/UrbanSound8K/audio/"
        self.uttr_labels= test
        self.sample_rate = sample_rate
        self.no_of_classes = 10
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['spec_Norm'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec

        label = row['classID']

        return uttr_melspec, label #return normalized
        
