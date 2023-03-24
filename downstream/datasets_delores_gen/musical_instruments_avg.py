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

duration = 4
print(duration,'duration')

class MusicalInstrumentsTrain(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/music/spec_byol_music/"
        self.uttr_labels= pd.read_csv(self.feat_root+"train_data.csv")
        self.sample_rate = sample_rate
        self.no_of_classes= 11
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms
    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec

        label = row['Label']

        return uttr_melspec, label #return normalized


class MusicalInstrumentsTest(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root = "/nlsasfs/home/nltm-pilot/ashishs/music/spec_byol_music/"
        self.uttr_labels= pd.read_csv(self.feat_root+"test_data.csv")
        self.sample_rate = sample_rate
        self.no_of_classes = 11
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms
    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec

        label = row['Label']

        return uttr_melspec, label #return normalized

        
