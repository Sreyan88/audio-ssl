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

duration = 1
print(duration,'duration')

class SpeechCommandsV2Train(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/scv2/spec_byol_scv2_test/"
        self.uttr_labels= pd.read_csv("/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/speechv2/train/"+"train_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'unknown': 0, 'down': 1, 'go': 2, 'silence': 3, 'on': 4, 'stop': 5, 'left': 6, 'no': 7,'up': 8, 'yes': 9, 'off': 10, 'right': 11}
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['Path'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec
        label = row['Label']

        return uttr_melspec, self.labels_dict[label] #return normalized

class SpeechCommandsV2Test(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root = "/nlsasfs/home/nltm-pilot/ashishs/scv2/spec_byol_scv2_test/"
        self.uttr_labels= pd.read_csv("/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/speechv2/train/"+"test_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'unknown': 0, 'down': 1, 'go': 2, 'silence': 3, 'on': 4, 'stop': 5, 'left': 6, 'no': 7,'up': 8, 'yes': 9, 'off': 10, 'right': 11}
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['Path'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec
        
        label = row['Label']

        return uttr_melspec, self.labels_dict[label] #return normalized