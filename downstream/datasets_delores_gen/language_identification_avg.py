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
duration = 7
print(duration,'duration')
#complete_data = pd.read_csv("/nlsasfs/home/nltm-pilot/ashishs/audio/complete_lid.csv")
#train, test = train_test_split(complete_data, test_size=0.2, random_state=1, stratify=complete_data['Label'])

class LanguageIdentificationTrain(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/audio/spec_byol_lid/"
        self.uttr_labels= pd.read_csv(self.feat_root+"train_voxforge.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'french':0, 'spanish':1, 'german':2, 'russian':3, 'english':4, 'italian':5}
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec

        label = row['Label']

        return uttr_melspec, self.labels_dict[label] #return normalized

class LanguageIdentificationTest(Dataset):
    def __init__(self,tfms = None,sample_rate=16000):        
        self.feat_root = "/nlsasfs/home/nltm-pilot/ashishs/audio/spec_byol_lid/"
        self.uttr_labels= pd.read_csv(self.feat_root+"test_voxforge.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'french':0, 'spanish':1, 'german':2, 'russian':3, 'english':4, 'italian':5}
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)
    
    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        uttr_melspec = np.load(uttr_path) #load unsqueezed normalized melspec

        label = row['Label']

        return uttr_melspec, self.labels_dict[label] #return normalized
