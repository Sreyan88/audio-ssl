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

class SpeechCommandsV2_35_Train(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/scv2/spec_byol_scv2_test/"
        self.uttr_labels= pd.read_csv("/nlsasfs/home/nltm-pilot/ashishs/speech_cmd_v2_data/"+"train_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = dict(zip(['sheila', 'left', 'four', 'up', 'stop', 'off', 'dog', 'go', 'three', 'cat', 'follow', 'wow', 'down', 'two', 'happy', 'six', 'one', 'eight', 'on', 'five', 'bird', 'nine', 'yes', 'marvin', 'tree', 'learn', 'seven', 'zero', 'right', 'no', 'visual', 'backward', 'forward', 'bed', 'house'],list(range(0,35))))
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


class SpeechCommandsV2_35_Test(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/scv2/spec_byol_scv2_test/"
        self.uttr_labels= pd.read_csv("/nlsasfs/home/nltm-pilot/ashishs/speech_cmd_v2_data/"+"test_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = dict(zip(['sheila', 'left', 'four', 'up', 'stop', 'off', 'dog', 'go', 'three', 'cat', 'follow', 'wow', 'down', 'two', 'happy', 'six', 'one', 'eight', 'on', 'five', 'bird', 'nine', 'yes', 'marvin', 'tree', 'learn', 'seven', 'zero', 'right', 'no', 'visual', 'backward', 'forward', 'bed', 'house'],list(range(0,35))))
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
