import os
import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from src.downstream_updated.datasets.data_utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa, signal_to_frame, get_avg_duration
from src.downstream_updated.datasets.data_utils import DataUtils
import torch.nn.functional as f
from sklearn.model_selection import train_test_split

duration = 4

class IEMOCAPTrain(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/iemocap/IEMOCAP/"
        annotations_file=os.path.join(self.feat_root,"train_data.csv")
        self.uttr_labels= pd.read_csv(annotations_file)
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} 
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        wave_audio = extract_window(wave_audio,data_size=duration)
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec)
        uttr_melspec=uttr_melspec.unsqueeze(0)

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec)
        
        label = row['Label_id']
        return uttr_melspec, label


class IEMOCAPTest(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root =  "/nlsasfs/home/nltm-pilot/sandeshk/icassp/data/iemocap/iemocap/IEMOCAP/"
        annotations_file=os.path.join(self.feat_root,"test_data.csv")
        self.uttr_labels= pd.read_csv(annotations_file)
        self.sample_rate = sample_rate
        self.labels_dict ={'neu':0, 'ang':1, 'sad':2, 'hap':3} 
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        uttr_path =os.path.join(self.feat_root,row['AudioPath'])
        wave_audio,sr = librosa.core.load(uttr_path, sr=self.sample_rate)
        wave_audio = torch.tensor(wave_audio)
        wave_audio = extract_window(wave_audio,data_size=duration)
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec)
        uttr_melspec=uttr_melspec.unsqueeze(0)

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec)
        
        label = row['Label_id']
        return uttr_melspec, label

