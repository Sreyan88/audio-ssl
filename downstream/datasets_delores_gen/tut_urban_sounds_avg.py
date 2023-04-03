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

duration = 10
print(duration,'duration')

class TutUrbanSoundsTrain(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):                
        self.feat_root =  "/nlsasfs/home/nltm-pilot/ashishs/acoustic_TUT/zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development/"
        self.uttr_labels= pd.read_csv(self.feat_root+"train_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3, 'park': 4,
         'public_square': 5, 'shopping_mall': 6, 'street_pedestrian': 7,
         'street_traffic': 8, 'tram': 9}
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
class TutUrbanSoundsTest(Dataset):
    def __init__(self,tfms=None,sample_rate=16000):        
        self.feat_root = "/nlsasfs/home/nltm-pilot/ashishs/acoustic_TUT/zenodo.org/record/1228142/files/TUT-urban-acoustic-scenes-2018-development/"
        self.uttr_labels= pd.read_csv(self.feat_root+"test_data.csv")
        self.sample_rate = sample_rate
        self.labels_dict = {'airport': 0, 'bus': 1, 'metro': 2, 'metro_station': 3, 'park': 4,
         'public_square': 5, 'shopping_mall': 6, 'street_pedestrian': 7,
         'street_traffic': 8, 'tram': 9}
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
