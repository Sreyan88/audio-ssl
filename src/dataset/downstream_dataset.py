import os
import torch
import librosa
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as f
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.utils import extract_log_mel_spectrogram, extract_window, MelSpectrogramLibrosa

class DownstreamDatasetHF(Dataset):

    def __init__(self, args, config, split, tfms=None):
        if 'speech_commands' in args.task:
            self.version = 'v0.02' if '2' in args.task else 'v0.01'

        self.dataset = load_dataset(self.task, self.version, split = split)

        if ('speech_commands' in args.task) and (self.version = 'v0.02') and ('35' in args.task):
            pass

        self.uttr_labels= pd.read_csv(self.feat_root)
        self.sample_rate = self.config["downstream"]["input"]["sampling_rate"]
        self.duration= self.config["run"]["duration"]
        self.labels_dict = self.get_id2label()
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def get_id2label(self):
        labels = speech_commands_v1["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label

        return id2label

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        wave_audio = self.dataset["array"][idx]
        wave_audio = torch.tensor(wave_audio) #convert into torch tensor
        wave_audio = extract_window(wave_audio, data_size=self.duration) #extract fixes size length
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec) #convert into logmel
        uttr_melspec=uttr_melspec.unsqueeze(0) # unsqueeze it for input to CNN

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec) #if tfms present, normalize it

        label = self.dataset["label"][idx]

        return uttr_melspec, label #return normalized

class DownstreamDataset(Dataset):
    def __init__(self, args, config, split, tfms=None, labels_dict=None):
        self.task = args.task
        self.split = split
        if self.split == 'train':
            self.dataset= pd.read_csv(args.train_csv)
        elif self.split == 'valid':
            self.dataset= pd.read_csv(args.valid_csv)
        elif self.split == 'test':
            self.dataset= pd.read_csv(args.valid_csv)
        self.sample_rate = self.config["downstream"]["input"]["sampling_rate"]
        self.duration= self.config["run"]["duration"]
        self.labels_dict = self.get_label2id(self.dataset) if labels_dict is None else labels_dict
        self.no_of_classes= len(self.labels_dict)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms

    def get_label2id(self):
        unique_labels = set(self.dataset['label'])
        id2label = {i:k for k,i in enumerate(unique_labels)}

        return id2label

    def __len__(self):
        return len(self.uttr_labels)

    def __getitem__(self, idx):
        row = self.uttr_labels.iloc[idx,:]
        wave_audio,sr = librosa.core.load(row['AudioPath'], sr=self.sample_rate) #load file
        wave_audio = torch.tensor(wave_audio) #convert into ttorch tensor
        wave_audio = extract_window(wave_audio,data_size=duration) #extract fixes size length
        uttr_melspec = extract_log_mel_spectrogram(wave_audio, self.to_mel_spec) #convert into logmel
        uttr_melspec=uttr_melspec.unsqueeze(0) #unsqueeze it

        if self.tfms:
            uttr_melspec=self.tfms(uttr_melspec) #if tfms present, normalize it

        label = row['label']

        return uttr_melspec, label #return normalized
