import torch
import numpy as np
import torch.utils.data as data
import tensorflow as tf
from torch.utils.data import Dataset
import librosa
import torch.nn.functional as f
from augmentations import MixupBYOLA, RandomResizeCrop, RunningNorm

from utils import extract_log_mel_spectrogram, extract_window, extract_log_mel_spectrogram_torch, extract_window_torch, MelSpectrogramLibrosa
import random

tf.config.set_visible_devices([], 'GPU')
AUDIO_SR = 16000


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## padd
    batch_1 = [torch.Tensor(t) for t,_ in batch]
    batch_1 = torch.nn.utils.rnn.pad_sequence(batch_1,batch_first = True)
    #batch = batch.reshape()
    batch_1 = batch_1.unsqueeze(1)

    batch_2 = [torch.Tensor(t) for _,t in batch]
    batch_2 = torch.nn.utils.rnn.pad_sequence(batch_2,batch_first = True)
    #batch = batch.reshape()
    batch_2 = batch_2.unsqueeze(1)

    return batch_1, batch_2


class BARLOW(Dataset):

    def __init__(self, args, data_dir_list, tfms):
        self.audio_files_list = data_dir_list
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms
        self.length = args.length_wave
        self.norm_status = args.use_norm

    def __getitem__(self, idx):
        audio_file = self.audio_files_list[idx]
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        
        waveform = extract_window_torch(self.length, wave) #extract a window

        if self.norm_status == "l2":
            waveform = f.normalize(waveform,dim=-1,p=2) #l2 normalize

        log_mel_spec = extract_log_mel_spectrogram_torch(waveform, self.to_mel_spec) #convert to logmelspec
        log_mel_spec = log_mel_spec.unsqueeze(0)

        if self.tfms:
            lms = self.tfms(log_mel_spec) #do augmentations

        if not isinstance(lms, list):
            lms = [lms]

        return idx, lms

    def __len__(self):
        return len(self.audio_files_list)


class DeepCluster(Dataset):

    def __init__(self, data_dir_list,epoch_samples,args):
        self.audio_files_list = data_dir_list
        self.norm_status = args.use_norm
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.length = 12 #args.length_wave

    def __getitem__(self, idx):
        audio_file = self.audio_files_list[idx]
        #wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        #log_mel_spec = extract_log_mel_spectrogram(wave)
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        
        waveform = extract_window_torch(self.length, wave) #extract a window
        log_mel_spec = extract_log_mel_spectrogram_torch(waveform, self.to_mel_spec) #convert to logmelspec
        log_mel_spec = log_mel_spec.unsqueeze(0)
        #mean-variance normalization 
        if self.norm_status == 'byol':
            log_mel_spec = self.pre_norm(log_mel_spec)
        t = random.randint(0,log_mel_spec.shape[2]-97)       
        return t, log_mel_spec

    def __len__(self):
        return len(self.audio_files_list)

class DeepCluster_Reassigned(Dataset):

    def __init__(self,args,audio_file_list,label_list,audio_indexes,tfms):
        self.audio_files = audio_file_list
        self.audio_labels = label_list
        self.audio_indexes = audio_indexes
        self.dataset = self.make_dataset()
        self.aug = tfms
        self.length = 12 #args.length_wave
        self.to_mel_spec = MelSpectrogramLibrosa()

        
    def make_dataset(self):
        label_to_idx = {label: idx for idx, label in enumerate(set(self.audio_labels))}
        audiopath_w_labels = []
        for i, index in enumerate(self.audio_indexes):
            path = self.audio_files[index]
            pseudolabel = label_to_idx[self.audio_labels[index]] #could have been pseudolabels, bekar confusion change later
            audiopath_w_labels.append((path,pseudolabel))
            
        return audiopath_w_labels
            
    def __getitem__(self, idx):
        audio_file,label = self.dataset[idx]
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        
        waveform = extract_window_torch(self.length, wave) #extract a window
        log_mel_spec = extract_log_mel_spectrogram_torch(waveform, self.to_mel_spec) #convert to logmelspec
        log_mel_spec = log_mel_spec.unsqueeze(0)        
        log_mel_spec = self.aug(log_mel_spec.clone().detach()) #augmentation 
        return log_mel_spec,label

    def __len__(self):
        return len(self.audio_files)        


# class BARLOW(Dataset):

#     def __init__(self, data_dir_list):
#         self.audio_files_list = data_dir_list

#     def __getitem__(self, idx):
#         audio_file = self.audio_files_list[idx]
#         wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
#         x = tf.math.l2_normalize(wave, epsilon=1e-9)

#         waveform_a = extract_window(x)
#         log_mel_spec_a = extract_log_mel_spectrogram(waveform_a)

#         waveform_b = extract_window(x)
#         log_mel_spec_b = extract_log_mel_spectrogram(waveform_b)

#         return log_mel_spec_a.numpy() , log_mel_spec_b.numpy()

#     def __len__(self):
#         return len(self.audio_files_list)
