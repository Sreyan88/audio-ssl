import torch
import numpy as np
import torch.utils.data as data
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import torch.nn.functional as f
import pytorch_lightning as pl

from src.utils import extract_log_mel_spectrogram, extract_window, \
extract_log_mel_spectrogram_torch, extract_window_torch, MelSpectrogramLibrosa


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
    batch_1 = batch_1.unsqueeze(1)

    batch_2 = [torch.Tensor(t) for _,t in batch]
    batch_2 = torch.nn.utils.rnn.pad_sequence(batch_2,batch_first = True)
    batch_2 = batch_2.unsqueeze(1)

    return batch_1, batch_2


class Basedataset(Dataset):

    def __init__(self, conf, data_dir_list, tfms):

        self.audio_files_list = data_dir_list
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms
        self.length = args.length_wave
        self.norm_status = args.use_norm

    def __getitem__(self, idx):

        audio_file = self.audio_files_list[idx]
        wave,sr = librosa.core.load(audio_file, sr=config.sampling_rate)
        wave = torch.tensor(wave)

        if self.norm_status == "l2":
            waveform = f.normalize(waveform,dim=-1,p=2) #l2 normalize

        log_mel_spec = extract_log_mel_spectrogram_torch(wave, self.to_mel_spec) #convert to logmelspec

        log_mel_spec = log_mel_spec.T

        n_frames = log_mel_spec.shape[0]

        p = 1024 - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0:1024, :]


        log_mel_spec = log_mel_spec.unsqueeze(0)
        log_mel_spec = log_mel_spec.permute(0,2,1)

        if self.tfms:
            lms = self.tfms(log_mel_spec) #do augmentations

        return (lms[0].permute(0,2,1), lms[1].permute(0,2,1))

    def __len__(self):
        return len(self.audio_files_list)





class BaselineDataModule(pl.LightningDataModule):

    def __init__(self, args, tfms, train_data_dir_list='./',valid_data_dir_list='./', batch_size=8, num_workers = 8):
        super().__init__()
        self.args = args
        self.data_dir_train = train_data_dir_list
        self.data_dir_valid = valid_data_dir_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformation = tfms
        self.dataset_sizes = {}

    def setup(self, stage = None):

        if stage == 'fit' or stage is None:

            self.train_dataset  = Basedataset(self.args,self.data_dir_train,self.transformation)
            self.dataset_sizes['train'] = len(self.train_dataset)
            

    def train_dataloader(self):

        return DataLoader(self.train_dataset, 
                          shuffle = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          drop_last =True,
                          pin_memory=True)
    
    def num_classes(self):
        return 2
