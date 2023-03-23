import torch
import numpy as np
import torch.utils.data as data
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
import torch.nn.functional as f
import pytorch_lightning as pl

from utils import extract_log_mel_spectrogram, extract_window, extract_log_mel_spectrogram_torch, extract_window_torch, MelSpectrogramLibrosa


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

    def __init__(self, args, data_dir_list, labels, tfms):
        self.audio_files_list = data_dir_list
        self.to_mel_spec = MelSpectrogramLibrosa()
        self.tfms = tfms
        self.length = 12 #args.length_wave,8
        self.norm_status = args.use_norm
        self.labels = labels

    def __getitem__(self, idx):
        audio_file = self.audio_files_list[idx]
        label = self.labels[idx]
        wave,sr = librosa.core.load(audio_file, sr=AUDIO_SR)
        wave = torch.tensor(wave)
        
        waveform = extract_window_torch(self.length, wave) #extract a window

        if self.norm_status == "l2":
            waveform = f.normalize(waveform,dim=-1,p=2) #l2 normalize

        log_mel_spec = extract_log_mel_spectrogram_torch(waveform, self.to_mel_spec) #convert to logmelspec
        log_mel_spec = log_mel_spec.unsqueeze(0)

        if self.tfms:
            lms = self.tfms(log_mel_spec) #do augmentations

        return lms, label

    def __len__(self):
        return len(self.audio_files_list)



class BaselineDataModule(pl.LightningDataModule):
    def __init__(self, args, tfms, train_data_dir_list='./', labels='None', valid_data_dir_list='./', batch_size=8, num_workers = 8):
        super().__init__()
        self.args = args
        self.data_dir_train = train_data_dir_list
        self.data_dir_valid = valid_data_dir_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformation = tfms
        self.dataset_sizes = {}
        self.labels = labels

#         num_train = 57146
#         indices = list(range(num_train))
#         valid_size = 0.2
#         split = int(np.floor(valid_size * num_train))
            
#         np.random.shuffle(indices)
#         self.train_idx, self.valid_idx = indices[split:], indices[:split]

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            self.train_dataset  = BARLOW(self.args, self.data_dir_train, self.labels, self.transformation)

            # self.val_dataset = BARLOW(self.args,self.data_dir,self.transformation)
            
            self.dataset_sizes['train'] = len(self.train_dataset)
            # self.dataset_sizes['val'] = len(self.val_dataset)
            
                        
        # if stage == 'test' or stage is None:
        #     self.test_dataset= datasets.ImageFolder(os.path.join(self.data_dir, 'test'), 
        #                                             transform=self.test_transforms)
        #     self.dataset_sizes['test'] = len(self.test_dataset)
                   
            
    # def _sampler(self):
    #     targets = self.train_dataset.targets
    #     class_sample_counts = torch.tensor(
    #         [(targets == t).sum() for t in torch.unique(targets, sorted=True)])

    #     weight = 1. / class_sample_counts.double()
    #     samples_weight = torch.tensor([weight[t] for t in targets])

    #     sampler = WeightedRandomSampler(
    #                     weights=samples_weight,
    #                     num_samples=len(samples_weight),
    #                     replacement=True)
    #     return sampler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          shuffle = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          drop_last =True,
                          pin_memory=True)
    
    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, 
    #                       shuffle=False,
    #                       batch_size=self.batch_size, 
    #                       num_workers=self.num_workers,
    #                       drop_last = True,
    #                       pin_memory = True)

    
    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, 
    #               batch_size = self.batch_size, 
    #               num_workers= self.num_workers,
    #               shuffle = False,
    #               drop_last = True,
    #               pin_memory = True)
    
    def num_classes(self):
        return 2
