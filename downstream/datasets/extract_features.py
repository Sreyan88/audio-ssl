import os 
import numpy as np
#import tensorflow as tf 
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import pandas as pd
from functools import partial
#tf.config.set_visible_devices([], 'GPU')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        help='down_stream task name')
    parser.add_argument('--no_workers', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--prefix', default='', type=str,
                        help='number of total epochs to run')     
    parser.add_argument('--suffix', default='spec' , type=str)
    parser.add_argument('--file', default=None , type=str)               
    return parser

def extract_window(wav, seg_length=16000, data_size=0.96):
    """Extract random window of data_size second"""
    unit_length = int(data_size * 16000)
    length_adj = unit_length - len(wav)
    if length_adj > 0:
        half_adj = length_adj // 2
        wav = F.pad(wav, (half_adj, length_adj - half_adj))

    # random crop unit length wave
    length_adj = unit_length - len(wav)
    start = random.randint(0, length_adj) if length_adj > 0 else 0
    wav = wav[start:start + unit_length]

    return wav

class MelSpectrogramLibrosa:
    """Mel spectrogram using librosa."""
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        return torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))
    
func = MelSpectrogramLibrosa()
    
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_log_mel_spectrogram(audio_path,
                                sample_rate=16000,
                                frame_length=400,
                                frame_step=160,
                                fft_length=1024,
                                n_mels=64,
                                fmin=60.0,
                                fmax=7800.0):
        """Extract frames of log mel spectrogram from a raw waveform."""
        
        waveform,sr = librosa.core.load(audio_path, sample_rate)
        wave_audio = torch.tensor(waveform)
        wave_audio = extract_window(wave_audio,data_size=duration) #please put duration here
        wave_random1sec=f.normalize(wave_audio,dim=-1,p=2)
        uttr_melspec = func(wave_random1sec, self.to_mel_spec)
        uttr_melspec = (uttr_melspec + torch.finfo().eps).log()
        return uttr_melspec.numpy()


def write_feats(root_dir,prefix,suffix,files_array):
    for file in tqdm(files_array):
        if file.endswith("wav"):
            feat = extract_log_mel_spectrogram(os.path.join(root_dir,file))
            file_path =os.path.join(root_dir,suffix,file)
            create_dir(os.path.dirname(file_path))
            np.save(file_path,feat)

def run_parallel(args):
    create_dir(os.path.join(args.root_dir,args.suffix))
    if(args.file != None):
        list_files = np.array(pd.read_csv(os.path.join(args.root_dir,args.file))['AudioPath'].values)
    else:
        list_files = np.array(os.listdir(os.path.join(args.root_dir,args.prefix)))
    print(args.prefix)
    list_ranges = np.array_split(list_files, args.no_workers)
    pfunc=partial(write_feats,args.root_dir,args.prefix,args.suffix)
    pool = Pool(processes=len(list_ranges))
    pool.map(pfunc, list_ranges)

# Driver code
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    run_parallel(args)