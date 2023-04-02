from datasets.iemocap_avg import  IEMOCAPTest, IEMOCAPTrain
from datasets.birdsong_dataset_avg import BirdSongDatasetTrain, BirdSongDatasetTest
from datasets.tut_urban_sounds_avg import TutUrbanSoundsTrain, TutUrbanSoundsTest
from datasets.speech_commands_v2_avg import SpeechCommandsV2Train, SpeechCommandsV2Test
from datasets.musical_instruments_avg import MusicalInstrumentsTrain, MusicalInstrumentsTest
from datasets.libri100_avg import Libri100Train, Libri100Test
from datasets.language_identification_avg import LanguageIdentificationTrain, LanguageIdentificationTest
from datasets.speech_commands_v1_avg import SpeechCommandsV1Train, SpeechCommandsV1Test
from datasets.voxceleb_avg import Voxceleb1DatasetTrain, Voxceleb1DatasetTest
from datasets.speech_commands_v2_avg_35 import SpeechCommandsV2_35_Train, SpeechCommandsV2_35_Test
import torch

def get_dataset(downstream_task_name,aug = None):
    if downstream_task_name == "birdsong_combined":
        return BirdSongDatasetTrain(tfms=aug), BirdSongDatasetTest(tfms=aug)
    elif downstream_task_name == "speech_commands_v1":
        return SpeechCommandsV1Train(tfms=aug) , SpeechCommandsV1Test(tfms=aug)
    elif downstream_task_name == "speech_commands_v2":
        return SpeechCommandsV2Train(tfms=aug) , SpeechCommandsV2Test(tfms=aug)
    elif downstream_task_name == "speech_commands_v2_35":
        return SpeechCommandsV2_35_Train(tfms=aug) , SpeechCommandsV2_35_Test(tfms=aug)
    elif downstream_task_name == "libri_100":
        return Libri100Train(tfms=aug) , Libri100Test(tfms=aug)      
    elif downstream_task_name == "musical_instruments":
        return MusicalInstrumentsTrain(tfms=aug) , MusicalInstrumentsTest(tfms=aug)
    elif downstream_task_name == "iemocap":
        return IEMOCAPTrain(tfms=aug),IEMOCAPTest(tfms=aug)            
    elif downstream_task_name == "tut_urban":
        return TutUrbanSoundsTrain(tfms=aug),TutUrbanSoundsTest(tfms=aug)    
    elif downstream_task_name == "voxceleb_v1":
        return Voxceleb1DatasetTrain(tfms=aug) , Voxceleb1DatasetTest(tfms=aug)   
    elif downstream_task_name == "language_identification":
        return LanguageIdentificationTrain(tfms=aug), LanguageIdentificationTest(tfms=aug)                 
    else:
        raise NotImplementedError

