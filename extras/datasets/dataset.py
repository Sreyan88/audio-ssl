#from src.downstream_updated.datasets.iemocap_avg import  IEMOCAPTest, IEMOCAPTrain
#from src.downstream_updated.datasets.birdsong_dataset_avg import BirdSongDatasetTrain, BirdSongDatasetTest
#from src.downstream_updated.datasets.tut_urban_sounds_avg import TutUrbanSoundsTrain, TutUrbanSoundsTest
#from src.downstream_updated.datasets.speech_commands_v2_avg import SpeechCommandsV2Train, SpeechCommandsV2Test
#from src.downstream_updated.datasets.musical_instruments_avg import MusicalInstrumentsTrain, MusicalInstrumentsTest
#from src.downstream_updated.datasets.libri100_avg import Libri100Train, Libri100Test
#from src.downstream_updated.datasets.language_identification_avg import LanguageIdentificationTrain, LanguageIdentificationTest
from src.downstream.datasets.speech_commands_v1_avg import SpeechCommandsV1Train, SpeechCommandsV1Test
#from src.downstream_updated.datasets.voxceleb_avg import Voxceleb1DatasetTrain, Voxceleb1DatasetTest
#from src.downstream_updated.datasets.speech_commands_v2_avg_35 import SpeechCommandsV2_35_Train, SpeechCommandsV2_35_Test
import torch

def get_dataset(args, config, aug = None):
    #if args.down_stream_task == "birdsong_combined":
    #    return BirdSongDatasetTrain(tfms=aug), BirdSongDatasetTest(tfms=aug)
    if args.down_stream_task == "speech_commands_v1":
        return SpeechCommandsV1Train(args ,config, tfms=aug) , SpeechCommandsV1Test(args, config, tfms=aug)
    #elif args.down_stream_task == "speech_commands_v2":
    #    return SpeechCommandsV2Train(tfms=aug) , SpeechCommandsV2Test(tfms=aug)
    #elif args.down_stream_task == "speech_commands_v2_35":
    #    return SpeechCommandsV2_35_Train(tfms=aug) , SpeechCommandsV2_35_Test(tfms=aug)
    #elif args.down_stream_task == "libri_100":
    #    return Libri100Train(tfms=aug) , Libri100Test(tfms=aug)      
    #elif args.down_stream_task == "musical_instruments":
    #    return MusicalInstrumentsTrain(tfms=aug) , MusicalInstrumentsTest(tfms=aug)
    #elif args.down_stream_task == "iemocap":
    #    return IEMOCAPTrain(tfms=aug),IEMOCAPTest(tfms=aug)            
    #elif args.down_stream_task == "tut_urban":
    #    return TutUrbanSoundsTrain(tfms=aug),TutUrbanSoundsTest(tfms=aug)    
    #elif args.down_stream_task == "voxceleb_v1":
    #    return Voxceleb1DatasetTrain(tfms=aug) , Voxceleb1DatasetTest(tfms=aug)   
    #elif args.down_stream_task == "language_identification":
    #    return LanguageIdentificationTrain(tfms=aug), LanguageIdentificationTest(tfms=aug)                 
    else:
        raise NotImplementedError

