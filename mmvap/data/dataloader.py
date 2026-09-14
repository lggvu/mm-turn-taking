import pandas as pd
import csv
import torch
import os
import pickle
import cv2
import random


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that maps numpy._core.* → numpy.core.* for files saved with NumPy 2.x."""
    def find_class(self, module, name):
        module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def _read_pickle_compat(path):
    """pd.read_pickle with a fallback that handles NumPy 2.x → 1.x pickle compatibility."""
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if 'numpy._core' not in str(exc):
            raise
        with open(path, 'rb') as fh:
            return _NumpyCompatUnpickler(fh).load()


from torch.utils.data import Dataset
from mmvap.data.audio_manager import AudioManager
from mmvap.data.data_config import config
from mmvap.data.transcript_processing import vads_from_transcript, vad_list_to_one_hot, training_labels_projection
from mmvap.data.codebook import Codebook, bin_times_to_frames
import einops
import numpy as np
from multiprocessing import Manager
import re


def read_csv_lines(file, start_line, end_line):
    """ 
        looking at the fastest way to read specific lines from a csv file...
    """

    with open(file, "r") as f:
        # lines = f.readlines()
        # target = lines[start_line:end_line]
        lines = list(csv.reader(f))[start_line:end_line]
        
    return lines


class ChunkedDataset(Dataset):
    def __init__(self, pickle_file: str, mode: str, fps:  int, sr: int) -> None:
        super().__init__()

        self.pickle_file = pickle_file
        self.dataset = None
        self.manager = Manager()
        self.fps = fps
        self.sr = sr
        # self.window_size = [2, self.sr*config['audio_window_length']]
        # self.future_window_size = [2, self.sr*config['future_context']]

        self.load_data()
        self.mode = mode

        assert mode in ['VAP', 'ind-4','ind-40', None]
        if mode is None:
            mode='VAP'
        
        if self.mode == 'VAP':
            bin_times = [0.2, 0.4, 0.6, 0.8]
            bin_frames = bin_times_to_frames(bin_times, frame_hz=self.sr)
            self.codebook = Codebook(bin_frames=bin_frames)
        

    def load_data(self):
        self.ids = np.load(self.pickle_file+'_ids.npy')
        self.start_times = np.load(self.pickle_file+'_starts.npy')
        self.end_times = np.load(self.pickle_file+'_ends.npy')
        self.vads = torch.load(self.pickle_file+'_vads.pt')
        self.vaps = torch.load(self.pickle_file+'_vaps.pt')


    def  __len__(self):
        return len(self.ids)
    

    def __getitem__(self, index):

        id =  self.ids[index].decode('utf8')
        start_time = self.start_times[index]
        end_time = self.end_times[index]
        vad = self.vads[index, ...]
        vap = self.vaps[index, ...] # read from .pt and this is already in the future, hence torch.Size([100, 2]) = 2 seconds

        # pad the current window with the next 2 seconds of future context to enable projection
        window_with_future = torch.concat((vad, vap), dim=0)

        if self.mode == 'VAP':
            vap = training_labels_projection(window_with_future, mode='VAP')
            inverse_vap = torch.stack((vap[:, 1, :], vap[:, 0, :]), dim=1) # just for augmentation
            inverse_vap = self.codebook.encode(inverse_vap)
            vap = self.codebook.encode(vap)

            # vap: 1 vap.shape: torch.Size([100, 2])
            # vap: 2 vap.shape: torch.Size([1000, 2, 4]) # [2, 4] represents the future
            # vap: 3 vap.shape: torch.Size([1000, 2, 4])
            # vap: 4 inverse_vap.shape: torch.Size([1000])
            # vap: 5 inverse_vap.shape: torch.Size([1000])
            return id, start_time, end_time, vad, vap, inverse_vap

        elif self.mode == 'ind-4':
            vap = training_labels_projection(window_with_future, mode='ind-4')

        elif self.mode == 'ind-40':
            vap = training_labels_projection(window_with_future, mode='ind-40')
        
        # print(f'vap.shape: {vap.shape}')
        inverse_vap = torch.stack((vap[:, 1, :], vap[:, 0, :]), dim=1)
        vap = einops.rearrange(vap, "n c d -> n (c d)")
        inverse_vap = einops.rearrange(inverse_vap, "n c d -> n (c d)")
        
        return id, start_time, end_time, vad, vap, inverse_vap


class AudioDataset(ChunkedDataset):

    
    def __init__(self, pickle_file: str, mode: str, wavdir: str, *args, **kwargs) -> None:
        self.wavdir = wavdir
        self.normalize = True
        super().__init__(pickle_file, mode, *args, **kwargs)

    
    def get_audio_chunk(self, id, start, end, sample_rate):
        # locate the audio file 
        wavfile = os.path.join(self.wavdir, f"{id}.wav")
        
        # audio segment 
        audio = AudioManager(wavfile, mono=False)
        audio_chunk = audio.get_segment(start, end, sample_rate, normalize=self.normalize)

        return audio_chunk[0]


    def __getitem__(self, index):

        id, start_time, end_time, vad, vap, inverse_vap = super().__getitem__(index)
        audio_chunk = self.get_audio_chunk(id, start_time, end_time, sample_rate=self.sr)
        return id, start_time, end_time, vad, vap, inverse_vap, audio_chunk




class ValidationAudioDataset(Dataset):
    """ returns everything in order with overlapping windows to enable validation of a file """

    def __init__(self, audio_file: str, transcript_file: str, sr: int, feature_sr: int, window_size: int, step_size: int, mode: str):

        super().__init__()
        
        self.audio_file = audio_file
        self.transcript_file = transcript_file

        self.id = os.path.basename(audio_file).split('.')[0]

        self.sr = sr
        self.feature_sr = feature_sr
        self.window_size = window_size
        self.step_size = step_size

        self.mode = mode
        self.audio_normalize = True

        self.load_audio()
        # self.load_transcript()

    def load_audio(self):
        audio, _= AudioManager.load_waveform(self.audio_file, self.sr, normalize=self.audio_normalize)
        self.track_size = audio.shape
        size, step = int(self.sr*self.window_size), int(self.sr*self.step_size)

        self.audio_unbathed = audio
        audio = audio.unfold(0, size, step) # auto know the number of windows
        audio = einops.rearrange(audio, "b c n -> b n c")


        self.audio = audio      
    
    def load_transcript(self):
        
        vad_list = vads_from_transcript(self.transcript_file, samples=False)
        vad = vad_list_to_one_hot(vad_list, [2, int(self.feature_sr*self.track_size[0]/self.sr)], samples=False, sr=self.feature_sr)
        vap = training_labels_projection(vad, mode=self.mode)
        
        size, step = int(self.feature_sr*self.window_size), int(self.feature_sr*self.step_size)

        self.vad_unbatched, self.vap_unbatched = vad, vap

        vad, vap = vad.unfold(0, size, step), vap.unfold(0, size, step)

        self.vad, self.vap = vad, vap


    def __getitem__(self, index):
        start_time, end_time = index * self.step_size, index * self.step_size + self.window_size
        try:
            return {"id": [self.id], "audio_chunk": self.audio[index, :], "start_time": start_time, "end_time": end_time, "vad": self.vad[index, :], "vap": self.vap[index, :]}
        except:
            return {"id": [self.id], "audio_chunk": self.audio[index, :], "start_time": start_time, "end_time": end_time}

    
    def __len__(self):
        return self.audio.shape[0] 


class AudioVisualDataset(AudioDataset):


    def __init__(self, video_directory, video_format, channelmaps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_directory = video_directory
        self.video_format = video_format
        self.channel_hash = _read_pickle_compat(channelmaps)
        # self.fps=30

    def __getitem__(self, idx):

        id, start, end, vad, vap, inverse_vap, audio_chunk = super().__getitem__(idx)

        # load the corresponding video file
        frames_lr = None
        ch = self.channel_hash[self.channel_hash['id']==id]
    
        for lr in ("L", "R"):

            video_id = ch[lr].item()
             
            video_file = os.path.join(self.video_directory, id + '--' + video_id + self.video_format) # candor pattern
            df = _read_pickle_compat(video_file)
            frames = df.iloc[int(start*self.fps):int(end*self.fps), :]
            frames = torch.Tensor(frames.values)
            if frames.shape[0]==0:
                print(f'warning frame.shape[0]==0: {id}, {frames.shape}')

            if frames_lr is None:
                frames_lr = frames
            else:
                frames_lr = torch.stack((frames_lr, frames), dim=-1)

            if lr == "L":
                channel_id_l = video_id

            if lr == "R":
                channel_id_r = video_id

        return id, start, end, vad, vap, inverse_vap, audio_chunk, channel_id_l, channel_id_r, frames_lr


class AVCocktailDataset(ChunkedDataset):
    """
    Training dataset for cocktail party scenario with updated video loading logic.
    Similar to AudioVisualDataset but with updated file path handling from ValidationAudioVisualDataset.
    """

    def __init__(self, video_directory, channelmaps, wavdir, dev: bool, *args, **kwargs):
        self.wavdir = wavdir
        self.normalize = True
        super().__init__(*args, **kwargs)
        self.video_directory = video_directory
        self.channel_hash = _read_pickle_compat(channelmaps)
        # self.fps = 30  # Updated FPS
        # self.is_central = "central" in channelmaps
        self.dev = dev
        # print(f'fps: {self.fps}')

    
    def get_audio_chunk(self, id, start, end, sample_rate):
        # locate the audio file 
        session, audioname = id.split("**")
        wavfile = os.path.join(self.wavdir, session, "stereo_audios", f"{audioname}.wav")
        # print(f'wavfile: {wavfile}')
        
        # audio segment 
        audio = AudioManager(wavfile, mono=False)
        audio_chunk = audio.get_segment(start, end, sample_rate, normalize=self.normalize)

        return audio_chunk[0]


    def __getitem__(self, idx):

        id, start, end, vad, vap, inverse_vap = super().__getitem__(idx)
        session, audioname = id.split("**")
        hash_id = session + "-" + audioname.replace("-30fps", "")
        if not self.dev:
            hash_id = "central-" + hash_id
        # print(f"hash_id: {hash_id}, session: {session}, audioname: {audioname}")

        audio_chunk = self.get_audio_chunk(id, start, end, sample_rate=self.sr)



        # Load the corresponding video file
        frames_lr = None
        ch = self.channel_hash[self.channel_hash['id'] == hash_id]
        # print(f'hash_id: {hash_id}, ch: {ch}')
    
        for lr in ("L", "R"):
            # print(f'ch[lr]: {ch[lr]}')

            video_id = ch[lr].item()
            
            # Construct video file path based on the new structure
            # if session:
            spk_idx = video_id
            if not self.dev:
                video_file = os.path.join(self.video_directory, session, "speakers", spk_idx, "central-pkl_fps30", video_id + '.pkl')
            else:
                video_file = os.path.join(self.video_directory, session, "speakers", spk_idx.split("-")[0], "pkl_fps30", video_id + '.pkl')
            # else:
            #     # Fallback to original path structure
            #     video_file = os.path.join(self.video_directory, video_id + '.pkl')

            df = _read_pickle_compat(video_file)
            frames = df.iloc[int(start * self.fps):int(end * self.fps), :]
            frames = torch.Tensor(frames.values)

            # print(frames.shape)
            
            if frames.shape[0] == 0:
                print(f'warning frame.shape[0]==0: {id}, {frames.shape}')

            if frames_lr is None:
                frames_lr = frames
            else:
                # Handle frame length mismatch
                if frames_lr.shape[0] != frames.shape[0]:
                    min_len = min(frames_lr.shape[0], frames.shape[0])
                    frames_lr = frames_lr[:min_len, :]
                    frames = frames[:min_len, :]
                frames_lr = torch.stack((frames_lr, frames), dim=-1)

            if lr == "L":
                channel_id_l = video_id

            if lr == "R":
                channel_id_r = video_id

        return id, start, end, vad, vap, inverse_vap, audio_chunk, channel_id_l, channel_id_r, frames_lr

        
class AVCocktailValidationAudioVisualDataset(ValidationAudioDataset):

    
    def __init__(self, video_pkl_dir, channelmap, **kwargs):
        super().__init__(**kwargs)
        self.channelmap = _read_pickle_compat(channelmap) # channelmap of the entire corpus
        self.video_pkl_dir = video_pkl_dir
        # self.id = os.path.basename(self.audio_file).split('.')[0]
        self.id = re.sub(r'.*(session_\d+).*/(spk_\d+-spk_\d+)-30fps\.wav', r'\1-\2', self.audio_file) # TODO generalise this
        # print(f'id: {self.id}') # session id, for example session_22-speaker1-speaker2
        if "central" and "train" in channelmap:
            self.id = "central-" + self.id

        ch = self.channelmap[self.channelmap.id==self.id]
        self.channelmap = {"L": ch["L"].item(), "R": ch["R"].item()}
        self.fps=30


    def __getitem__(self, index):
        ret = super().__getitem__(index)

        start = ret['start_time']
        end = ret['end_time']

        frames_lr = None
        for video_id in [self.channelmap["L"], self.channelmap["R"]]:
            spk_idx = video_id
            
            match = re.search(r'session_\d+', self.id)
            session_name = match.group(0) if match else None
            if "central" in self.id:
                video_file = os.path.join(self.video_pkl_dir, session_name, "speakers", spk_idx, "central-pkl_fps30", video_id + '.pkl')
            else:
                video_file = os.path.join(self.video_pkl_dir, session_name, "speakers", spk_idx.split("-")[0], "annotated-pkl_fps30", video_id + '.pkl') # TODO generalise this

            df = _read_pickle_compat(video_file)

            frames = df.iloc[int(start*self.fps):int(end*self.fps), :]
            frames = torch.Tensor(frames.values)
            if frames.shape[0]==0:
                print(f'warning frame.shape[0]==0: {ret["id"]}, {frames.shape}')
            if frames_lr is None:
                frames_lr = frames
            else:
                if frames_lr.shape[0] != frames.shape[0]:
                    min_len = min(frames_lr.shape[0], frames.shape[0])
                    frames_lr = frames_lr[:min_len, :]
                    frames = frames[:min_len, :]
                frames_lr = torch.stack((frames_lr, frames), dim=-1)

        # time last
        ret['channel_ids'] = np.array([self.channelmap["L"], self.channelmap["R"]])
        ret['frames'] = frames_lr

        return ret
    
    def __len__(self):
        return super().__len__()


class ValidationAudioVisualDataset(ValidationAudioDataset):

    
    def __init__(self, video_pkl_dir, channelmap, **kwargs):
        super().__init__(**kwargs)
        
        self.channelmap = _read_pickle_compat(channelmap)
        self.video_pkl_dir = video_pkl_dir
        self.id = os.path.basename(self.audio_file).split('.')[0]
        ch = self.channelmap[self.channelmap.id==self.id]
        self.channelmap = {"L": ch["L"].item(), "R": ch["R"].item()}
        self.fps=30


    def __getitem__(self, index):
        ret = super().__getitem__(index)

        start = ret['start_time']
        end = ret['end_time']

        frames_lr = None
        for video_id in [self.channelmap["L"], self.channelmap["R"]]:
            
            video_file = os.path.join(self.video_pkl_dir, ret['id'][0] + '--' + video_id + '.pkl')

            df = _read_pickle_compat(video_file)
            frames = df.iloc[int(start*self.fps):int(end*self.fps), :]
            frames = torch.Tensor(frames.values)
            if frames.shape[0]==0:
                print(ret['id'], frames.shape)
            if frames_lr is None:
                frames_lr = frames
            else:
                frames_lr = torch.stack((frames_lr, frames), dim=-1)

        # time last
        ret['channel_ids'] = np.array([self.channelmap["L"], self.channelmap["R"]])
        ret['frames'] = frames_lr

        return ret
    
    def __len__(self):
        return super().__len__()


