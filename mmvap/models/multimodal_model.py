
import einops
from mmvap.models.model import GPT, GPTStereo, Combinator
from mmvap.models.visualisation_mixin import VisualisationMixin
from mmvap.audio_encoders.encoders import StereoEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader


class Early_VA_Combiner(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # audio self attention
        self.self_attn_audio = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # video self attn
        self.self_attn_video = GPT(
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # project video to audio dim
        self.video_proj_ln = nn.LayerNorm(cfg['multimodal_model_cfg']['audio']['d_model'])
        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'], bias=False)

        # cross attention of video with audio
        self.av_cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_av']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_av']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_av']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )   

        # Activation
        self.activation = getattr(nn, cfg['multimodal_model_cfg']['ffn_activation'])()

    def forward(self, x_l, x_r, v_l, v_r):
        # print(f'x_l shape: {x_l.shape}')
        # print(f'v_l shape: {v_l.shape}')
        # print(f'x_r shape: {x_r.shape}')
        # print(f'v_r shape: {v_r.shape}')
        
        # audio self attention
        x_l = self.self_attn_audio(x_l, return_attention=False)
        x_r = self.self_attn_audio(x_r, return_attention=False)

        # video 
        v_l = self.self_attn_video(v_l, return_attention=False)
        v_r = self.self_attn_video(v_r, return_attention=False)

        v_l['x'] = self.activation(self.video_proj_ln(self.video_proj(v_l['x'])))
        v_r['x'] = self.activation(self.video_proj_ln(self.video_proj(v_r['x'])))

        # cross attention
        out_l = self.av_cross_attn(x_l["x"], v_l["x"], return_attention=False)
        out_r = self.av_cross_attn(x_r["x"], v_r["x"], return_attention=False)

        return out_l, out_r


class Late_VA_Combiner(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # audio self attention
        self.self_attn_audio = GPT(
            dim=cfg['multimodal_model_cfg']['audio']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['audio']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # video self attn
        self.self_attn_video = GPT(
            dim=cfg['multimodal_model_cfg']['video']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['video']['n_layers_self'],
            num_heads=cfg['multimodal_model_cfg']['video']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # project video to audio dim
        self.video_proj_ln = nn.LayerNorm(cfg['multimodal_model_cfg']['audio']['d_model'])
        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'])

        # cross attention of video with audio
        self.cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_av']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_av']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_av']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # Activation
        self.activation = getattr(nn, cfg['multimodal_model_cfg']['ffn_activation'])()

    def forward(self, x_l, x_r, v_l, v_r):
        
        # audio self attention
        x_l = self.self_attn_audio(x_l, return_attention=False)
        x_r = self.self_attn_audio(x_r, return_attention=False)

        # video 
        v_l = self.self_attn_video(v_l, return_attention=False)
        v_r = self.self_attn_video(v_r, return_attention=False)

        v_l['x'] = self.activation(self.video_proj_ln(self.video_proj(v_l['x'])))
        v_r['x'] = self.activation(self.video_proj_ln(self.video_proj(v_r['x'])))

        # cross attention
        out_a = self.cross_attn(x_l["x"], x_r["x"], return_attention=False)
        out_v = self.cross_attn(v_l["x"], v_r["x"], return_attention=False)

        return out_a, out_v


class EarlyVAFusion(VisualisationMixin, nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # VA fusion
        self.va_fusion = Early_VA_Combiner(self.cfg)

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")
        # self.audio_encoder = StereoEncoderSpectrogram().to("cuda")
        # self.audio_encoder = StereoEncoderWhisper().to("cuda")

        # cross attention
        self.cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_speakers']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_speakers']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob'],
        )

        # for upsampling the N dimension of the video
        scale_factor = cfg['multimodal_model_cfg']['audio']['sequence_len'] / cfg['multimodal_model_cfg']['video']['sequence_len']
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'])

        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=1)
        self.vap_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=cfg['model_cfg']['d_out'])

        self._init_viz_state()

        # for ablations
        self.ablation_cfg = {
            "null_audio": False,
            "null_video": False,
            "audio_noise_snr": None,     # e.g. 20, 10, 0
            "audio_shuffle": False,      # shuffle non-silent segments
            "audio_lag_ms": 0,           # positive = audio delayed
            "freeze_video": False,       # static face
            "null_chunks_audio": False,
            "null_chunks_video": False
        }

    def add_noise_snr(self, audio, snr_db):
        print(f'!!!! In EarlyVAFusion add_noise_snr: adding noise to audio with SNR of {snr_db} dB')
        # audio: [B, T, F, C]
        signal_power = audio.pow(2).mean(dim=(1,2,3), keepdim=True)
        snr = 10 ** (snr_db / 10)
        noise_power = signal_power / snr
        noise = torch.randn_like(audio) * noise_power.sqrt()
        return audio + noise
    
    def apply_audio_lag(self, audio, lag_frames):
        print(f'!!!! In EarlyVAFusion apply_audio_lag: applying audio lag of {lag_frames} frames')
        # positive lag = audio delayed
        if lag_frames == 0:
            return audio
        B, T, F, C = audio.shape
        out = torch.zeros_like(audio)
        if lag_frames > 0:
            out[:, lag_frames:] = audio[:, :-lag_frames]
        else:
            out[:, :lag_frames] = audio[:, -lag_frames:]
        return out

    def freeze_video(self, frames):
        print('!!!! In EarlyVAFusion freeze_video: freezing video input to first frame')
        return frames[:, :1].repeat(1, frames.size(1), 1, 1)
    
    def null_audio(self, audio):
        print('!!!! In EarlyVAFusion null_audio: nullifying audio input')
        return torch.zeros_like(audio).to("cuda")

    def null_video(self, frames):
        print('!!!! In EarlyVAFusion null_video: nullifying video input')
        return torch.ones_like(frames)*(-1e-4)
    
    def null_chunks(self, x, modality, proportion, null_length):
        """
        proportion: how many % of the total audio/video
        null_length: length of each null chunk (in frames/samples)
        """ 
        print(f'!!!! In EarlyVAFusion null_chunks: nullifying random chunks of {modality} input')
        B, T, F, C = x.shape
        total_length = T
        num_null_chunks = int((total_length / null_length) * proportion)
        for b in range(B):
            null_starts = torch.randperm(total_length - null_length)[:num_null_chunks]
            for start in null_starts:
                x[b, start:start+null_length] = 0 if modality == 'audio' else -1e-4
        return x



    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        return x
        


    def forward(self, batch):
        # Calculate current chunk's time range
        chunk_start_time = self.chunk_count * self.step_size_sec
        chunk_end_time = chunk_start_time + self.window_size_sec
        self.chunk_count += 1

        with torch.no_grad():
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            ### START ABLATIONS
            audio = self.null_chunks(audio, modality='audio', proportion=0.2, null_length=100) if self.ablation_cfg['null_chunks_audio'] else audio
            ### END ABLATIONS

            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            video = batch['frames'].to(device='cuda')
            video = torch.nan_to_num(video)
            ### START ABLATIONS
            video = self.null_chunks(video, modality='video', proportion=0.2, null_length=60) if self.ablation_cfg['null_chunks_video'] else video
            ### END ABLATIONS
            video = self.upsample(video).to(device='cuda')
            v_l, v_r = video[:,:,:,0].to("cuda"), video[:,:,:,1].to("cuda")
            # print(f'in EarlyVAFusion forward: v_l shape {v_l.shape}, v_r shape {v_r.shape}')

        # fuse A and V from each speaker independently
        s_l, s_r = self.va_fusion(x_l, x_r, v_l, v_r)
        # print(f'in EarlyVAFusion forward after va_fusion: s_l shape {s_l["x"].shape}, s_r shape {s_r["x"].shape}')

        out = self.cross_attn(s_l['x'], s_r['x'])

        self._maybe_accumulate_embeddings(out['x'], chunk_start_time)

        # output
        v1 = self.vad_classifier(out["x1"])
        v2 = self.vad_classifier(out["x2"])

        vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap

    def set_event_timestamps(self, event_timestamps: list, window_sec: float = 0.5):
        super().set_event_timestamps(event_timestamps, window_sec)


class LateVAFusion(nn.Module):


    def __init__(self, cfg):
        super().__init__()
        
        self.cfg=cfg
        
        # VA fusion
        self.fusion = Early_VA_Combiner(self.cfg)  # TODO: should this be Late_VA_Combiner?

        # encoder
        self.audio_encoder = StereoEncoder().to("cuda")

        # cross attention
        self.cross_attn = GPTStereo(
            dim=cfg['multimodal_model_cfg']['fuse_speakers']['d_model'],
            dff_k=cfg['multimodal_model_cfg']['dff_k'],
            num_layers=cfg['multimodal_model_cfg']['fuse_speakers']['n_layers_cross'],
            num_heads=cfg['multimodal_model_cfg']['fuse_speakers']['n_heads'],
            activation=cfg['multimodal_model_cfg']['ffn_activation'],
            dropout=cfg['multimodal_model_cfg']['dropout_prob']
        )

        # for upsampling the N dimension of the video
        scale_factor = cfg['multimodal_model_cfg']['audio']['sequence_len'] / cfg['multimodal_model_cfg']['video']['sequence_len']
        scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)

        self.video_proj = nn.Linear(in_features=cfg['multimodal_model_cfg']['video']['d_model'], out_features=cfg['multimodal_model_cfg']['audio']['d_model'], bias=False)

        self.vad_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=2)
        self.vap_classifier = nn.Linear(in_features=cfg['model_cfg']['d_model'], out_features=cfg['model_cfg']['d_out'])
        
        self.combine_speakers = Combinator(cfg['model_cfg']['d_model'])
    
    def upsample(self, x):
        B, N, D, C = x.shape
        x=einops.rearrange(x,"b n d c -> (b c) d n")
        x=self.upsampler(x)
        x=einops.rearrange(x,"(b c) d n -> b n d c", b=B, c=C)
        return x


    def forward(self, batch):

        with torch.no_grad():

            # get the L and R channels of the audio
            audio = self.audio_encoder(batch['audio_chunk'].to("cuda"))
            x_l, x_r = audio[:,:,:,0].to("cuda"), audio[:,:,:,1].to("cuda")

            # upsample the video
            batch['frames']=torch.nan_to_num(batch['frames'])
            video = self.upsample(batch['frames']).to(device='cuda')
            v_l, v_r = video[:,:,:,0].to("cuda"), video[:,:,:,1].to("cuda")

        # fuse A and V from each speaker independently
        v, a = self.fusion(x_l, x_r, v_l, v_r)

        out = self.cross_attn(v['x'], a['x'])

        x = self.combine_speakers(out["x1"], out["x2"])

        # output
        vad = self.vad_classifier(x)

        # vad = torch.cat((v1, v2), dim=-1)
        vap = self.vap_classifier(out['x'])

        return vad, vap


