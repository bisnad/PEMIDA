import os
import math
import numpy as np
import pickle
from collections import OrderedDict
import threading
import queue
import time
import sounddevice as sd
import torch
import torchaudio
from torch import nn
from torchaudio.functional import highpass_biquad

# osc specific imports
from pythonosc import dispatcher
from pythonosc import osc_server

# vocos specific
from vocos import Vocos

# mocap specific
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, q_conj_np, qmul_np, slerp
from common.pose_renderer import PoseRenderer

# -------------------------------------------------------
# Settings
# -------------------------------------------------------

# compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')


# mocap settings

mocap_data_file = "E:/data/mocap/Eleni/Solos/ZHdK_04.12.2025/xsens2osc_30hz/Eline_Session-002.pkl"

# manual settings are only necessary if motion data is loaded from pkl file
mocap_joint_count = 23 # will be overwritten for fbx and bvh files
mocap_joint_dim = 4 # will be overwritten for fbx and bvh files
mocap_root_joint_index = 0 # will be overwritten for fbx and bvh files
mocap_pose_dim = mocap_joint_count * mocap_joint_dim

mocap_pos_scale = 1.0
mocap_fps = 30 # 60

load_mocap_stats = True
mocap_mean_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/mocap_mean.pt"
mocap_std_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/mocap_std.pt"

# audio settings

print(sd.query_devices())
audio_data_file = "E:/data/audio/Eleni/48khz/4_5870821179501060412.wav"
#audio_data_file = "E:/data/audio/Tim/48khz/SajuHariPlacePrizeEntry2010.wav"
audio_output_device = 8
audio_sample_rate = 48000
audio_channels = 1

audio_waveform_length_per_mocap_frame = int(1.0 / mocap_fps * audio_sample_rate)
audio_window_offset = audio_waveform_length_per_mocap_frame
audio_window_length = None
play_buffer_size = None
playback_latency = 1.0
hpf_cutoff_hz = 15.0

load_audio_stats = True
audio_mean_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/audio_mean.pt"
audio_std_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/audio_std.pt"


# OSC parameters
osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007


# Vocos pre-trained model
vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"
audio_vocos_waveform_length = 44800

# Motion VAE Model Settings

motion_vae_mocap_length = 8
motion_vae_latent_dim = 128 # 128
motion_vae_rnn_layer_count = 2
motion_vae_rnn_layer_size = 512
motion_vae_dense_layer_sizes = [ 512 ]

motion_vae_encoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2osc_mocap8_ld128_kld0.1_cl0.1_offset1/weights/motion_encoder_weights_epoch_400"
motion_vae_decoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2osc_mocap8_ld128_kld0.1_cl0.1_offset1/weights/motion_decoder_weights_epoch_400"

# Audio VAE Model Settings

audio_vae_latent_dim = 128 # 128
audio_vae_conv_channel_counts = [ 16, 32, 64, 128 ]
audio_vae_conv_kernel_size = (5, 3)
audio_vae_dense_layer_sizes = [ 512 ]

audio_vae_encoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2osc_mocap8_ld128_kld0.1_cl0.1_offset1/weights/audio_encoder_weights_epoch_400"
audio_vae_decoder_weights_file = "../../../VAE/Multimodal_VAE/Multimodal_VAE_Vocos_CL/Training/results_Eleni_Take2osc_mocap8_ld128_kld0.1_cl0.1_offset1/weights/audio_decoder_weights_epoch_400"

audio_vae_waveform_length = int(motion_vae_mocap_length / mocap_fps * audio_sample_rate)

audio_window_length = audio_vae_waveform_length
play_buffer_size = audio_window_length * 4 # * 16
max_fifo_queue_length = play_buffer_size // audio_window_offset * 2 # 32

# FIFO Queue Settings
max_fifo_queue_length = play_buffer_size // audio_window_offset * 2 # 32

print("play_buffer_size ", play_buffer_size)
print("audio_window_offset ", audio_window_offset)
print("max_fifo_queue_length ", max_fifo_queue_length)

audio_vae_mel_count = None # automatically calculated
audio_waveform_start_offset = audio_vocos_waveform_length - audio_vae_waveform_length
mocap_frame_start_offset = int(audio_waveform_start_offset / audio_sample_rate * mocap_fps)

# Transformer Model Settings

transformer_layer_count = 6
transformer_head_count = 8
transformer_embed_dim = 256
transformer_dropout = 0.1   

transformer_weights_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/weights/transformer_weights_epoch_200"

motion_latents_mean_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/motion_latents_mean.pt"
motion_latents_std_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/motion_latents_std.pt"

audio_latents_mean_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/audio_latents_mean.pt"
audio_latents_std_file = "../Training/results_Eleni_Take2osc_30hz_transformer_multimodalvae_mocap8_ld128_kld0.1_cl0.1_offset1/stats/audio_latents_std.pt"

transformer_motion_latent_input_length = 60 # 60
transformer_motion_latent_output_length = 10
transformer_motion_mocap_input_length = transformer_motion_latent_input_length
transformer_motion_mocap_output_length = transformer_motion_latent_output_length 
transformer_motion_mocap_prepend_length = motion_vae_mocap_length // 2
transformer_motion_mocap_append_length = motion_vae_mocap_length // 2
transformer_motion_mocap_total_length = transformer_motion_mocap_input_length + transformer_motion_mocap_output_length + transformer_motion_mocap_prepend_length + transformer_motion_mocap_append_length

transformer_audio_latent_input_length = transformer_motion_latent_input_length
transformer_audio_latent_output_length = transformer_motion_latent_output_length
transformer_audio_waveform_input_length = transformer_motion_mocap_input_length * audio_waveform_length_per_mocap_frame
transformer_audio_waveform_output_length = transformer_motion_mocap_output_length * audio_waveform_length_per_mocap_frame
transformer_audio_waveform_prepend_length = audio_waveform_length_per_mocap_frame + audio_vae_waveform_length // 2 + (audio_vocos_waveform_length - audio_vae_waveform_length)
transformer_audio_waveform_append_length =  audio_vae_waveform_length // 2
transformer_audio_waveform_total_length = transformer_audio_waveform_input_length + transformer_audio_waveform_output_length + transformer_audio_waveform_prepend_length + transformer_audio_waveform_append_length

# the input length for the transformer is the same for mocap and audio
# since the transformer_mocap_input_length equals the number of latents for both modalities
pos_encoding_max_length = transformer_motion_latent_input_length

# -------------------------------------------------------
# Data Loading Functions
# -------------------------------------------------------

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        pkl_data = pickle.load(f)
        
    return pkl_data

def pkl_to_mocap(pkl_data):
    
    mocap_data = {}
    mocap_data["motion"] = {}
    
    # unique osc message addresses
    sensor_ids = pkl_data["sensor_ids"]
    sensor_values = pkl_data["sensor_values"]
    
    # get unique sensor ids
    unique_sensor_ids = list(set(sensor_ids))
    #print("unique_sensor_ids ", unique_sensor_ids)
    
    # get rid of all unique sensor ids that don't correspond to skeleton 0
    unique_sensor_ids = [ unique_sensor_id for unique_sensor_id in unique_sensor_ids if "/mocap/0" in unique_sensor_id ]
    #print("unique_sensor_ids ", unique_sensor_ids)
    
    # get indices of unique sensor ids into sensor values list
    unique_sensor_indices = {}
    for unique_sensor_id in unique_sensor_ids:
        unique_indices = [i for i, s in enumerate(sensor_ids) if s == unique_sensor_id]
        unique_sensor_indices[unique_sensor_id] = unique_indices
    
    # create motion data
    for unique_sensor_id in unique_sensor_ids:
        
        indices = unique_sensor_indices[unique_sensor_id]
        values = [sensor_values[i] for i in indices]
        
        mocap_data["motion"][unique_sensor_id.replace("/mocap/0/joint/", "")] = np.array(values)


    return mocap_data

def remove_root_position(mocap_data, root_joint_index):
    
    pos_local = mocap_data["motion"]["pos_local"]
    pos_world = mocap_data["motion"]["pos_world"]
    
    #print("pos_local s ", pos_local.shape)
    #print("pos_world s ", pos_world.shape)
    
    pos_local = pos_local.reshape(-1, mocap_joint_count, 3)
    pos_world = pos_world.reshape(-1, mocap_joint_count, 3)
    
    #print("pos_local 2 s ", pos_local.shape)
    #print("pos_world 2 s ", pos_world.shape)
    
    pos_local[:, root_joint_index, :] = 0.0
    pos_world_root = np.expand_dims(pos_world[:, root_joint_index, :], 1)
    
    #print("pos_world_root s ", pos_world_root.shape)
    
    pos_world -= pos_world_root
    
    pos_local = pos_local.reshape(-1, mocap_joint_count * 3)
    pos_world = pos_world.reshape(-1, mocap_joint_count * 3)
    
    mocap_data["motion"]["pos_local"] = pos_local
    mocap_data["motion"]["pos_world"] = pos_world

def quat_relative(q1, q2, normalize_inputs=True):
    """
    Relative rotation that takes q1 to q2, batched.
    q1, q2: shape (N, 4) (or any broadcastable (..., 4))
    Returns q_rel: shape (N, 4), with q_rel * q1 = q2.
    """
    if normalize_inputs:
        q1 = qnormalize_np(q1)
        q2 = qnormalize_np(q2)

    q1_inv = q_conj_np(q1)   # for unit quats, inverse == conjugate
    return qmul_np(q2, q1_inv)

def remove_root_rotation(mocap_data, root_joint_index):
    rot_local = mocap_data["motion"]["rot_local"]
    rot_world = mocap_data["motion"]["rot_world"]

    rot_local = rot_local.reshape(-1, mocap_joint_count, 4)
    rot_world = rot_world.reshape(-1, mocap_joint_count, 4)

    # set root local rotation to identity
    rot_local[:, root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))

    # extract root world rotation (F,4)
    rot_world_root = rot_world[:, root_joint_index, :]

    # inverse root rotation (F,4)
    rot_world_root_inv = q_conj_np(qnormalize_np(rot_world_root))

    # broadcast to (F,J,4)
    rot_world_root_inv = np.expand_dims(rot_world_root_inv, 1)
    rot_world_root_inv = np.repeat(rot_world_root_inv, repeats=mocap_joint_count, axis=1)

    # apply: q_new = q_root_inv * q_world
    rot_world = qmul_np(rot_world_root_inv, rot_world)

    rot_local = rot_local.reshape(-1, mocap_joint_count * 4)
    rot_world = rot_world.reshape(-1, mocap_joint_count * 4)

    mocap_data["motion"]["rot_local"] = rot_local
    mocap_data["motion"]["rot_world"] = rot_world

def load_mocap(file_path, scale):
    bvh_tools = bvh.BVH_Tools()
    fbx_tools = fbx.FBX_Tools()
    mocap_tools = mocap.Mocap_Tools()
    
    bvh_data = None
    fbx_data = None
    pkl_data = None
    
    if file_path.endswith(".bvh") or file_path.endswith(".BVH"):
        bvh_data = bvh_tools.load(file_path)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif file_path.endswith(".fbx") or file_path.endswith(".FBX"):
        fbx_data = fbx_tools.load(file_path)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only  
    elif file_path.endswith(".pkl"):
        pkl_data = load_pkl(file_path)
        mocap_data = pkl_to_mocap(pkl_data)
    
    if bvh_data is not None or fbx_data is not None:
        
        mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
        mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
        print("pos_local shape", mocap_data["motion"]["pos_local"].shape)
        #print("rot_local_euler shape", mocap_data["motion"]["rot_local_euler"].shape)

        # set x and z offset of root joint to zero
        mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
        mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
        
    elif pkl_data is not None:
        
        mocap_data["motion"]["pos_local"] *= mocap_pos_scale
        mocap_data["motion"]["pos_world"] *= mocap_pos_scale

    if bvh_data is not None:
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif fbx_data is not None:
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_local"][:, mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
    elif pkl_data is not None:
        remove_root_position(mocap_data, mocap_root_joint_index)
        remove_root_rotation(mocap_data, mocap_root_joint_index)

    return mocap_data

def load_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    return waveform, sr

# -------------------------------------------------------
# Model Definitions
# -------------------------------------------------------

class MotionEncoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(MotionEncoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create final dense layers
            
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x[:, -1, :] # only last time step 
        
        #print("x 3 ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x 3 ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", log_var.shape)
    
        return mu, std
    
    def reparameterize(self, mu, std):
            z = mu + std*torch.randn_like(std)
            return z

class MotionDecoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(MotionDecoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create rnn layers
        rnn_layers = []

        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # final output dense layer
        final_layers = []
        
        final_layers.append(("decoder_dense_{}".format(dense_layer_count), nn.Linear(self.rnn_layer_size, self.pose_dim)))
        
        self.final_layers = nn.Sequential(OrderedDict(final_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        x = self.dense_layers(x)
        #print("x 2 ", x.size())
        
        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, self.sequence_length, 1)
        #print("x 3 ", x.size())
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        #print("x 4 ", x.size())
        
        # final time distributed dense layer
        x_reshaped = x.contiguous().view(-1, self.rnn_layer_size)  # (batch_size * sequence, input_size)
        #print("x 5 ", x_reshaped.size())
        
        yhat = self.final_layers(x_reshaped)
        #print("yhat 1 ", yhat.size())
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)
        #print("yhat 2 ", yhat.size())

        return yhat

class AudioEncoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        #print("conv_kernel_size ", conv_kernel_size)
        #print("stride ", stride)
        
        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_input_size ", dense_layer_input_size)
        #print("self.dense_layer_sizes[0] ", self.dense_layer_sizes[0])
        
        self.dense_layers.append(nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        # create final dense layers
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)


    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x1 s ", x.shape)
        
        x = self.flatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x3 s ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", std.shape)

        return mu, std
    
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z

class AudioDecoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)
        
        #print("stride ", stride)
                
        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))
        
        #print("last_conv_layer_size_x ", last_conv_layer_size_x)
        #print("last_conv_layer_size_y ", last_conv_layer_size_y)
        
        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]
        
        #print("preflattened_size ", preflattened_size)
        
        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y
        
        #print("dense_layer_output_size ", dense_layer_output_size)

        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):
        
        #print("x0 s ", x.shape)
        
        for lI, layer in enumerate(self.dense_layers):
            
            #print("dense layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("dense layer ", lI, " x out ", x.shape)
            
        #print("x1 s ", x.shape)
        
        x = self.unflatten(x)
        
        #print("x2 s ", x.shape)

        for lI, layer in enumerate(self.conv_layers):
            
            #print("conv layer ", lI, " x in ", x.shape)
            
            x = layer(x)
            
            #print("conv layer ", lI, " x out ", x.shape)
    
        #print("x3 s ", x.shape)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # for batch-first: [1, max_len, dim_model]
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        
        #print("token_embedding s ", token_embedding.shape)
        #print("pos_encoding s ", self.pos_encoding.shape)
        
        # token_embedding: [batch_size, seq_len, dim_model]
        seq_len = token_embedding.size(1)
        # broadcast over batch dimension
        pe = self.pos_encoding[:, :seq_len, :]
        
        return self.dropout(token_embedding + pe)

class Transformer(nn.Module):

    # Constructor
    def __init__(
        self,
        mocap_dim,
        audio_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
        pos_encoding_max_length
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # LAYERS
        self.mocap2embed = nn.Linear(mocap_dim, embed_dim) # map mocap data to embedding
        self.audio2embed = nn.Linear(audio_dim, embed_dim) # map audio data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = num_decoder_layers)
        
        self.embed2audio = nn.Linear(embed_dim, audio_dim) # map embedding to audio data
        
    def get_src_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.ones(size, size)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
       
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
        
       
    def forward(self, mocap_data, audio_data):
        
        #print("forward")
        
        #print("data s ", data.shape)

        src_mask = self.get_src_mask(mocap_data.shape[1]).to(mocap_data.device)
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)
        
        mocap_embedded = self.mocap2embed(mocap_data) * math.sqrt(self.embed_dim)
        mocap_embedded = self.positional_encoder(mocap_embedded)
        
        audio_embedded = self.audio2embed(audio_data) * math.sqrt(self.embed_dim)
        audio_embedded = self.positional_encoder(audio_embedded)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        encoder_out = self.encoder(mocap_embedded, mask=src_mask)
        decoder_out = self.decoder(audio_embedded, encoder_out, tgt_mask =tgt_mask)
        
        out = self.embed2audio(decoder_out)
        
        return out
    
# -------------------------------------------------------
# Motion Helper Functions
# -------------------------------------------------------

@torch.no_grad()
def encode_motion(motion, motion_encoder_model):
    """Encode motion to normalised motion latents."""
    
    motion_mocap_vae_excerpt_i = motion.to(device)

    # normalise motion
    motion_encoder_input_i = (motion_mocap_vae_excerpt_i - mocap_mean) / (mocap_std + 1e-8)

    # calculate motion latent
    motion_encoder_input_i = motion_encoder_input_i.unsqueeze(0)
    motion_encoder_output_mu_i, motion_encoder_output_std_i = motion_encoder(motion_encoder_input_i)
    mu_i = motion_encoder_output_mu_i
    std_i = torch.nn.functional.softplus(motion_encoder_output_std_i) + 1e-8
    motion_latent_i = motion_encoder.reparameterize(mu_i, std_i)

    # normalise motion latent
    motion_latent_i = (motion_latent_i - motion_latents_mean) / (motion_latents_std + 1e-8)

    return motion_latent_i

# -------------------------------------------------------
# Audio Helper Functions
# -------------------------------------------------------

@torch.no_grad()
def encode_audio(audio_waveform, vocos_model, audio_encoder_model):
    """Encode audio waveform to normalised audio latents."""
    
    #print("audio s ", audio.shape)

    audio_waveform_vocos_excerpt_i = audio_waveform.unsqueeze(0).to(device)

    # calculate mels
    audio_mels_vocos_excerpt_i = vocos_model.feature_extractor(audio_waveform_vocos_excerpt_i)

    # get the last audio mels required for the audio vae
    audio_mels_vae_excerpt_i = audio_mels_vocos_excerpt_i[:, :, -audio_vae_mel_count:]#

    # normalise audio mels
    audio_encoder_input_i = (audio_mels_vae_excerpt_i - audio_mean) / (audio_std + 1e-8)

    # calculate audio latent
    audio_encoder_input_i = audio_encoder_input_i.unsqueeze(1)
    audio_encoder_output_mu_i, audio_encoder_output_std_i = audio_encoder(audio_encoder_input_i)
    mu_i = audio_encoder_output_mu_i
    std_i = torch.nn.functional.softplus(audio_encoder_output_std_i) + 1e-8
    audio_latent_i = audio_encoder.reparameterize(mu_i, std_i)

    # normalise audio latent
    audio_latent_i = (audio_latent_i - audio_latents_mean) / (audio_latents_std + 1e-8)

    return audio_latent_i

@torch.no_grad()
def decode_audio(audio_latents_norm, vocos_model, audio_decoder_model):
    """Decode vocos latents to waveform."""
    
    #print("decode_audio audio_latents_norm s ", audio_latents_norm.shape)
    
    global current_vocos_audio_mels_in

    # denormalise audio latents
    audio_latents = audio_latents_norm * audio_latents_std + audio_latents_mean
    
    #print("audio_latents s ", audio_latents.shape, " audio_latents_std s ", audio_latents_std.shape, " audio_latents_mean s ", audio_latents_mean.shape)

    # decode audio latents
    audio_mels_norm = audio_decoder(audio_latents) 
    
    #print("audio_mels_norm s ", audio_mels_norm.shape)
    
    audio_mels_norm = audio_mels_norm.squeeze(1)
    
    #print("audio_mels_norm 2 s ", audio_mels_norm.shape)
    
    #print("audio_mels_norm s ", audio_mels_norm.shape, " audio_std s ", audio_std.shape, " audio_mean s ", audio_mean.shape)

    # denormalise audio mels
    audio_mels = audio_mels_norm * audio_std + audio_mean
    
    #print("audio_mels s ", audio_mels.shape)

     # insert gen audio mels to current_vocos_mels
     
    #print("current_vocos_audio_mels_in s ", current_vocos_audio_mels_in.shape)
     
    current_vocos_audio_mels_in = torch.roll(current_vocos_audio_mels_in, shifts=-audio_vae_mel_count, dims=2)
    current_vocos_audio_mels_in[:, :, -audio_vae_mel_count:] = audio_mels
    
    #print("current_vocos_audio_mels_in s ", current_vocos_audio_mels_in.shape)

    # get waveform from mels
    current_audio_waveform = vocos.decode(current_vocos_audio_mels_in)
    
    #print("current_audio_waveform s ", current_audio_waveform.shape)

    # get last part of waveform
    current_audio_waveform = current_audio_waveform[0, -audio_vae_waveform_length:]
    current_audio_waveform = current_audio_waveform.reshape(-1)
    
    #print("current_audio_waveform 2 s ", current_audio_waveform.shape)

    return current_audio_waveform

@torch.no_grad()
def audio_synthesis(motion_latents_norm, audio_latents_norm):
    
    #print("audio_synthesis motion_latents_norm s ", motion_latents_norm.shape, " audio_latents_norm s ", audio_latents_norm.shape)

    transformer_motion_input = motion_latents_norm.unsqueeze(0)
    transformer_audio_input = audio_latents_norm.unsqueeze(0)
    transformer_audio_output = transformer(transformer_motion_input, transformer_audio_input)
    transformer_audio_output = transformer_audio_output.squeeze(0)

    return transformer_audio_output

# -------------------------------------------------------
# Load Audio and Mocap Data
# -------------------------------------------------------

mocap_data = load_mocap(mocap_data_file, mocap_pos_scale)
mocap_sequence = mocap_data["motion"]["rot_local"]
mocap_pose_dim = mocap_sequence.shape[1]

if load_mocap_stats == True:
    mocap_mean = torch.load(mocap_mean_file)
    mocap_std = torch.load(mocap_std_file)
    
    mocap_mean = mocap_mean.to(device)
    mocap_std = mocap_std.to(device)
else:

    mocap_mean = np.mean(mocap_sequence, axis=0, keepdims=True)
    mocap_std = np.std(mocap_sequence, axis=0, keepdims=True)
    
    mocap_mean = torch.from_numpy(mocap_mean).to(dtype=torch.float32)
    mocap_std = torch.from_numpy(mocap_std).to(dtype=torch.float32)
    
    print("mocap_mean s ", mocap_mean.shape)
    print("mocap_std s ", mocap_std.shape)
    
    torch.save(mocap_mean, mocap_mean_file)
    torch.save(mocap_std, mocap_std_file)
    
    mocap_mean = mocap_mean.to(device)
    mocap_std = mocap_std.to(device)

audio_waveform, _ = torchaudio.load(audio_data_file)
audio_waveform = audio_waveform[0]

if "skeleton" in mocap_data:
    mocap_skeleton = mocap_data["skeleton"]
    mocap_root_joint_index = mocap_skeleton["joints"].index(mocap_skeleton["root"])

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------

vocos = Vocos.from_pretrained(vocos_pretrained_config).to(device).eval()

audio_vae_waveform = torch.zeros((1, audio_vae_waveform_length), dtype=torch.float32).to(device)
audio_vae_mels = vocos.feature_extractor(audio_vae_waveform)
audio_vae_waveform_rec = vocos.decode(audio_vae_mels)

print("audio_vae_waveform s ", audio_vae_waveform.shape)
print("audio_vae_mels s ", audio_vae_mels.shape)
print("audio_vae_waveform_rec s ", audio_vae_waveform_rec.shape)

audio_mel_filter_count = audio_vae_mels.shape[1]
audio_vae_mel_count = audio_vae_mels.shape[-1]

print("audio_mel_filter_count ", audio_mel_filter_count)
print("audio_vae_mel_count ", audio_vae_mel_count)

if load_audio_stats == True:
    audio_mean = torch.load(audio_mean_file)
    audio_std = torch.load(audio_std_file)
    
    audio_mean = audio_mean.to(device)
    audio_std = audio_std.to(device)
else:
    
    audio_mels = vocos.feature_extractor(audio_waveform.to(device))
        
    print("audio_mels s ", audio_mels.shape)
    
    audio_mean = torch.mean(audio_mels, dim=1, keepdim=True)
    audio_std = torch.std(audio_mels, dim=1, keepdim=True)
    
    print("audio_mean s ", audio_mean.shape)
    print("audio_std s ", audio_std.shape)
    
    torch.save(audio_mean.detach().cpu(), audio_mean_file)
    torch.save(audio_std.detach().cpu(), audio_std_file)


motion_encoder = MotionEncoder(motion_vae_mocap_length, mocap_pose_dim, motion_vae_latent_dim, motion_vae_rnn_layer_count, motion_vae_rnn_layer_size, motion_vae_dense_layer_sizes).to(device)
motion_encoder.load_state_dict(torch.load(motion_vae_encoder_weights_file, map_location=device))
motion_encoder.eval()

motion_latents_mean = torch.load(motion_latents_mean_file).to(device)
motion_latents_std = torch.load(motion_latents_std_file).to(device)

audio_encoder = AudioEncoder(audio_vae_latent_dim, audio_vae_mel_count, audio_mel_filter_count, audio_vae_conv_channel_counts, audio_vae_conv_kernel_size, audio_vae_dense_layer_sizes).to(device)
audio_encoder.load_state_dict(torch.load(audio_vae_encoder_weights_file, map_location=device))
audio_encoder.eval()

audio_decoder = AudioDecoder(audio_vae_latent_dim, audio_vae_mel_count, audio_mel_filter_count, list(reversed(audio_vae_conv_channel_counts)), audio_vae_conv_kernel_size, list(reversed(audio_vae_dense_layer_sizes))).to(device)
audio_decoder.load_state_dict(torch.load(audio_vae_decoder_weights_file, map_location=device))
audio_decoder.eval()

audio_latents_mean = torch.load(audio_latents_mean_file).to(device)
audio_latents_std = torch.load(audio_latents_std_file).to(device)


transformer = Transformer(
    motion_vae_latent_dim,
    audio_vae_latent_dim,
    transformer_embed_dim,
    transformer_head_count,
    transformer_layer_count,
    transformer_layer_count,
    transformer_dropout,
    pos_encoding_max_length
).to(device)

print(transformer)

transformer.load_state_dict(torch.load(transformer_weights_file, map_location=device))
transformer.eval()

# Precompute audio window
hann_window = torch.from_numpy(np.hanning(audio_window_length)).float().to(device)

# =========================================================
# FIFO Queues and Buffers
# =========================================================

motion_latents_norm_queue =queue.Queue(maxsize=max_fifo_queue_length)
audio_waveforms_queue = queue.Queue(maxsize=max_fifo_queue_length)

current_motion_encoder_in = torch.zeros((motion_vae_mocap_length, mocap_joint_count, mocap_joint_dim))
current_motion_encoder_in[:, :, 0] = 1.0
current_motion_encoder_in = current_motion_encoder_in.reshape(motion_vae_mocap_length, mocap_pose_dim)
current_motion_encoder_in = current_motion_encoder_in.to(device)

current_transformer_motion_latents_in = torch.zeros((transformer_motion_latent_input_length, motion_vae_latent_dim)).to(device)

init_waveform = audio_waveform[0, :transformer_audio_waveform_input_length]  # 60 frames worth
init_latents = []
for i in range(transformer_audio_latent_input_length):
    start = i * audio_waveform_length_per_mocap_frame
    end   = start + audio_vae_waveform_length  # keep same as training window
    segment = init_waveform[start:end]
    latent_i = encode_audio(segment, vocos, audio_encoder)
    init_latents.append(latent_i.squeeze(0))  # (latent_dim,)
current_transformer_audio_latents_in = torch.stack(init_latents, dim=0).to(device)

#current_transformer_audio_latents_in = torch.zeros((transformer_audio_latent_input_length, audio_vae_latent_dim)).to(device)
current_vocos_audio_mels_in = vocos.feature_extractor(torch.zeros((1, audio_vocos_waveform_length)).to(device))


export_audio_buffer = []

# =========================================================
# Motion Capture Receiver
# =========================================================

class MocapReceiver:
    """Receives mocap OSC messages and pushes frames to mocap_queue."""

    def __init__(self, ip, port):
        self.ip, self.port = ip, port
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/mocap/0/joint/rot_local", self.updateMocapQueue)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)
        self.thread = None

    def start(self):
        """Start OSC server in background thread."""
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def stop(self):
        """Gracefully shutdown OSC server."""
        self.server.shutdown()
        self.server.server_close()

    def join(self):
        """Wait for server thread to exit."""
        if self.thread is not None:
            self.thread.join()

    def updateMocapQueue(self, address, *args):
        """OSC handler for receiving mocap frames."""
        
        global current_motion_encoder_in
        
        mocap_frame = np.asarray(args, dtype=np.float32)
        
        #print("mocap_frame s ", mocap_frame.shape)
        
        # set root rotation to zero
        mocap_frame = mocap_frame.reshape((mocap_joint_count, mocap_joint_dim))
        mocap_frame[mocap_root_joint_index, :] = np.array((1.0, 0.0, 0.0, 0.0))
        mocap_frame = mocap_frame.flatten()
        
        #print("mocap_frame [4:8] ", mocap_frame[4:8])
        
        mocap_frame = torch.from_numpy(mocap_frame).to(torch.float32).unsqueeze(0).to(device)
        
        #print("mocap_frame 2 s ", mocap_frame.shape)
        
        # normalise mocap frame
        mocap_frame_norm = (mocap_frame - mocap_mean) / (mocap_std + 1e-8)
        
        #print("mocap_frame_norm s ", mocap_frame_norm.shape, " mocap_mean s ", mocap_mean.shape, " mocap_std s ", mocap_std.shape)
        
        # create motion encoding
        
        #print("current_motion_encoder_in s ", current_motion_encoder_in.shape)
        
        current_motion_encoder_in = torch.roll(current_motion_encoder_in, shifts=-1, dims=0)
        current_motion_encoder_in[-1, :] = mocap_frame_norm[0]
        motion_latents_norm = encode_motion(current_motion_encoder_in, motion_encoder)
        
        while motion_latents_norm_queue.full():
            motion_latents_norm_queue.get_nowait()  # Drop oldest
            

        motion_latents_norm_queue.put(motion_latents_norm.detach().cpu())

# =========================================================
# Audio Producer Thread
# =========================================================

shutdown_event = threading.Event()

save_counter = 0 # debug


def producer_thread():
    """Continuously predicts and enqueues new audio buffers."""
    global current_transformer_audio_latents_in, current_transformer_motion_latents_in
    global save_counter # debug
    while not shutdown_event.is_set():
        if not audio_waveforms_queue.full():
            current_transformer_motion_latents_in = torch.roll(current_transformer_motion_latents_in, shifts=-1, dims=0)
            
            if not motion_latents_norm_queue.empty():
                current_transformer_motion_latents_in[-1] = motion_latents_norm_queue.get_nowait().to(device)

            transformer_audio_latents_out = audio_synthesis(current_transformer_motion_latents_in, current_transformer_audio_latents_in)
            
            #print("current_transformer_audio_latents_in s ", current_transformer_audio_latents_in.shape, " transformer_audio_latents_out s ", transformer_audio_latents_out.shape)
            #print("transformer_audio_latents_out s ", transformer_audio_latents_out.shape)
            
            gen_waveform = decode_audio(transformer_audio_latents_out[-1:, :], vocos, audio_decoder)
            
            #print("gen_waveform s ", gen_waveform.shape)
            
            #gen_waveform = gen_waveform[-audio_window_length:]
             
            # Optional highpass:
            # gen_waveform = highpass_biquad(gen_waveform, audio_sample_rate, hpf_cutoff_hz)
            audio_waveforms_queue.put(gen_waveform.cpu().numpy())
            current_transformer_audio_latents_in = transformer_audio_latents_out.detach().squeeze(0)
        else:
            time.sleep(0.01)  # Avoid busy-waiting

# =========================================================
# AUDIO CALLBACK (playback)
# =========================================================

# This must be maintained across audio_callback calls:
last_chunk = np.zeros(audio_window_length, dtype=np.float32)

def audio_callback(out_data, frames, time_info, status):
        
    #print("audio_callback")
    
    """Overlap-add from audio_queue to output audio buffer."""
    global export_audio_buffer, last_chunk
    output = np.zeros((frames, audio_channels), dtype=np.float32)
    cursor = 0
  
    # Start with second half of last block from previous callback
    #output[cursor:cursor+audio_window_offset, 0] += last_chunk[-audio_window_offset:]
    output[cursor:cursor+audio_window_length // 2, 0] += last_chunk[-audio_window_length // 2:]

    while cursor < frames:
        
        #print("cursor ", cursor, " frames ", frames, " queue ", audio_waveforms_queue.qsize())
        
        try:
            chunk = audio_waveforms_queue.get_nowait()
            chunk = chunk * hann_window.cpu().numpy()
        except queue.Empty:
            chunk = np.zeros(audio_window_length, dtype=np.float32)  # Output silence
        chunk_size = output[cursor:cursor+audio_window_length, 0].shape[0]
        output[cursor:cursor+chunk_size, 0] += chunk[:chunk_size]
        
        cursor += audio_window_offset
        last_chunk[:] = chunk[:]

    out_data[:] = output
    export_audio_buffer.append(output[:, 0])

# =========================================================
# MAIN RUNTIME
# =========================================================
if __name__ == "__main__":
    
    mocap_receiver = MocapReceiver(osc_receive_ip, osc_receive_port)

    mocap_receiver.start()
    threading.Thread(target=producer_thread, daemon=True).start()

    sd.sleep(2000)  # Allow queue to prefill

    with sd.OutputStream(
        samplerate=audio_sample_rate,
        device=audio_output_device,
        channels=audio_channels,
        callback=audio_callback,
        blocksize=play_buffer_size,
        latency=playback_latency
    ):
        print("Streaming audio... Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopping...")
            shutdown_event.set()
            mocap_receiver.stop()
            mocap_receiver.join()

    # Save streamed audio to WAV
    export_audio_buffer_np = np.concatenate(export_audio_buffer, axis=0)
    export_audio_buffer_tensor = torch.from_numpy(export_audio_buffer_np).unsqueeze(0)
    torchaudio.save("audio_export.wav", export_audio_buffer_tensor, audio_sample_rate)