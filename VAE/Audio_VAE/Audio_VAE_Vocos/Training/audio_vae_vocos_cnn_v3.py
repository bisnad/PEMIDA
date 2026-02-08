"""
Imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import scipy.linalg as sclinalg
from sklearn.manifold import TSNE

import math
import random
import os, sys, time, subprocess
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
import json

# audio specific imports

import torchaudio
import torchaudio.transforms as transforms
import simpleaudio as sa
import auraloss

# vocos specific imports

from vocos import Vocos 


"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Seed
"""

seed = 42 # if None, then no random seed will be used

"""
Mocap Settings
"""

mocap_fps = 30


"""
Audio Settings
"""

"""
# Eleni
audio_data_path = "E:/Data/audio/Eleni/48khz/"
audio_data_files = ["4_5870821179501060412.wav"]
audio_valid_ranges = [[2.71, 825.75]]

# Diane
audio_data_path = "E:/Data/audio/Diane/48khz/"
audio_data_files = ["4d69949b.wav"]
audio_valid_ranges = [[5.0, 377.91]]

# Tim 
audio_data_path = "E:/Data/audio/Tim/48khz/"
audio_data_files = ["SajuHariPlacePrizeEntry2010.wav"]
audio_valid_ranges = [[0.0, 394.06]]
"""

audio_data_path = "E:/Data/audio/Eleni/48khz/"
audio_data_files = ["4_5870821179501060412.wav"]
audio_valid_ranges = [[2.71, 825.75]]

audio_sample_rate = 48000
audio_channels = 1
audio_waveform_length_per_mocap_frame = int(1.0 / mocap_fps * audio_sample_rate)

"""
Vocos Settings
"""

vocos_pretrained_config = "kittn/vocos-mel-48khz-alpha1"
audio_vocos_waveform_length = 44800

"""
Dataset Settings
"""

mocap_frame_incr = 4
batch_size = 128 # 32
test_percentage = 0.1


"""
Motion VAE Model Settings
"""

motion_vae_mocap_length = 8

"""
Audio VAE Model Settings
"""

audio_vae_waveform_length = int(motion_vae_mocap_length / mocap_fps * audio_sample_rate)
audio_vae_mel_count = None # automatically calculated

audio_vae_latent_dim = 32 # 128
audio_vae_conv_channel_counts = [ 16, 32, 64, 128 ]
audio_vae_conv_kernel_size = (5, 3)
audio_vae_dense_layer_sizes = [ 512 ]

"""
General Training Settings
"""

test_percentage  = 0.1
epochs = 400
model_save_interval = 50

vae_learning_rate = 1e-4 #1e-4

vae_beta = 0.0 # will be calculated
vae_beta_cycle_duration = 100
vae_beta_min_const_duration = 20
vae_beta_max_const_duration = 20
vae_min_beta = 0.0
vae_max_beta = 0.1


"""
Audio VAE Training Settings
"""

audio_vae_rec_loss_scale = 5.0

load_audio_vae_weights = True
save_audio_vae_weights = False
audio_vae_encoder_weights_file = "results_Eleni_audio_vae_vocos_cnn_v3_ld32_kld0.1_offset1/weights/audio_encoder_weights_epoch_400"
audio_vae_decoder_weights_file = "results_Eleni_audio_vae_vocos_cnn_v3_ld32_kld0.1_offset1/weights/audio_decoder_weights_epoch_400"


"""
Fix Seeds
"""

def set_all_seeds(seed: int):
    # Python's built-in RNG
    random.seed(seed)
    # NumPy RNG
    np.random.seed(seed)
    # PyTorch RNG (CPU)
    torch.manual_seed(seed)
    # PyTorch RNG (CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # (optional) PyTorch backend for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(seed)

"""
Load Data - Audio
"""

audio_all_data = []

for audio_data_file, audio_valid_range in zip(audio_data_files, audio_valid_ranges):   
    
    audio_data, _ = torchaudio.load(audio_data_path + audio_data_file)
  
    print("audio_data s ", audio_data.shape)
    
    audio_range_begin = audio_valid_range[0]
    audio_range_end = audio_valid_range[1]
    
    print("audio_valid_range ", audio_valid_range)
    
    if audio_range_begin > 0 and audio_range_end > 0:
        audio_valid_range_sample = [ int(audio_valid_range[0] * audio_sample_rate), int(audio_valid_range[1] * audio_sample_rate)]    
    else: 
        audio_valid_range_sample = [ 0, audio_data.shape[-1]]   

    print("audio_valid_range_sample ", audio_valid_range_sample)
    
    audio_data = audio_data[:, audio_valid_range_sample[0]:audio_valid_range_sample[1]]
    
    print("audio_data 2 s ", audio_data.shape)
    
    audio_all_data.append(audio_data)

"""
Load Vocos Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()

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

"""
Create Dataset
"""

audio_dataset = []

audio_waveform_start_offset = audio_vocos_waveform_length - audio_vae_waveform_length
mocap_frame_start_offset = int(audio_waveform_start_offset / audio_sample_rate * mocap_fps)

for sI in range(len(audio_all_data)):
    
    with torch.no_grad():
        
        audio_data = audio_all_data[sI][0]
    
        print(sI)
        print("audio_data s ", audio_data.shape)
        
        mocap_frame_count = int(audio_data.shape[0] / audio_sample_rate * mocap_fps)

        for mfI in range(mocap_frame_start_offset, mocap_frame_count - motion_vae_mocap_length * 3, mocap_frame_incr):
            
            print("mfI ", mfI, " out of ", (mocap_frame_count - motion_vae_mocap_length))
            
            # audio sequence part
            asI = mfI * audio_waveform_length_per_mocap_frame
            audio_waveform_start = asI - audio_waveform_start_offset
            audio_waveform_end = audio_waveform_start + audio_vocos_waveform_length
            
            #print("audio_waveform_start ", audio_waveform_start)
            #print("audio_waveform_end ", audio_waveform_end)
            
            audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0).to(device)
            
            #print("mfI ", mfI, " audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
            
            audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
            
            #print("mfI ", mfI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)
            
            audio_dataset.append(audio_mels_excerpt.cpu())

audio_dataset = torch.cat(audio_dataset, dim=0)

print("audio_dataset s ", audio_dataset.shape)

class AudioDataset(Dataset):
    def __init__(self, audio):
        self.audio = audio
    
    def __len__(self):
        return self.audio.shape[0]
    
    def __getitem__(self, idx):
        return self.audio[idx, ...]
    
full_dataset = AudioDataset(audio_dataset)

item_audio = full_dataset[0]

print("item_audio s ", item_audio.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

batch_audio = next(iter(train_loader))

print("batch_audio s ", batch_audio.shape)

"""
Create Models
"""

"""
Create Audio VAE
"""

"""
Create Audio VAE Encoder
"""

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

audio_encoder = AudioEncoder(audio_vae_latent_dim, audio_vae_mel_count, audio_mel_filter_count, audio_vae_conv_channel_counts, audio_vae_conv_kernel_size, audio_vae_dense_layer_sizes).to(device)

print(audio_encoder)

if load_audio_vae_weights and audio_vae_encoder_weights_file:
    audio_encoder.load_state_dict(torch.load(audio_vae_encoder_weights_file, map_location=device))

# test audio encoder

audio_encoder_input = batch_audio.unsqueeze(1).to(device)
audio_encoder_input = audio_encoder_input[:, :, :, -audio_vae_mel_count:]

audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
audio_encoder_output = audio_encoder.reparameterize(audio_encoder_output_mu, torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8)

print("audio_encoder_input s ", audio_encoder_input.shape)
print("audio_encoder_output s ", audio_encoder_output.shape)

"""
Create Audio VAE Decoder
"""

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
        
        print("stride ", stride)
                
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
    
audio_vae_conv_channel_counts_reversed = audio_vae_conv_channel_counts.copy()
audio_vae_conv_channel_counts_reversed.reverse()
    
audio_vae_dense_layer_sizes_reversed = audio_vae_dense_layer_sizes.copy()
audio_vae_dense_layer_sizes_reversed.reverse()

audio_decoder = AudioDecoder(audio_vae_latent_dim, audio_vae_mel_count, audio_mel_filter_count, audio_vae_conv_channel_counts_reversed, audio_vae_conv_kernel_size, audio_vae_dense_layer_sizes_reversed).to(device)

print(audio_decoder)

if load_audio_vae_weights and audio_vae_decoder_weights_file:
    audio_decoder.load_state_dict(torch.load(audio_vae_decoder_weights_file, map_location=device))

# test audio decoder
audio_decoder_input = audio_encoder_output
audio_decoder_output = audio_decoder(audio_decoder_input)

print("audio_decoder_input s ", audio_decoder_input.shape)
print("audio_decoder_output s ", audio_decoder_output.shape)

"""
Training
"""

def calc_vae_beta_values():
    
    vae_beta_values = []

    for e in range(epochs):
        
        cycle_step = e % vae_beta_cycle_duration
        
        #print("cycle_step ", cycle_step)

        if cycle_step < vae_beta_min_const_duration:
            vae_beta_value = vae_min_beta
            vae_beta_values.append(vae_beta_value)
        elif cycle_step > vae_beta_cycle_duration - vae_beta_max_const_duration:
            vae_beta_value = vae_max_beta
            vae_beta_values.append(vae_beta_value)
        else:
            lin_step = cycle_step - vae_beta_min_const_duration
            vae_beta_value = vae_min_beta + (vae_max_beta - vae_min_beta) * lin_step / (vae_beta_cycle_duration - vae_beta_min_const_duration - vae_beta_max_const_duration)
            vae_beta_values.append(vae_beta_value)
            
    return vae_beta_values

vae_beta_values = calc_vae_beta_values()

"""
Audio VAE Optimizer & Scheduler
"""

audio_vae_optimizer = torch.optim.Adam(list(audio_encoder.parameters()) + list(audio_decoder.parameters()), lr=vae_learning_rate)
audio_vae_scheduler = torch.optim.lr_scheduler.StepLR(audio_vae_optimizer, step_size=100, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

"""
General Loss Functions
"""

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# KL Divergence
def variational_loss(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl

"""
Audio VAE Specific Loss Functions
"""

# Define perceptial loss: MR-STFT with perceptual mel weighting
perc_loss = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[1024, 2048, 8192],
    hop_sizes=[256, 512, 2048],
    win_lengths=[1024, 2048, 8192],
    scale="mel",          # use mel-scaled spectrograms
    n_bins=128,           # number of mel bins for the perceptual weighting
    sample_rate=48000,    # set to the actual SR used (48 kHz in your code)
    perceptual_weighting=True
)

def audio_vae_mel_loss(y_norm, yhat_norm):
    
    _aml = mse_loss(yhat_norm, y_norm)

    return _aml

def audio_vae_perc_loss(y_wave, yhat_wave):
    
    apl = perc_loss(yhat_wave, y_wave)
    
    return apl

"""
Training and Test Steps
"""    

"""
Audio VAE Training and Test Step
"""

def audio_vae_train_step(y_norm):
    
    #print("y_norm s ", y_norm.shape)
    
    y_norm = y_norm.unsqueeze(1).to(device)
    
    #print("y_norm 2 s ", y_norm.shape)
    
    # use only the last mels for y_norm for the vae training
    audio_encoder_input = y_norm[:,:,:, -audio_vae_mel_count:]
    
    #print("audio_encoder_input s ", audio_encoder_input.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    mu = audio_encoder_output_mu
    std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    
    audio_encoder_output = audio_encoder.reparameterize(mu, std)
    
    #print("audio_encoder_output s ", audio_encoder_output.shape)
    
    audio_decoder_input = audio_encoder_output
    
    #print("audio_decoder_input s ", audio_decoder_input.shape)
    
    audio_decoder_output = audio_decoder(audio_decoder_input)
    
    #print("audio_decoder_output s ", audio_decoder_output.shape)
    
    # kld loss
    _kld_loss = variational_loss(mu, std)
    
    # mel rec loss
    _mel_rec_loss = audio_vae_mel_loss(audio_encoder_input.squeeze(1), audio_decoder_output.squeeze(1))
    
    # perc rec loss
    y_mels = y_norm.squeeze(1)
    yhat_mels = audio_decoder_output.squeeze(1)
    
    #print("y_mels s ", y_mels.shape)
    #print("yhat_mels s ", yhat_mels.shape)
    
    # prepend the first y_mels to yhat_norm to create the full number of mels required by vocos
    yhat_mels = torch.cat([y_mels[:, :, :-audio_vae_mel_count], yhat_mels], dim=2)
    
    #print("yhat_mels 2 s ", yhat_mels.shape)
    
    # convert mels to waveforms
    y_wave = vocos.decode(y_mels)
    yhat_wave = vocos.decode(yhat_mels)
    
    #print("y_wave s ", y_wave.shape)
    #print("yhat_wave s ", yhat_wave.shape)
    
    # take only the last part of y_wave and yhat_wave that correspond to audio_vae_waveform_length
    y_wave = y_wave[:, -audio_vae_waveform_length:]
    yhat_wave = yhat_wave[:, -audio_vae_waveform_length:]
    
    #print("y_wave 2 s ", y_wave.shape)
    #print("yhat_wave 2 s ", yhat_wave.shape)
    
    y_wave = y_wave.unsqueeze(1)
    yhat_wave = yhat_wave.unsqueeze(1)
    
    #print("y_wave 3 s ", y_wave.shape)
    #print("yhat_wave 3 s ", yhat_wave.shape)
    
    _perc_rec_loss = audio_vae_perc_loss(y_wave, yhat_wave)
    
    _total_loss = 0.0
    _total_loss += _mel_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _perc_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _kld_loss * vae_beta
    
    # Backpropagation
    audio_vae_optimizer.zero_grad()
    _total_loss.backward()
    
    audio_vae_optimizer.step()
    
    return _total_loss, _mel_rec_loss, _perc_rec_loss, _kld_loss

"""
_losses = audio_vae_train_step(batch_audio)
"""

@torch.no_grad()
def audio_vae_test_step(y_norm):
    
    #print("y_norm s ", y_norm.shape)
    
    y_norm = y_norm.unsqueeze(1).to(device)
    
    #print("y_norm 2 s ", y_norm.shape)
    
    # use only the last mels for y_norm for the vae training
    audio_encoder_input = y_norm[:,:,:, -audio_vae_mel_count:]
    
    #print("audio_encoder_input s ", audio_encoder_input.shape)
    
    audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
    mu = audio_encoder_output_mu
    std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
    
    audio_encoder_output = audio_encoder.reparameterize(mu, std)
    
    #print("audio_encoder_output s ", audio_encoder_output.shape)
    
    audio_decoder_input = audio_encoder_output
    
    #print("audio_decoder_input s ", audio_decoder_input.shape)
    
    audio_decoder_output = audio_decoder(audio_decoder_input)
    
    #print("audio_decoder_output s ", audio_decoder_output.shape)
    
    # kld loss
    _kld_loss = variational_loss(mu, std)
    
    # mel rec loss
    _mel_rec_loss = audio_vae_mel_loss(audio_encoder_input.squeeze(1), audio_decoder_output.squeeze(1))
    
    # perc rec loss
    y_mels = y_norm.squeeze(1)
    yhat_mels = audio_decoder_output.squeeze(1)
    
    #print("y_mels s ", y_mels.shape)
    #print("yhat_mels s ", yhat_mels.shape)
    
    # prepend the first y_mels to yhat_norm to create the full number of mels required by vocos
    yhat_mels = torch.cat([y_mels[:, :, :-audio_vae_mel_count], yhat_mels], dim=2)
    
    #print("yhat_mels 2 s ", yhat_mels.shape)
    
    # convert mels to waveforms
    y_wave = vocos.decode(y_mels)
    yhat_wave = vocos.decode(yhat_mels)
    
    #print("y_wave s ", y_wave.shape)
    #print("yhat_wave s ", yhat_wave.shape)
    
    # take only the last part of y_wave and yhat_wave that correspond to audio_vae_waveform_length
    y_wave = y_wave[:, -audio_vae_waveform_length:]
    yhat_wave = yhat_wave[:, -audio_vae_waveform_length:]
    
    #print("y_wave 2 s ", y_wave.shape)
    #print("yhat_wave 2 s ", yhat_wave.shape)
    
    y_wave = y_wave.unsqueeze(1)
    yhat_wave = yhat_wave.unsqueeze(1)
    
    #print("y_wave 3 s ", y_wave.shape)
    #print("yhat_wave 3 s ", yhat_wave.shape)
    
    _perc_rec_loss = audio_vae_perc_loss(y_wave, yhat_wave)
    
    _total_loss = 0.0
    _total_loss += _mel_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _perc_rec_loss * audio_vae_rec_loss_scale * 0.5
    _total_loss += _kld_loss * vae_beta

    return _total_loss, _mel_rec_loss, _perc_rec_loss, _kld_loss


"""
Train Function
"""

def train(train_dataloader, test_dataloader, epochs):
    
    global vae_beta
    
    loss_history = {}

    loss_history["audio vae train"] = []
    loss_history["audio vae test"] = []
    loss_history["audio vae mel"] = []
    loss_history["audio vae perc"] = []
    loss_history["audio vae kld"] = []

    for epoch in range(epochs):

        start = time.time()
        
        vae_beta = vae_beta_values[epoch]
        
        #print("vae_beta ", vae_beta)

        audio_vae_train_loss_per_epoch = []
        audio_vae_test_loss_per_epoch = []
        audio_vae_mel_loss_per_epoch = []
        audio_vae_perc_loss_per_epoch = []
        audio_vae_kld_loss_per_epoch = []

        for audio_batch in train_dataloader:
            audio_batch = audio_batch.to(device)

            # audio vae
            _audio_vae_total_loss, _audio_vae_mel_rec_loss, _audio_vae_perc_rec_loss, _audio_vae_kld_loss = audio_vae_train_step(audio_batch)
           
            _audio_vae_total_loss = _audio_vae_total_loss.detach().cpu().numpy()
            _audio_vae_mel_rec_loss = _audio_vae_mel_rec_loss.detach().cpu().numpy()
            _audio_vae_perc_rec_loss = _audio_vae_perc_rec_loss.detach().cpu().numpy()
            _audio_vae_kld_loss = _audio_vae_kld_loss.detach().cpu().numpy()
            
            audio_vae_train_loss_per_epoch.append(_audio_vae_total_loss)
            audio_vae_mel_loss_per_epoch.append(_audio_vae_mel_rec_loss)
            audio_vae_perc_loss_per_epoch.append(_audio_vae_perc_rec_loss)
            audio_vae_kld_loss_per_epoch.append(_audio_vae_kld_loss)
            
        audio_vae_train_loss_per_epoch = np.mean(np.array(audio_vae_train_loss_per_epoch))
        audio_vae_mel_loss_per_epoch = np.mean(np.array(audio_vae_mel_loss_per_epoch))
        audio_vae_perc_loss_per_epoch = np.mean(np.array(audio_vae_perc_loss_per_epoch))
        audio_vae_kld_loss_per_epoch = np.mean(np.array(audio_vae_kld_loss_per_epoch))

        for audio_batch in test_dataloader:
            audio_batch = audio_batch.to(device)

            # audio vae
            _audio_vae_total_loss, _, _, _ = audio_vae_test_step(audio_batch)
           
            _audio_vae_total_loss = _audio_vae_total_loss.detach().cpu().numpy()
            
            audio_vae_test_loss_per_epoch.append(_audio_vae_total_loss)
            
        audio_vae_test_loss_per_epoch = np.mean(np.array(audio_vae_test_loss_per_epoch))

        if epoch % model_save_interval == 0 and save_audio_vae_weights == True:
            torch.save(audio_encoder.state_dict(), "results/weights/audio_encoder_weights_epoch_{}".format(epoch))
            torch.save(audio_decoder.state_dict(), "results/weights/audio_decoder_weights_epoch_{}".format(epoch))

        loss_history["audio vae train"].append(audio_vae_train_loss_per_epoch)
        loss_history["audio vae test"].append(audio_vae_test_loss_per_epoch)
        loss_history["audio vae mel"].append(audio_vae_mel_loss_per_epoch)
        loss_history["audio vae perc"].append(audio_vae_perc_loss_per_epoch)
        loss_history["audio vae kld"].append(audio_vae_kld_loss_per_epoch)

        print ("epoch {} : ".format(epoch + 1) +
               "audio train : {:01.4f} ".format(audio_vae_train_loss_per_epoch) +
               "audio test : {:01.4f} ".format(audio_vae_test_loss_per_epoch) +
               "audio mel : {:01.4f} ".format(audio_vae_mel_loss_per_epoch) +
               "audio perc : {:01.4f} ".format(audio_vae_perc_loss_per_epoch) +
               "audio kld : {:01.4f} ".format(audio_vae_kld_loss_per_epoch) +
               "time {:01.2f} ".format(time.time()-start))

        audio_vae_scheduler.step()
        
    return loss_history

"""
Fit Model
"""

if save_audio_vae_weights == True:
    loss_history = train(train_loader, test_loader, epochs)

"""
Save Training History
"""

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()
    
def save_loss_as_image_normalised(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        losses = np.array(loss_history[key])
        loss_norm = losses / np.max(losses)
        plt.plot(range(epochs), loss_norm, label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()
    

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)

if save_audio_vae_weights == True:
    save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
    save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))
    save_loss_as_image_normalised(loss_history, "results/histories/history_norm_{}.png".format(epochs))

"""
Save Final Model Weights
"""

if save_audio_vae_weights == True:
    torch.save(audio_encoder.state_dict(), "results/weights/audio_encoder_weights_epoch_{}".format(epochs))
    torch.save(audio_decoder.state_dict(), "results/weights/audio_decoder_weights_epoch_{}".format(epochs))

"""
Inference
"""

"""
# Eleni

audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]
audio_highlight_duration_sec = 10.0

# Diane

audio_highlight_starts_sec = [5.0, 50.0, 100.0, 140.0, 160.0, 214.0, 270.0, 355.0]
audio_highlight_duration_sec = 10.0


# Tim

audio_highlight_starts_sec = [5.0, 25.0, 54.0, 90.0, 106.0, 226.0, 290.0, 320.0]
audio_highlight_duration_sec = 10.0
"""

audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]
audio_highlight_duration_sec = 10.0

import colorsys
from matplotlib.colors import hsv_to_rgb

def distinct_hsv_colors(n, s=0.8, v=0.9):
    """
    Generate n HSV colors (h, s, v in [0, 1]) with
    evenly spaced hues and fixed saturation/value.
    """
    if n <= 0:
        return []
    # Golden ratio conjugate gives decent spacing on [0, 1)
    phi = 0.618033988749895
    colors = []
    h = 0.0
    for _ in range(n):
        colors.append((h % 1.0, s, v))
        h += phi
    return colors


"""
Audio Inference
"""

def export_audio(waveform, file_name):
    
    torchaudio.save("{}".format(file_name), waveform, audio_sample_rate)
    
def export_vocos_audio(waveform, file_name):
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_vocos_waveform_length // 2
    audio_window_env = torch.hann_window(audio_vocos_waveform_length)
    
    audio_window_count = int(waveform_length - audio_vocos_waveform_length) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    #print("pred_audio_sequence s ", pred_audio_sequence.shape)
    
    for i in range(audio_window_count):
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_vocos_waveform_length
        
        target_audio = waveform[:, window_start:window_end]
        
        #print("i ", i, " target_audio s ", target_audio.shape)
        
        with torch.no_grad():
            audio_features = vocos.feature_extractor(target_audio.to(device))
            voc_audio = vocos.decode(audio_features).detach().cpu()

        #print("voc_audio s ", voc_audio.shape)
        #print("grain_env s ", grain_env.shape)

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_vocos_waveform_length] += voc_audio[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)

@torch.no_grad()
def encode_audio(orig_waveform, sample_indices):
    
    audio_encoder.eval()
    
    waveform = orig_waveform.to(device)
    waveform_length = waveform.shape[-1]
    
    latent_vectors = []
    
    waveform_excerpt_count = len(sample_indices)
    
    for excerpt_index in range(waveform_excerpt_count):
        
        excerpt_start_sample = sample_indices[excerpt_index]
        excerpt_end_sample = excerpt_start_sample + audio_vocos_waveform_length
        
        audio_waveform_excerpt = waveform[:, excerpt_start_sample:excerpt_end_sample].to(device)
        
        #print("audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
        
        audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
        
        audio_mels_excerpt = audio_mels_excerpt.unsqueeze(1)
        
        # use only the last audio mels
        audio_encoder_input = audio_mels_excerpt[:,:,:, -audio_vae_mel_count:]
        
        #print("audio_encoder_input s ", audio_encoder_input.shape)
        
        # excode audio mels
        audio_encoder_output_mu, audio_encoder_output_std = audio_encoder(audio_encoder_input)
        mu = audio_encoder_output_mu
        std = torch.nn.functional.softplus(audio_encoder_output_std) + 1e-8
        
        latent_vector = audio_encoder.reparameterize(mu, std)
        
        #print("latent_vector s ", latent_vector.shape)
        
        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()

        latent_vectors.append(latent_vector)
    
    audio_encoder.train()
    
    return latent_vectors

@torch.no_grad()
def decode_audio(latent_vectors, waveform_offset):
    
    audio_decoder.eval()
    
    seq_env = np.hanning(audio_vae_waveform_length)
    latent_count = len(latent_vectors)
    waveform_length = (latent_count - 1) * waveform_offset + audio_vae_waveform_length

    vocos_mels = vocos.feature_extractor(torch.zeros((1, audio_vocos_waveform_length), dtype=torch.float32).to(device))

    #print("vocos_mels s ", vocos_mels.shape)

    gen_waveform = np.zeros(shape=(waveform_length)).astype(np.float32)
    
    for latent_index in range(len(latent_vectors)):
        
        latent_vector = latent_vectors[latent_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)
        
        #print("latent_vector s ", latent_vector.shape)
        
        vae_mels = audio_decoder(latent_vector)
        
        #print("vae_mels s ", vae_mels.shape)
        
        vae_mels = vae_mels.squeeze(1)
        
        #print("vae_mels 2 s ", vae_mels.shape)

        
        vocos_mels = torch.roll(vocos_mels, shifts=-audio_vae_mel_count, dims=2)
        vocos_mels[:, :, -audio_vae_mel_count:] = vae_mels
        
        vocos_waveform = vocos.decode(vocos_mels)
        
        #print("vocos_waveform s ", vocos_waveform.shape)
        
        vae_waveform = vocos_waveform[0, -audio_vae_waveform_length:]
        
        #print("vae_waveform s ", vae_waveform.shape)
        
        vae_waveform = vae_waveform.detach().detach().cpu().numpy()
        
        asI = latent_index * waveform_offset
        gen_waveform[asI:asI + audio_vae_waveform_length] += vae_waveform * seq_env
    
    return np.reshape(gen_waveform, (1, -1))

@torch.no_grad()
def create_audio_space_representation(sequence_excerpts):
    
    audio_encoder.eval()

    encodings = []
    
    excerpt_count = sequence_excerpts.shape[0]
    
    for eI in range(0, excerpt_count, batch_size):
        
        excerpt = sequence_excerpts[eI:eI+batch_size]
 
        excerpt = torch.from_numpy(excerpt).to(device)
        
        excerpt = excerpt.unsqueeze(1)

        # use only the last mels
        excerpt = excerpt[:,:,:, -audio_vae_mel_count:]
  
        encoder_output_mu, encoder_output_std = audio_encoder(excerpt)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = audio_encoder.reparameterize(mu, std)

        latent_vector = latent_vector.detach().cpu()

        encodings.append(latent_vector)
        
    encodings = torch.cat(encodings, dim=0)
    
    print("encodings s ", encodings.shape)
    
    encodings = encodings.numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, max_iter=5000, verbose=1)    
    Z_tsne = tsne.fit_transform(encodings)
    
    audio_encoder.train()
    
    return Z_tsne


def create_audio_space_image(Z_tsne, audio_highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors_hsv = distinct_hsv_colors(len(audio_highlight_excerpt_ranges))
    plot_colors = [ hsv_to_rgb(hsv)  for hsv in plot_colors_hsv ]
    
    plt.figure()
    fig, ax = plt.subplots()
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)
        
    for hI, hR in enumerate(audio_highlight_excerpt_ranges):
        
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='o', facecolors="none", s=10.0, linewidths=0.2, edgecolors=plot_colors[hI], alpha=0.4)
        
        print("hR", hR, " hI ", hI, " hR[0]:hR[1] ", hR[0], ":", hR[1], " Z_tsne_x ", Z_tsne_x.shape, " color ", plot_colors[hI])
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=600)
    plt.close()
    
"""
Create Audio Space Plot
"""

audio_waveform_offset = int(mocap_frame_incr / mocap_fps * audio_sample_rate)

audio_highlight_ranges = []

for audio_highlight_start_sec in audio_highlight_starts_sec:
    audio_highlight_ranges.append([ int(audio_highlight_start_sec * audio_sample_rate) // audio_waveform_offset, int((audio_highlight_start_sec + audio_highlight_duration_sec) * audio_sample_rate) // audio_waveform_offset ])

Z_tsne = create_audio_space_representation(audio_dataset.numpy())
create_audio_space_image(Z_tsne, audio_highlight_ranges, "audio_space_plot_epoch_{}.png".format(epochs))

"""
Create Original Audio
"""

audio_data = audio_all_data[0]


for audio_start_sec in audio_highlight_starts_sec:

    audio_duration_sec = audio_highlight_duration_sec

    export_audio(audio_data[:, int(audio_start_sec * audio_sample_rate):int(audio_start_sec* audio_sample_rate + audio_duration_sec* audio_sample_rate)], "results/audio/orig_audio_start_{}_length_{}.wav".format(audio_start_sec, audio_duration_sec))
    export_vocos_audio(audio_data[:, int(audio_start_sec * audio_sample_rate):int(audio_start_sec* audio_sample_rate + audio_duration_sec* audio_sample_rate)], "results/audio/orig_vocos_audio_start_{}_length_{}.wav".format(audio_start_sec, audio_duration_sec))

"""
Recontruct Original Audio
"""

audio_waveform_offset = audio_vae_waveform_length // 8

for audio_start_sec in audio_highlight_starts_sec:

    audio_duration_sec = audio_highlight_duration_sec

    sample_indices = [ sample_index for sample_index in range(int(audio_start_sec * audio_sample_rate), int((audio_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

    audio_encodings = encode_audio(audio_data, sample_indices)
    gen_audio = decode_audio(audio_encodings, audio_waveform_offset)
    export_audio(torch.from_numpy(gen_audio), "results/audio/rec_audio_start_{}_length_{}_epochs_{}.wav".format(audio_start_sec, audio_duration_sec, epochs))

"""
Random Walk in Audio Space
"""

audio_waveform_offset = audio_vae_waveform_length // 8

for audio_start_sec in audio_highlight_starts_sec:

    audio_duration_sec = audio_highlight_duration_sec

    sample_indices = [int(audio_start_sec * audio_sample_rate)]

    audio_encodings = encode_audio(audio_data, sample_indices)

    for index in range(0, int(audio_duration_sec * audio_sample_rate) // audio_waveform_offset):
        random_step = np.random.random((audio_vae_latent_dim)).astype(np.float32) * 2.0
        audio_encodings.append(audio_encodings[index] + random_step)
        
    gen_audio = decode_audio(audio_encodings, audio_waveform_offset)
    export_audio(torch.from_numpy(gen_audio), "results/audio/randwalk_audio_start_{}_length_{}_epochs_{}.wav".format(audio_start_sec, audio_duration_sec, epochs))


"""
Sequence Offset Following in Audio Space
"""

audio_waveform_offset = audio_vae_waveform_length // 8

for audio_start_sec in audio_highlight_starts_sec:

    audio_duration_sec = audio_highlight_duration_sec

    sample_indices = [ sample_index for sample_index in range(int(audio_start_sec * audio_sample_rate), int((audio_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

    audio_encodings = encode_audio(audio_data, sample_indices)

    offset_audio_encodings = []

    for index in range(len(audio_encodings)):
        sin_value = np.sin(index / (len(audio_encodings) - 1) * np.pi * 4.0)
        offset = np.ones(shape=(audio_vae_latent_dim), dtype=np.float32) * sin_value * 4.0
        offset_audio_encoding = audio_encodings[index] + offset
        offset_audio_encodings.append(offset_audio_encoding)
        
    gen_audio = decode_audio(offset_audio_encodings, audio_waveform_offset)
    export_audio(torch.from_numpy(gen_audio), "results/audio/offset_audio_start_{}_length_{}_epochs_{}.wav".format(audio_start_sec, audio_duration_sec, epochs))

"""
Interpolate Two Audio Excerpts in Audio Space
"""

audio_waveform_offset = audio_vae_waveform_length // 8

for i in range(len(audio_highlight_starts_sec) - 1):

    audio1_start_sec = audio_highlight_starts_sec[i]
    audio2_start_sec = audio_highlight_starts_sec[i+1]
    audio_duration_sec = audio_highlight_duration_sec

    sample1_indices = [ sample_index for sample_index in range(int(audio1_start_sec * audio_sample_rate), int((audio1_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]
    sample2_indices = [ sample_index for sample_index in range(int(audio2_start_sec * audio_sample_rate), int((audio2_start_sec + audio_duration_sec) * audio_sample_rate), audio_waveform_offset)]

    audio1_encodings = encode_audio(audio_data, sample1_indices)
    audio2_encodings = encode_audio(audio_data, sample2_indices)

    mix_encodings = []

    for index in range(len(audio1_encodings)):
        mix_factor = index / (len(audio1_encodings) - 1)
        mix_encoding = audio1_encodings[index] * (1.0 - mix_factor) + audio2_encodings[index] * mix_factor
        mix_encodings.append(mix_encoding)
        
    gen_audio = decode_audio(mix_encodings, audio_waveform_offset)
    export_audio(torch.from_numpy(gen_audio), "results/audio/mix_audio_start1_{}_start2_{}_length_{}_epochs_{}.wav".format(audio1_start_sec, audio2_start_sec, audio_duration_sec, epochs))

