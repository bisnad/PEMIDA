"""
Same as audio_vae_vocos_cnn_v3.5 but with perceptual loss instead of simple MSE less on mel spectra
"""

"""
Imports
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchaudio
import simpleaudio as sa
import numpy as np
import random
import glob
from matplotlib import pyplot as plt
import os, time
import json
import csv
import scipy.linalg as sclinalg
from sklearn.manifold import TSNE

from vocos import Vocos

import auraloss


"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_fps = 30

"""
Audio Settings
"""

"""
# Eleni

audio_file_path = "E:/Data/audio/Eleni/48khz/"
audio_files = ["4_5870821179501060412.wav"]
audio_valid_ranges = [[2.71, 825.75]]

# Diane
audio_file_path = "E:/Data/audio/Diane/48khz/"
audio_files = ["4d69949b.wav"]
audio_valid_ranges = [[5.0, 377.91]]

# Tim 
audio_file_path = "E:/Data/audio/Tim/48khz/"
audio_files = ["SajuHariPlacePrizeEntry2010.wav"]
audio_valid_ranges = [[0.0, 394.06]]
"""

audio_file_path = "E:/Data/audio/Eleni/48khz/"
audio_files = ["4_5870821179501060412.wav"]
audio_valid_ranges = [[2.71, 825.75]]


audio_sample_rate = 48000 # numer of audio samples per sec
audio_channels = 1

audio_window_length_vocos = 65280 # 256 mel frames worth of audio
audio_window_length_vae = 1792 # 8 mel frames worth of audio
audio_mel_count_vocos = None # will be calculated
audio_mel_count_vae = None
audio_window_offset = 960 # don't change if this vae will later be used for the motion2audio transformer model

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

set_all_seeds(42)

"""
Vocoder Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)

# freeze model parameters
for param in vocos.parameters():
    param.requires_grad = False

# determine number of mel spectra procuced by waveform of length audio_window_length_vocos
vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vocos), dtype=torch.float32).to(device))
audio_mel_count_vocos = vocoder_features.shape[-1]
audio_mel_filter_count = vocoder_features.shape[1]

print("audio_mel_count_vocos ", audio_mel_count_vocos, " audio_mel_filter_count ", audio_mel_filter_count)

# assert that the waveform length for vocos feature extraction is the same as the waveform length after vocos decodes the features
assert vocos.decode(vocoder_features).shape[-1] == audio_window_length_vocos, "the length of the waveform that vocos encodes into mels and decodes from mels must be identical"

# determine number of mel spectra procuced by waveform of length audio_window_length_vae
vocoder_features = vocos.feature_extractor(torch.rand(size=(1, audio_window_length_vae), dtype=torch.float32).to(device))
audio_mel_count_vae = vocoder_features.shape[-1]
audio_mel_filter_count = vocoder_features.shape[1]

print("audio_mel_count_vae ", audio_mel_count_vae, " audio_mel_filter_count ", audio_mel_filter_count)

#assert that the number of mels spectra produced by Vocos is an integer multiple of the number of mels consumed by the vae
assert audio_mel_count_vocos % audio_mel_count_vae == 0, "vocos mel count must be an integer multiple of vae mel count "

audio_vae_mels_per_vocos_mels = audio_mel_count_vocos // audio_mel_count_vae

# assert that the waveform length for vocos feature extraction is the same as the waveform length after vocos decodes the features
assert vocos.decode(vocoder_features).shape[-1] == audio_window_length_vae, "the length of the waveform that vocos encodes into mels and decodes from mels must be identical"

"""
VAE Model Settings
"""

latent_dim = 32
vae_conv_channel_counts = [ 16, 32, 64, 128 ]
vae_conv_kernel_size = (5, 3)
vae_dense_layer_sizes = [ 512 ]

"""
# Diane Version

encoder_weights_file = "results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"


# Eleni Version

encoder_weights_file = "results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "results_Eleni_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"


# Tim Version

encoder_weights_file = "results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "results_Tim_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"
"""

encoder_weights_file = "results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/encoder_weights_epoch_400"
decoder_weights_file = "results_Diane_audio_vae_vocos_cnn_ld32_kld0.1/weights/decoder_weights_epoch_400"


"""
Dataset Settings
"""

batch_size = 32
mocap_frame_incr = 4 


audio_waveform_length_per_mocap_frame = int(audio_sample_rate / mocap_fps)

"""
Load Audio
"""

audio_all_data = []

for audio_data_file, audio_valid_range in zip(audio_files, audio_valid_ranges):  
    
    print("audio_data_file ", audio_data_file)
    
    audio_data, _ = torchaudio.load(audio_file_path + audio_data_file)
  
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
Create Dataset
"""


audio_dataset = []

audio_waveform_start_offset = audio_window_length_vocos - audio_window_length_vae

for sI in range(len(audio_all_data)):
    
    with torch.no_grad():
        
        audio_data = audio_all_data[sI][0]
    
        print(sI)
        print("audio_data s ", audio_data.shape)
        
        for asI in range(audio_waveform_start_offset, audio_data.shape[0] - audio_window_length_vocos, audio_waveform_length_per_mocap_frame):
            
            audio_waveform_start = asI - audio_waveform_start_offset
            audio_waveform_end = audio_waveform_start + audio_window_length_vocos
            
            audio_waveform_excerpt = audio_data[audio_waveform_start:audio_waveform_end].unsqueeze(0).to(device)
            
            #print("asI ", asI, " audio_waveform_excerpt s ", audio_waveform_excerpt.shape)
            
            audio_mels_excerpt = vocos.feature_extractor(audio_waveform_excerpt)
            
            #print("asI ", asI, " audio_mels_excerpt s ", audio_mels_excerpt.shape)

            audio_dataset.append(audio_mels_excerpt.cpu())
 

audio_dataset = torch.cat(audio_dataset, dim=0)

print("audio_dataset s ", audio_dataset.shape)


"""
Create Models
"""

# create encoder model

class Encoder(nn.Module):
    
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

encoder = Encoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, vae_conv_channel_counts, vae_conv_kernel_size, vae_dense_layer_sizes).to(device)

print(encoder)

encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))

encoder.eval()
    
# Decoder 
class Decoder(nn.Module):
    
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
    
vae_conv_channel_counts_reversed = vae_conv_channel_counts.copy()
vae_conv_channel_counts_reversed.reverse()
    
vae_dense_layer_sizes_reversed = vae_dense_layer_sizes.copy()
vae_dense_layer_sizes_reversed.reverse()

vae_conv_channel_counts_reversed

decoder = Decoder(latent_dim, audio_mel_count_vae, audio_mel_filter_count, vae_conv_channel_counts_reversed, vae_conv_kernel_size, vae_dense_layer_sizes_reversed).to(device)

decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))

decoder.eval()

"""
Inference
"""

def export_audio(waveform, file_name):
    
    torchaudio.save("{}".format(file_name), waveform, audio_sample_rate)

def export_vocos_audio(waveform, file_name):
    
    waveform_length = waveform.shape[1]
    audio_window_offset = audio_window_length_vocos // 2
    audio_window_env = torch.hann_window(audio_window_length_vocos)
    
    audio_window_count = int(waveform_length - audio_window_length_vocos) // audio_window_offset
    pred_audio_sequence = torch.zeros((waveform_length), dtype=torch.float32)
    
    #print("pred_audio_sequence s ", pred_audio_sequence.shape)
    
    for i in range(audio_window_count):
        
        window_start = i * audio_window_offset
        window_end = window_start + audio_window_length_vocos
        
        target_audio = waveform[:, window_start:window_end]
        
        #print("i ", i, " target_audio s ", target_audio.shape)
        
        with torch.no_grad():
            audio_features = vocos.feature_extractor(target_audio.to(device))
            voc_audio = vocos.decode(audio_features).detach().cpu()

        #print("voc_audio s ", voc_audio.shape)
        #print("grain_env s ", grain_env.shape)

        pred_audio_sequence[i*audio_window_offset:i*audio_window_offset + audio_window_length_vocos] += voc_audio[0] * audio_window_env

    torchaudio.save("{}".format(file_name), torch.reshape(pred_audio_sequence, (1, -1)), audio_sample_rate)


def encode_audio(waveform):
    
    encoder.eval()
    
    with torch.no_grad():

        waveform = waveform.to(device)
 
        y_mels = vocos.feature_extractor(waveform.to(device))
        mel_count = y_mels.shape[-1]
        mel_count = (mel_count // audio_mel_count_vae) * audio_mel_count_vae
        y_mels = y_mels[:, :, -mel_count:]
        y_mels = y_mels.reshape((1, audio_mel_filter_count, -1, audio_mel_count_vae))
        y_mels = y_mels.permute((2, 0, 1, 3))

        latent_vectors = []
        
        for i in range(mel_count // audio_mel_count_vae):
            
            #print("i ", i)
            
            audio_encoder_in = y_mels[i, ...].unsqueeze(0)
            
            #print("audio_encoder_in s ", audio_encoder_in.shape)
            
            audio_encoder_out_mu, audio_encoder_out_std = encoder(audio_encoder_in)
            mu = audio_encoder_out_mu
            std = torch.nn.functional.softplus(audio_encoder_out_std) + 1e-6
            audio_encoder_out = encoder.reparameterize(mu, std)
            
            #print("audio_encoder_out s ", audio_encoder_out.shape)

            latent_vector = audio_encoder_out.detach().cpu().numpy()
        
            latent_vectors.append(latent_vector)
            
    encoder.train()
    
    return latent_vectors
  
def decode_audio_encodings(encodings, file_name):
    
    decoder.eval()
    
    with torch.no_grad():
        
        encoding_count = len(encodings)
        
        yhat_mels_all = []
        
        for i in range(encoding_count):
            
            encoding = torch.Tensor(encodings[i]).to(device)
            yhat_mels = decoder(encoding)
            
            #print("yhat_mels s ", yhat_mels.shape)
            
            yhat_mels = yhat_mels.reshape(audio_mel_filter_count, audio_mel_count_vae).detach()
            
            #print("yhat_mels 2 s ", yhat_mels.shape)
            
            yhat_mels_all.append(yhat_mels)
            
        yhat_mels_all = torch.cat(yhat_mels_all, dim=1).unsqueeze(0)
        
        #print("yhat_mels_all s ", yhat_mels_all.shape)
         
        yhat_audio = vocos.decode(yhat_mels_all).detach().cpu()
        
        #print("yhat_audio s ", yhat_audio.shape)

    
    torchaudio.save("{}".format(file_name), torch.reshape(yhat_audio, (1, -1)), audio_sample_rate)

    
    decoder.train()

@torch.no_grad()
def create_audio_space_representation(sequence_excerpts):

    encodings = []
    
    excerpt_count = sequence_excerpts.shape[0]
    
    for eI in range(0, excerpt_count, batch_size):
        
        excerpt = sequence_excerpts[eI:eI+batch_size]
 
        excerpt = torch.from_numpy(excerpt).to(device)
        
        excerpt = excerpt.unsqueeze(1)

        # use only the last mels
        excerpt = excerpt[:,:,:, -audio_mel_count_vae:]
  
        encoder_output_mu, encoder_output_std = encoder(excerpt)
        mu = encoder_output_mu
        std = torch.nn.functional.softplus(encoder_output_std) + 1e-8
        latent_vector = encoder.reparameterize(mu, std)

        latent_vector = latent_vector.detach().cpu()

        encodings.append(latent_vector)
        
    encodings = torch.cat(encodings, dim=0)
    
    print("encodings s ", encodings.shape)
    
    encodings = encodings.numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, max_iter=5000, verbose=1)    
    Z_tsne = tsne.fit_transform(encodings)
    
    return Z_tsne

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

def create_audio_space_image(Z_tsne, audio_highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]
    
    plot_colors_hsv = distinct_hsv_colors(len(audio_highlight_excerpt_ranges))
    plot_colors = [ hsv_to_rgb(hsv)  for hsv in plot_colors_hsv ]
    
    plt.figure()
    fig, ax = plt.subplots()
    #ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.2)
        
    for hI, hR in enumerate(audio_highlight_excerpt_ranges):
        
        #ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], marker='o', facecolors="none", s=10.0, linewidths=0.4, edgecolors=plot_colors[hI], alpha=0.5)
        
        print("hR", hR, " hI ", hI, " hR[0]:hR[1] ", hR[0], ":", hR[1], " Z_tsne_x ", Z_tsne_x.shape, " color ", plot_colors[hI])
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=600)
    plt.close()

"""
Create Audio Space Plot
"""

"""
# Diane Version

audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]

# Eleni Version

audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]

# Tim Version

audio_highlight_starts_sec = [5.0, 25.0, 54.0, 90.0, 106.0, 226.0, 290.0, 320.0]

"""

audio_highlight_starts_sec = [20.0, 130.0, 240.0, 360.0, 480.0, 540.0, 660.0, 780.0]
audio_highlight_duration_sec = 10.0

audio_waveform_offset = int(mocap_frame_incr / mocap_fps * audio_sample_rate)

audio_highlight_ranges = []

for audio_highlight_start_sec in audio_highlight_starts_sec:
    audio_highlight_ranges.append([ int(audio_highlight_start_sec * audio_sample_rate) // audio_waveform_offset, int((audio_highlight_start_sec + audio_highlight_duration_sec) * audio_sample_rate) // audio_waveform_offset ])

Z_tsne = create_audio_space_representation(audio_dataset.numpy())
create_audio_space_image(Z_tsne, audio_highlight_ranges, "audio_space_plot.png")

# save orig and vocos waveforms

test_waveform = audio_all_data[0]
test_duration_sec = 10.0

for audio_start_sec in audio_highlight_starts_sec:
    
    start_time_frames = int(audio_start_sec * audio_sample_rate)
    end_time_frames = int((audio_start_sec + test_duration_sec) * audio_sample_rate)
    
    export_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/orig_audio_audio_{}-{}.wav".format(audio_start_sec, test_duration_sec))
    export_vocos_audio(test_waveform[:, start_time_frames:end_time_frames], "results/audio/vocos_audio_audio_{}-{}.wav".format(audio_start_sec, test_duration_sec))
    
    latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])
    decode_audio_encodings(latent_vectors, "results/audio/rec_audio_audio_{}-{}.wav".format(audio_start_sec, test_duration_sec))
    

# reconstruct original waveform

test_waveform = audio_all_data[0]
test_duration_sec = 10.0

for audio_start_sec in audio_highlight_starts_sec:
    
    start_time_frames = int(audio_start_sec * audio_sample_rate)
    end_time_frames = int((audio_start_sec + test_duration_sec) * audio_sample_rate)
    

    latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])
    decode_audio_encodings(latent_vectors, "results/audio/rec_audio_audio_{}-{}.wav".format(audio_start_sec, test_duration_sec))
    

# random perturbation

test_start_time_sec = 20
test_duration_sec = 25
perturbation_sizes = [ 0.0, 0.1, 0.2, 0.5, 1.0 ]

for audio_start_sec in audio_highlight_starts_sec:

    start_time_frames = int(audio_start_sec * audio_sample_rate)
    end_time_frames = int((audio_start_sec + test_duration_sec) * audio_sample_rate)
    latent_vectors = encode_audio(test_waveform[:, start_time_frames:end_time_frames])
    
    z = torch.randn((1, latent_dim))  # or your encoded latent
    
    perturbed_encodings = []
    
    for index in range(len(latent_vectors)):
        
        #print("index ", index)
        
        sigma = perturbation_sizes[ min(index // (len(latent_vectors) // len(perturbation_sizes)), len(perturbation_sizes) - 1) ]
        
        #print("index ", index, " sigma ", sigma)
        
        epsilon = torch.randn_like(z) * sigma  # ~ N(0, sigma^2 I)
        
        #print("epsilon s ", epsilon.shape)
        #print("latent_vectors[index] s ", latent_vectors[index].shape)
        
        latent_vector_perturbed = latent_vectors[index] + epsilon.numpy()
    
        perturbed_encodings.append(latent_vector_perturbed)
        
    decode_audio_encodings(perturbed_encodings, "results/audio/perturb_audio_audio_{}-{}.wav".format(audio_start_sec, test_duration_sec))
    

# interpolate two original sequences

test_waveform = audio_all_data[0]
test_duration_sec = 25.0

for i in range(len(audio_highlight_starts_sec) - 1):

    test1_start_time_sec = audio_highlight_starts_sec[i]
    test2_start_time_sec = audio_highlight_starts_sec[i + 1]

    start1_time_frames = int(test1_start_time_sec * audio_sample_rate)
    end1_time_frames = int(start1_time_frames + test_duration_sec * audio_sample_rate)
    
    start2_time_frames = int(test2_start_time_sec * audio_sample_rate)
    end2_time_frames = int(start2_time_frames + test_duration_sec * audio_sample_rate)
    
    latent_vectors_1 = encode_audio(test_waveform[:, start1_time_frames:end1_time_frames])
    latent_vectors_2 = encode_audio(test_waveform[:, start2_time_frames:end2_time_frames])
    
    
    mix_encodings = []
    
    for index in range(len(latent_vectors_1)):
        mix_factor = index / (len(latent_vectors_1) - 1)
        mix_encoding = latent_vectors_1[index] * (1.0 - mix_factor) + latent_vectors_2[index] * mix_factor
        mix_encodings.append(mix_encoding)
    
    decode_audio_encodings(mix_encodings, "results/audio/mix_audio_audio1_{}-{}_audio2_{}-{}.wav".format(test1_start_time_sec, test1_start_time_sec + test_duration_sec, test2_start_time_sec, test2_start_time_sec + test_duration_sec))
    


