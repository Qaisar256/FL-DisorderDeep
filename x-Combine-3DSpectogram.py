import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.signal import cwt, ricker
from scipy.signal import fftconvolve
# Ensure you have the required directories or create them
import os

# Directory to save plots
plot_save_dir = 'C:\\Users\\User\\Desktop\\Automated Lung Sound Classification\\sample-dataset'
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)

# Load an audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Save the raw audio waveform as an image
def save_waveform(audio, sr, filename='waveform.png'):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Raw Audio Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(plot_save_dir, filename))
    plt.close()

# Save a Spectrogram as an image
def save_spectrogram(audio, sr, filename='spectrogram.png'):
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(plot_save_dir, filename))
    plt.close()

# Save a Scalogram as an image
def save_scalogram(audio, sr, filename='scalogram.png'):
    widths = np.arange(1, 128)
    cwtmatr = cwt(audio, ricker, widths)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')    
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(cwtmatr), extent=[0, len(audio)/sr, 1, 128], cmap='PRGn', aspect='auto')
    plt.title('Scalogram')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Scale')
    plt.savefig(os.path.join(plot_save_dir, filename))
    plt.close()
def make_erb_filters(fs, num_channels, low_freq=20):
    """Generate a bank of Gammatone filters using the Equivalent Rectangular Bandwidth (ERB) scale."""
    # Ear parameters
    ear_q = 9.26449  # Glasberg and Moore Parameters
    min_bw = 24.7
    order = 1
    
    # Compute ERB and generate filters
    max_freq = fs / 2
    erb = ((np.linspace(1, num_channels, num_channels) - 1) * (ear_q * min_bw) + 
           ((low_freq)**order + min_bw**order)**(1/order))**(1/order)
    center_freq = erb
    
    filter_length = int(fs / low_freq * 2)
    filters = np.zeros((num_channels, filter_length))
    
    for j in range(num_channels):
        f = center_freq[j]
        l = np.arange(0, filter_length) / fs
        filters[j, :] = l**(order - 1) * np.exp(-2 * np.pi * erb[j] * l) * np.cos(2 * np.pi * f * l)
        return filters, center_freq

# Placeholder for saving a Gammatone Spectrogram as an image
def save_gammatone_spectrogram(audio, sr, filename='gammatone_spectrogram.png', num_filters=64, low_freq=50):
    """Generate and save a more detailed Gammatone Spectrogram."""
    filters, center_freq = make_erb_filters(sr, num_filters, low_freq)
    gammatone_spectrogram = np.zeros((num_filters, len(audio)))
    
    for i, filter in enumerate(filters):
        filtered_signal = fftconvolve(audio, filter, mode='same')
        gammatone_spectrogram[i, :] = 10 * np.log10(np.abs(filtered_signal)**2)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(gammatone_spectrogram, aspect='auto', origin='lower', 
               extent=[0, len(audio)/sr, low_freq, sr / 2], cmap='jet')
    plt.title('Gammatone Spectrogram with Improved Resolution')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(os.path.join(plot_save_dir, filename))
    plt.close()
def save_mfsc(audio, sr, filename='mfsc.png', n_mels=40):
    """Calculate and save a plot of the Mel-Frequency Spectral Coefficients (MFSC)."""
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Frequency Spectral Coefficients (MFSC)')
    plt.savefig(os.path.join(plot_save_dir, filename))
    plt.close()
# Main
if __name__ == '__main__':
    # Load your audio file here
    audio_file_path = 'C:\\Users\\User\\Desktop\\Automated Lung Sound Classification\\sample-dataset\\URTI.mp3'  # Update this path
    audio, sr = load_audio(audio_file_path)
    save_waveform(audio, sr, filename='8URTI.png')
    save_spectrogram(audio, sr, filename='8spectogram.png')
    save_scalogram(audio, sr, filename='8scalogram.png')
    # Implement and call save_gammatone_spectrogram() when ready
    save_gammatone_spectrogram(audio, sr, filename='8gammatone.png')
    save_mfsc(audio, sr, filename='8mfsc.png')
# Combining features into a 3D stack
fs=sr
signal=audio
mel = librosa.feature.melspectrogram(y=audio, sr=fs , n_mels=128)
# Convert power spectrogram to dB scale (logarithmic)
log_mel_spectrogram = librosa.power_to_db(mel, ref=np.max)
# Define a common hop_length
hop_length = 512  # Common hop length
# Compute the Mel spectrogram
mel = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=128, hop_length=hop_length)
# Compute MFCCs
mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=128, hop_length=hop_length, n_mels=128)
# Compute Chroma features
chroma = librosa.feature.chroma_stft(y=signal, sr=fs, hop_length=hop_length, n_chroma=128)
max_time = max(mel.shape[1], mfccs.shape[1], chroma.shape[1])
# Function to resize features
def resize_feature(feature, target_size):
    # Linear interpolation for resizing
    feature_resized = np.zeros((feature.shape[0], target_size))
    for i in range(feature.shape[0]):
        x_old = np.linspace(0, feature.shape[1] - 1, feature.shape[1])
        x_new = np.linspace(0, feature.shape[1] - 1, target_size)
        feature_resized[i, :] = np.interp(x_new, x_old, feature[i, :])
    return feature_resized
mel_resized = resize_feature(mel, max_time)
mfccs_resized = resize_feature(mfccs, max_time)
chroma_resized = resize_feature(chroma, max_time)
# Stack the features
features_stack1 = np.stack([mel_resized, mfccs_resized, chroma_resized], axis=-1)
# Visualization
plt.imshow(features_stack1, aspect='auto')  # This makes 'features_stack' a mappable object
plt.title('Shape of the combined feature stack')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
# Create the colorbar with a format specification
cbar = plt.colorbar(format='%+2.0f dB')
cbar.set_label('Decibel (dB)')
# Save the plot
plt.savefig(os.path.join(plot_save_dir, '8combine.png'))
plt.close()