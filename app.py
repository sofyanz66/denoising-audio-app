# app.py (versi stabil untuk Python 3.11 dan TensorFlow 2.15)
import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import pywt
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from scipy.signal import butter, lfilter

# Konfigurasi halaman
st.set_page_config(page_title="Denoising Audio App", layout="centered")
st.title("🎧 Denoising Audio App - Multi Metode")

# Load model Autoencoder
@st.cache_resource
def load_autoencoder():
    return load_model("autoencoder_model.h5", compile=False)

model = load_autoencoder()

# Upload audio file
uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

# Fungsi High-Pass Filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=1000, fs=44100, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# PCEN
def pcen_denoise(signal, sr):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=2048, hop_length=512)
    pcen = librosa.pcen(mel * (2**31))
    inv = librosa.feature.inverse.mel_to_audio(pcen, sr=sr, n_fft=2048, hop_length=512)
    return inv

# STFT Masking
def stft_masking_denoise(signal, sr, threshold=0.01):
    stft = librosa.stft(signal)
    magnitude, phase = np.abs(stft), np.angle(stft)
    mask = magnitude > threshold
    filtered_mag = magnitude * mask
    stft_denoised = filtered_mag * np.exp(1j * phase)
    denoised = librosa.istft(stft_denoised)
    return denoised

# Wavelet + Autoencoder
def wavelet_autoencoder_denoise(signal):
    coeffs = pywt.wavedec(signal, 'db1', level=4)
    arr, slices = pywt.coeffs_to_array(coeffs)
    arr = arr / np.max(np.abs(arr))
    arr = arr.reshape(1, -1, 1)
    denoised_arr = model.predict(arr)
    denoised_arr = denoised_arr.reshape(-1)
    coeffs_denoised = pywt.array_to_coeffs(denoised_arr, slices, output_format='wavedec')
    reconstructed = pywt.waverec(coeffs_denoised, 'db1')
    return reconstructed

# Main proses
if uploaded_file is not None:
    st.audio(uploaded_file)
    signal, sr = sf.read(uploaded_file)

    method = st.selectbox(
        "Pilih Metode Noise Reduction",
        ["High-Pass Filter", "PCEN", "STFT Masking", "Wavelet + Autoencoder"]
    )

    if st.button("🧠 Proses Denoising"):
        with st.spinner("Memproses audio..."):
            if method == "High-Pass Filter":
                denoised = highpass_filter(signal, cutoff=1000, fs=sr)
            elif method == "PCEN":
                denoised = pcen_denoise(signal, sr)
            elif method == "STFT Masking":
                denoised = stft_masking_denoise(signal, sr)
            elif method == "Wavelet + Autoencoder":
                denoised = wavelet_autoencoder_denoise(signal)

        # Visualisasi
        fig, ax = plt.subplots()
        ax.plot(signal, alpha=0.5, label="Original")
        ax.plot(denoised, alpha=0.7, label="Denoised")
        ax.set_title(f"Hasil: {method}")
        ax.legend()
        st.pyplot(fig)

        # Simpan hasil
        os.makedirs("denoised_outputs", exist_ok=True)
        output_file = f"denoised_outputs/denoised_{method.replace(' ', '_')}.wav"
        sf.write(output_file, denoised, sr)

        # Unduh
        with open(output_file, "rb") as f:
            st.download_button(
                label="⬇️ Unduh Hasil",
                data=f,
                file_name=os.path.basename(output_file),
                mime="audio/wav"
            )
