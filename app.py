# Save this code as app.py
import streamlit as st
import numpy as np
import librosa
import PyWavelets
import tensorflow as tf
from tensorflow.keras import layers, models
import soundfile as sf
import scipy.signal
import os # To handle file uploads

# Define the functions used in the notebook
# ===== Wavelet + Autoencoder functions =====

def wavelet_decompose(signal, wavelet='db8', level=4):
    # Ensure signal is a numpy array and handle potential non-finite values
    signal = np.asarray(signal)
    signal = signal[np.isfinite(signal)]
    if signal.size == 0:
        return [np.array([])] # Return empty coefficients if signal is empty
    try:
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        # Ensure all coefficients are finite
        coeffs = [np.asarray(c) for c in coeffs]
        coeffs = [c[np.isfinite(c)] for c in coeffs]
        return coeffs
    except ValueError as e:
        st.error(f"Error during wavelet decomposition: {e}")
        return [np.array([])]


def coeffs_to_vector(coeffs):
    # Filter out empty arrays before concatenation
    valid_coeffs = [c for c in coeffs if c.size > 0]
    if not valid_coeffs:
        return np.array([])
    return np.concatenate(valid_coeffs)

def vector_to_coeffs(vector, coeffs_template):
    split_coeffs = []
    idx = 0
    # Need to be careful if coeffs_template contains empty arrays
    template_sizes = [len(c) for c in coeffs_template]
    for size in template_sizes:
        if idx + size > len(vector):
             # This can happen if the vector is shorter than expected
             # due to filtering out non-finite values or errors
             st.warning(f"Vector length mismatch during reconstruction. Expected at least {idx + size}, got {len(vector)}. Appending empty array.")
             split_coeffs.append(np.array([]))
             break # Stop processing if vector is exhausted
        split_coeffs.append(vector[idx:idx+size])
        idx += size
    # If we stopped early, pad with empty arrays to match template structure
    while len(split_coeffs) < len(coeffs_template):
         split_coeffs.append(np.array([]))
    return split_coeffs


def build_autoencoder(input_dim):
    if input_dim <= 0:
        st.error(f"Invalid input dimension for autoencoder: {input_dim}")
        return None
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(encoded)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)
    model = models.Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

# Placeholder for loading the trained autoencoder model
@st.cache_resource # Cache the model loading
def load_autoencoder_model(model_path="autoencoder_model.h5"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure 'autoencoder_model.h5' is in the same directory.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None


def denoise_with_wavelet_autoencoder(signal, sr, autoencoder_model):
    if autoencoder_model is None:
        return None

    # Use the same wavelet and level as used for training
    wavelet = 'db8'
    level = 4

    try:
        coeffs = wavelet_decompose(signal, wavelet=wavelet, level=level)
        if not coeffs or all(c.size == 0 for c in coeffs):
             st.warning("Wavelet decomposition resulted in empty coefficients.")
             return np.array([])

        coeff_vector = coeffs_to_vector(coeffs)

        if coeff_vector.size == 0:
             st.warning("Coefficient vector is empty.")
             return np.array([])

        # Ensure the input dimension matches the model's expected input shape
        model_input_shape = autoencoder_model.input_shape[1] # Get input dimension excluding batch size
        if coeff_vector.shape[0] != model_input_shape:
            st.error(f"Input dimension mismatch. Model expects {model_input_shape}, but got {coeff_vector.shape[0]}. Cannot denoise.")
            return None # Return None or original signal if dimensions don't match

        # Predict denoised coefficients
        denoised_vector = autoencoder_model.predict(np.expand_dims(coeff_vector, axis=0))[0]

        # Reconstruct signal from denoised coefficients
        reconstructed_coeffs = vector_to_coeffs(denoised_vector, coeffs)

        # Need to handle potential size differences in reconstructed_coeffs vs original coeffs structure
        # Simple check:
        if len(reconstructed_coeffs) != len(coeffs):
             st.warning("Number of reconstructed coefficient arrays does not match original.")
             # Attempt to reconstruct with available coeffs, or return original signal
             return pywt.waverec(reconstructed_coeffs, wavelet=wavelet) if reconstructed_coeffs and all(c.size > 0 for c in reconstructed_coeffs) else signal


        denoised_signal = pywt.waverec(reconstructed_coeffs, wavelet=wavelet)

        # Ensure the denoised signal has the same length as the original signal
        # This can sometimes be an issue with wavelet reconstruction
        if len(denoised_signal) != len(signal):
            st.warning(f"Denoised signal length ({len(denoised_signal)}) mismatch with original ({len(signal)}). Resampling.")
            # Resample to original length
            denoised_signal = scipy.signal.resample(denoised_signal, len(signal))


        return denoised_signal

    except Exception as e:
        st.error(f"An error occurred during wavelet-autoencoder denoising: {e}")
        return None


# Fungsi high-pass filter
def high_pass_filter(audio, sr, cutoff_freq=300):
    # Ensure audio is a numpy array and handle potential non-finite values
    audio = np.asarray(audio)
    audio = audio[np.isfinite(audio)]
    if audio.size == 0:
        return np.array([])
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    if normal_cutoff >= 1:
        st.warning(f"High-pass cutoff frequency {cutoff_freq} Hz is at or above Nyquist frequency {nyquist} Hz. No filtering applied.")
        return audio
    try:
        b, a = scipy.signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered_audio = scipy.signal.filtfilt(b, a, audio)
        return filtered_audio
    except Exception as e:
        st.error(f"Error during high-pass filtering: {e}")
        return np.array([])


# Fungsi autoencoder (keeping the definition here but we'll load the model)
# def build_autoencoder(input_dim):
#     input_layer = layers.Input(shape=(input_dim,))
#     encoded = layers.Dense(128, activation='relu')(input_layer)
#     encoded = layers.Dense(64, activation='relu')(encoded)
#     decoded = layers.Dense(128, activation='relu')(encoded)
#     output_layer = layers.Dense(input_dim, activation='linear')(decoded)
#     model = models.Model(input_layer, output_layer)
#     model.compile(optimizer='adam', loss='mse')
#     return model

# Fungsi untuk PCEN (Perceptual Cepstral Normalization) - Placeholder
def pcen(audio):
    # Placeholder PCEN function (use an actual implementation in practice)
    # librosa.effects.preemphasis might introduce NaNs if input has NaNs
    audio = np.asarray(audio)
    audio = audio[np.isfinite(audio)]
    if audio.size == 0:
        return np.array([])
    try:
        return librosa.effects.preemphasis(audio)
    except Exception as e:
        st.error(f"Error during PCEN (preemphasis): {e}")
        return np.array([])

# Fungsi noise reduction menggunakan STFT masking (Optional to include)
def removeNoise_stft(audio, sr, visual=False):
    # Ensure audio is a numpy array and handle potential non-finite values
    audio = np.asarray(audio)
    audio = audio[np.isfinite(audio)]
    if audio.size == 0:
        return np.array([])
    try:
        D = librosa.stft(audio)
        D_mag, D_phase = librosa.magphase(D)
        mean_mag = np.mean(D_mag, axis=1, keepdims=True)
        std_mag = np.std(D_mag, axis=1, keepdims=True)

        # Masking
        noise_mask = D_mag > (mean_mag + 2 * std_mag)
        D_clean = D * noise_mask

        # Reconstruct the audio signal from clean STFT
        cleaned_audio = librosa.istft(D_clean)
        return cleaned_audio
    except Exception as e:
        st.error(f"Error during STFT noise reduction: {e}")
        return np.array([])


# --- Streamlit App ---

st.title("Pengurangan Noise Audio dengan Wavelet Autoencoder")
st.write("Unggah file audio (.wav) untuk mengurangi noise menggunakan kombinasi Wavelet Transform dan Autoencoder.")

uploaded_file = st.file_uploader("Pilih file audio (.wav)...", type=["wav"])

# Load the pre-trained autoencoder model
autoencoder_model = load_autoencoder_model()

if uploaded_file is not None:
    # To read the audio file, we need to save it temporarily
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Load the audio file
        y, sr = librosa.load("uploaded_audio.wav", sr=16000) # Use consistent sample rate

        st.subheader("Audio Asli")
        st.audio(y, sample_rate=sr)

        # --- Apply Denoising Process ---

        processed_audio = y # Start with original audio

        # Option to apply High-pass filter
        apply_hp = st.checkbox("Terapkan High-pass Filter (cutoff 300Hz)", value=False)
        if apply_hp:
            processed_audio = high_pass_filter(processed_audio, sr)
            if processed_audio is None or processed_audio.size == 0:
                 st.warning("High-pass filtering failed or resulted in empty audio.")
                 processed_audio = y # Revert to original if filtering fails
            else:
                 st.write("High-pass filter applied.")


        # Option to apply PCEN
        apply_pcen = st.checkbox("Terapkan PCEN (Placeholder)", value=False)
        if apply_pcen:
            processed_audio = pcen(processed_audio)
            if processed_audio is None or processed_audio.size == 0:
                 st.warning("PCEN failed or resulted in empty audio.")
                 processed_audio = y # Revert to original if PCEN fails
            else:
                 st.write("PCEN applied (Placeholder).")


        # Option to apply STFT Masking Noise Reduction
        apply_stft_denoise = st.checkbox("Terapkan Noise Reduction (STFT Masking)", value=False)
        if apply_stft_denoise:
             processed_audio = removeNoise_stft(processed_audio, sr)
             if processed_audio is None or processed_audio.size == 0:
                 st.warning("STFT denoising failed or resulted in empty audio.")
                 processed_audio = y # Revert to original if STFT denoising fails
             else:
                 st.write("STFT Masking Noise Reduction applied.")


        # Apply Wavelet Autoencoder Denoising
        st.subheader("Hasil Pengurangan Noise (Wavelet Autoencoder)")
        if autoencoder_model is not None:
            denoised_audio_wa = denoise_with_wavelet_autoencoder(processed_audio, sr, autoencoder_model)

            if denoised_audio_wa is not None and denoised_audio_wa.size > 0:
                st.audio(denoised_audio_wa, sample_rate=sr)

                # Optional: Save the denoised audio
                # if st.button("Unduh Audio Hasil Denoising"):
                #     sf.write("denoised_audio.wav", denoised_audio_wa, sr)
                #     with open("denoised_audio.wav", "rb") as f:
                #          st.download_button(label="Klik untuk Mengunduh", data=f, file_name="denoised_audio.wav", mime="audio/wav")

            elif denoised_audio_wa is not None and denoised_audio_wa.size == 0:
                 st.warning("Wavelet Autoencoder denoising resulted in an empty audio signal.")
            else:
                 st.error("Gagal menerapkan Wavelet Autoencoder denoising.")

        else:
            st.warning("Model Autoencoder belum dimuat. Pastikan file 'autoencoder_model.h5' ada.")


    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file audio: {e}")

    # Clean up the temporary file
    if os.path.exists("uploaded_audio.wav"):
        os.remove("uploaded_audio.wav")
