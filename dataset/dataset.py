import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample
from datasets import Dataset
from huggingface_hub import login

# Log in using your specific access token
login("token")
# ==========================================
# 1. Signal Processing Functions
# ==========================================

def apply_butterworth_filter(emg_signal, fs=2000):
    """
    Applies a 4th-order bandpass (74.5-149.5 Hz) and a 50 Hz notch filter.
    Forces output to float32 precision.
    """
    # Ensure input is float32
    emg_signal = emg_signal.astype(np.float32)
    nyq = 0.5 * fs
    
    # Bandpass Filter (74.5 - 149.5 Hz)
    low = 74.5 / nyq
    high = 149.5 / nyq
    b_band, a_band = butter(4, [low, high], btype='band')
    
    # filtfilt returns float64 by default, cast back to float32 immediately
    filtered_emg = filtfilt(b_band, a_band, emg_signal, axis=0).astype(np.float32)
    
    # Notch Filter (50 Hz)
    b_notch, a_notch = iirnotch(50 / nyq, Q=30)
    filtered_emg = filtfilt(b_notch, a_notch, filtered_emg, axis=0).astype(np.float32)
    
    return filtered_emg

def resample_signal(emg_signal, orig_fs=2000, target_fs=1000):
    """
    Resamples the signal to match the frequency of the target sensor.
    """
    if orig_fs == target_fs:
        return emg_signal.astype(np.float32)
    
    num_samples = int(emg_signal.shape[0] * (target_fs / orig_fs))
    resampled_emg = resample(emg_signal, num_samples, axis=0).astype(np.float32)
    
    return resampled_emg

def create_sliding_windows(emg_signal, window_samples, step_samples):
    """
    Chunks the continuous signal into sliding windows and flattens channels.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    windows = sliding_window_view(emg_signal, window_shape=window_samples, axis=0)
    windows = windows[::step_samples, :, :]
    
    num_windows, num_channels, w_samples = windows.shape
    flattened_windows = windows.reshape(num_windows * num_channels, w_samples)
    
    # Final safety cast
    return flattened_windows.astype(np.float32)

# ==========================================
# 2. Plotting Function
# ==========================================

def save_filter_comparison_plot(raw_signal, filtered_signal, save_path, fs=2000, channel_idx=0, num_samples=2000):
    """
    Saves a matplotlib figure comparing the raw and filtered signal.
    Plots exactly 1 second of data (num_samples = fs) for clear visualization.
    """
    time = np.arange(num_samples) / fs
    
    plt.figure(figsize=(12, 6))
    
    # Plot Raw (Before)
    plt.subplot(2, 1, 1)
    plt.plot(time, raw_signal[:num_samples, channel_idx], color='red', alpha=0.8)
    plt.title(f"Before Filtering: Raw EMG (Channel {channel_idx + 1})")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Filtered (After)
    plt.subplot(2, 1, 2)
    plt.plot(time, filtered_signal[:num_samples, channel_idx], color='blue', alpha=0.8)
    plt.title(f"After Filtering: 4th Order Bandpass (74.5-149.5 Hz) + 50Hz Notch")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved successfully to: {save_path}")

# ==========================================
# 3. File and Pipeline Processing
# ==========================================

def process_mat_file(file_path, orig_fs=2000, target_fs=1000, window_ms=250, overlap_ms=125, plot_save_path=None):
    """
    Full pipeline for a single .mat file. 
    If plot_save_path is provided, it will generate the comparison plot before windowing.
    """
    mat_data = loadmat(file_path)
    emg_raw = mat_data['emg'].astype(np.float32)
    
    # 1. Filter
    emg_filtered = apply_butterworth_filter(emg_raw, fs=orig_fs)
    
    # 2. Plotting (Triggered only if a path is passed)
    if plot_save_path:
        save_filter_comparison_plot(emg_raw, emg_filtered, plot_save_path, fs=orig_fs)
        
    # 3. Resample
    emg_resampled = resample_signal(emg_filtered, orig_fs=orig_fs, target_fs=target_fs)
    
    # 4. Windowing
    window_samples = int((window_ms / 1000) * target_fs)
    step_samples = int(((window_ms - overlap_ms) / 1000) * target_fs)
    windows = create_sliding_windows(emg_resampled, window_samples, step_samples)
    
    return windows

def emg_data_generator(data_dir):
    """
    Generator function that finds all .mat files and yields 1D arrays one by one.
    """
    search_pattern = os.path.join(data_dir, "**", "*.mat")
    mat_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(mat_files)} .mat files. Starting processing...")
    
    # Flag to ensure we only plot the very first file
    has_plotted = False
    
    for file_path in mat_files:
        try:
            # Set path for the first file, None for the rest
            plot_path = "emg_filter_comparison.png" if not has_plotted else None
            
            windows = process_mat_file(
                file_path, 
                orig_fs=2000, 
                target_fs=1000, 
                window_ms=250, 
                overlap_ms=125,
                plot_save_path=plot_path
            )
            
            if not has_plotted:
                has_plotted = True
            
            for window in windows:
                yield {"emg_signal": window}
                
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

# ==========================================
# 4. Execution 
# ==========================================

def main():
    root_data_dir = "/teamspace/studios/this_studio/Data/"
    
    print("Building Hugging Face Dataset stream...")
    hf_dataset = Dataset.from_generator(lambda: emg_data_generator(root_data_dir))
    
    print(f"Dataset generated! Total 1D samples: {len(hf_dataset)}")
    hf_dataset.push_to_hub("Codemaster67/ninapro-db2-1d-pretrain")

if __name__ == "__main__":
    main()
