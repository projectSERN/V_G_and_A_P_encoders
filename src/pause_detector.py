from typing import List
import numpy as np
import pywt
from scipy.signal import butter, filtfilt
from moviepy import VideoFileClip


class PauseDetector:
    def __init__(self) -> None:
        pass

    def __dwt(self, signal: np.ndarray, wavelet: str = "db4") -> np.ndarray:
        """
        Applies the Discrete Wavelet Transform (DWT) to a signal.

        Args:
        - signal (np.ndarray): The signal to apply the DWT to
        - wavelet (str): The wavelet to use

        Returns:
        - np.ndarray: The envelope created from the signal after DWT
        """
        coefficients = pywt.wavedec(signal, wavelet, level=5)

        # Use approximation coefficients (low-freq components) to reconstruct the signal
        approximation = coefficients[0]

        # Reconstruct the envelope
        envelope = np.abs(approximation)

        # Upsample to match the original signal length
        envelope_resampled = np.interp(
            np.linspace(0, len(envelope), len(signal)), np.arange(len(envelope)), envelope)

        return envelope_resampled


    def __lowpass_filter(self, signal: np.ndarray, cutoff: int, sampling_freq: int, order: int = 4
                         ) -> np.ndarray:
        """
        Applies a low-pass filter to a signal.

        Arg(s):
        - signal (np.ndarray): The signal to apply the low-pass filter to
        - cutoff (int): The cutoff frequency of the filter
        - sampling_freq (int): The sampling frequency of the signal
        - order (int): The order of the filter
        """
        # Normalize the cutoff frequency
        nyquist = 0.5 * sampling_freq

        # Design the filter
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        return filtfilt(b, a, signal)
    

    def __bandpass_filter(self, signal: np.ndarray, lowcut: int, highcut: int, sampling_freq: int, order: int = 5) -> np.ndarray:
        """
        Applies a band-pass filter to a signal.

        Arg(s):
        - signal (np.ndarray): The signal to apply the band-pass filter to
        - lowcut (int): The low cutoff frequency of the filter
        - highcut (int): The high cutoff frequency of the filter
        - sampling_freq (int): The sampling frequency of the signal
        - order (int): The order of the filter
        """
        nyquist = 0.5 * sampling_freq
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)


    def __extract_audio(self, filepath: str, sampling_freq: int = 22050) -> np.ndarray:
        """
        Extracts an audio from video with the specified filepath.
        The extracted audio is also subclipped to make the duration of the audio an exact integer in seconds.

        Arg(s):
        - filepath (str): The video filepath to extract an audio from

        Returns:
        - ArrayLike: The time-series audio array
        """
        # Extract audio
        video = VideoFileClip(filename=filepath)
        audio = video.audio
        if audio is None:
            return None

        # Get duration of audio and clip the audio
        duration = int(audio.duration)
        audio = audio.subclipped(0, duration)

        # Resample when converting to sound array
        audio_array = audio.to_soundarray(fps=sampling_freq)

        # Convert from two-channel stereo to mono
        audio_array = np.mean(audio_array, axis=1)

        # Close audio and video to prevent any leakage
        audio.close()
        video.close()

        return audio_array


    def __envelope_detection(self, signal: np.ndarray, sampling_freq: int = 22050
                      ) -> np.ndarray:
        """
        Outputs the smoothed envelope of the audio signal extracted from a video
        using the Discrete Wavelet Transform (DWT).

        Args:
        - signal (np.ndarray): The audio signal to detect pauses in
        - threshold (float): The threshold to consider a pause

        Returns:
        - np.ndarray: The indices of the pauses in the audio signal
        """
        # Extract audio from the video
        #audio_signal = self.__extract_audio(video_path, sampling_freq)

        # Apply a bandpass filter on the signal
        bandpass_filtered_signal = self.__bandpass_filter(signal, lowcut=80, highcut=3000, sampling_freq=sampling_freq)

        # Take the absolute values of the signal
        amplitude = np.abs(bandpass_filtered_signal)

        # Apply the Discrete Wavelet Transform (DWT) to the signal
        envelope = self.__dwt(amplitude)

        # Apply a low-pass filter to the envelope
        smoothed_envelope = self.__lowpass_filter(envelope, cutoff=4, sampling_freq=sampling_freq)

        return smoothed_envelope

    
    def __low_amplitude_masking(self, signal: np.ndarray, sampling_rate: int,
                                amplitude_threshold: float = 0.01, time_frame_threshold: float = 0.1
                                ) -> tuple[List[tuple[float, float]], np.ndarray]:
        """
        Masks the sections of the audio signal that are below the defined amplitude threshold.

        Arg(s):
        - signal (np.ndarray): The audio signal to detect pauses in
        - sampling_rate (int): The sampling rate of the audio signal
        - amplitude_threshold (float): The threshold beneath which a signal will be said to have a pause
        - time_frame_threshold (float): The minimum length a section must be to be considered a pause (in seconds)

        Returns:
        - tuple[List[(float, float)], np.ndarray]: The list contains the start and end times for each pause. The ArrayLike object contains an array of 0's and 1's
        """
        # Get absolute value of signal
        amplitude = np.abs(signal)

        # Create binary mask to know which sections are below the defined threshold
        low_amplitude_mask = amplitude < amplitude_threshold

        # Make sure section of low amplitude is a minimum length to be considered a pause
        num_samples_threshold = int(sampling_rate * time_frame_threshold)

        # Instantiate arrays to store results
        low_amplitude_sections = []
        # pauses = np.zeros((len(signal), 1))
        pauses = np.full((len(signal), 1), -1)

        # Flag when looping through list
        start = None

        for i in range(len(low_amplitude_mask)):
            if low_amplitude_mask[i]:
                if start is None: # The start of a low amplitude section
                    # Get the index of the start of the low amplitude section
                    start = i
            else:
                # Has been going over a low amplitude section and is bigger than the threshold length to be considered a pause
                if start is not None and (i - start) >= num_samples_threshold:
                    # Append indices to list to visualise later
                    low_amplitude_sections.append((start/sampling_rate, i/sampling_rate))

                    # Mark those time frames as there is a pause
                    for i in range(start, i+1):
                        pauses[i] = 1
                start = None

        return low_amplitude_sections, pauses


    def detect_pauses(self, video_path: str, sampling_rate: int = 22050,
                             amplitude_threshold: float = 0.01, time_frame_threshold: float = 0.1
                             ) -> tuple[List[tuple[float, float]], np.ndarray]:
        """
        Detects the pauses within a speech signal by determining whether the absolute amplitude of the audio signal is below the defined threshold.
        If a section is below the defined threshold, it is marked as a section where this is a pause with an array of 0's (no pause) and 1's (pause).
        The start and end times for each section is stored in a list of tuples.

        Arg(s):
        - video_path (str): The video filepath to extract an audio from
        - sampling_rate (int): The sampling rate of the audio signal
        - amplitude threshold (float): The threshold beneath which a signal will be said to have a pause
        - time_frame_threshold (float): The minimum length a section must be to be considered a pause (in seconds)

        Returns:
        - tuple[List[(float, float)], ArrayLike]: The list contains the start and end times for each pause. The ArrayLike object contains an array of 0's and 1's
        """
        # Extract audio from the video
        audio_signal = self.__extract_audio(video_path, sampling_rate)
        if audio_signal is None:
            print("ERROR: No audio found with video")
            return None, None, None

        # Envelope detection
        envelope = self.__envelope_detection(audio_signal, sampling_rate)

        # Low amplitude masking
        low_amplitude_sections, pauses = self.__low_amplitude_masking(envelope, sampling_rate, amplitude_threshold, time_frame_threshold)

        return low_amplitude_sections, pauses, sampling_rate