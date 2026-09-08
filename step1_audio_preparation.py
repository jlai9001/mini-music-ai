from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 32000

TARGET_PEAK_DBFS = -3.0

MIN_PEAK_AMPLITUDE = 0.000001


# Normalize the waveform so every recording has a similar peak level
def normalize_audio(
    audio,
):
    # Measure the loudest absolute waveform value
    current_peak = np.max(
        np.abs(
            audio
        )
    )

    # Leave effectively silent audio unchanged
    if current_peak < MIN_PEAK_AMPLITUDE:
        return audio

    # Convert the target decibel level into waveform amplitude
    target_peak = 10 ** (
        TARGET_PEAK_DBFS
        / 20
    )

    # Calculate how much the waveform needs to be scaled
    gain = (
        target_peak
        / current_peak
    )

    # Apply the same gain to the entire recording
    normalized_audio = (
        audio
        * gain
    )

    # Protect against accidental values outside the valid audio range
    normalized_audio = np.clip(
        normalized_audio,
        -1.0,
        1.0,
    )

    # Return 32-bit floating-point audio
    return normalized_audio.astype(
        np.float32
    )


# Prepare one audio file as 32 kHz mono audio
def resample_audio(
    input_path,
    output_path,
):
    # Load the original waveform and sample rate
    audio, sample_rate = sf.read(
        input_path,
        always_2d=True,
    )

    # Convert stereo or multi-channel audio into mono
    audio = audio.mean(
        axis=1
    )

    # Resample the audio when it is not already 32 kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        # Calculate the greatest common divisor
        divisor = np.gcd(
            sample_rate,
            TARGET_SAMPLE_RATE,
        )

        # Calculate the resampling numerator
        up = (
            TARGET_SAMPLE_RATE
            // divisor
        )

        # Calculate the resampling denominator
        down = (
            sample_rate
            // divisor
        )

        # Resample the waveform
        audio = resample_poly(
            audio,
            up,
            down,
        )

    # Convert the audio into 32-bit floating point samples
    audio = audio.astype(
        np.float32
    )

    # Normalize the recording to the common training volume
    audio = normalize_audio(
        audio
    )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the prepared audio
    sf.write(
        output_path,
        audio,
        TARGET_SAMPLE_RATE,
    )

    # Return the new audio path
    return str(
        output_path
    )

# Prepare every WAV file inside the training folder
def resample_training_audio(
    input_folder="audio/training",
    output_folder="audio/training_resampled",
):
    # Convert the folder paths into Path objects
    input_folder = Path(
        input_folder
    )

    output_folder = Path(
        output_folder
    )

    # Create the output folder if necessary
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Store the paths of every prepared recording
    output_paths = []

    # Find every WAV training recording
    input_paths = sorted(
        path
        for path in input_folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".wav"
        )
    )

    # Make sure training audio was found
    if not input_paths:
        raise RuntimeError(
            f"No WAV files were found in {input_folder}"
        )

    # Process each training recording
    for input_path in input_paths:
        # Build the destination path
        output_path = (
            output_folder
            / input_path.name
        )

        # Prepare the current recording
        prepared_path = resample_audio(
            str(
                input_path
            ),
            str(
                output_path
            ),
        )

        # Remember the prepared recording
        output_paths.append(
            prepared_path
        )

        # Show progress
        print(
            "Prepared:",
            prepared_path,
        )

    # Show how many recordings were processed
    print(
        "\nTraining files prepared:",
        len(
            output_paths
        ),
    )

    # Return every prepared file path
    return output_paths
