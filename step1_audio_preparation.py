from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 32000


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
            "No WAV training files were found in audio/training"
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
