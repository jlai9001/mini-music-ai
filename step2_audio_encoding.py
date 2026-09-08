# pyright: reportPrivateImportUsage=false

from pathlib import Path
from typing import cast

import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecEncoderOutput


MODEL_NAME = "facebook/encodec_32khz"
EXPECTED_SAMPLE_RATE = 32000


# Choose the GPU when CUDA is available
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Load the audio preprocessor used by EnCodec
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)


# Load the pretrained EnCodec model
encodec_model = EncodecModel.from_pretrained(
    MODEL_NAME
)


# Move EnCodec onto the selected device
torch.nn.Module.to(
    encodec_model,
    device,
)


# Switch EnCodec into inference mode
torch.nn.Module.eval(
    encodec_model
)


# Encode one prepared WAV file into EnCodec tokens
def encode_audio(
    input_path,
    output_path,
):
    # Load the prepared waveform
    audio, sample_rate = sf.read(
        input_path,
        dtype="float32",
    )

    # Convert multi-channel audio to mono if necessary
    if audio.ndim > 1:
        audio = audio.mean(
            axis=1
        )

    # Make sure Step 1 produced the sample rate EnCodec expects
    if sample_rate != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_RATE} Hz audio, "
            f"but received {sample_rate} Hz"
        )

    # Prepare the waveform for EnCodec
    inputs = feature_extractor(
        raw_audio=audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    # Move the waveform onto the selected device
    input_values = inputs[
        "input_values"
    ].to(
        device
    )

    # Move the padding mask onto the selected device
    padding_mask = inputs[
        "padding_mask"
    ].to(
        device
    )

    # Disable gradient calculations because EnCodec is not being trained
    with torch.no_grad():
        # Convert the waveform into discrete EnCodec tokens
        encoded = cast(
            EncodecEncoderOutput,
            encodec_model.encode(
                input_values,
                padding_mask,
                return_dict=True,
            ),
        )

    # Make sure EnCodec actually produced audio tokens
    if encoded.audio_codes is None:
        raise RuntimeError(
            "EnCodec did not return audio codes"
        )

    # Make sure EnCodec returned its scale information
    if encoded.audio_scales is None:
        raise RuntimeError(
            "EnCodec did not return audio scales"
        )

    # Correct the overly broad Transformers type annotation
    raw_audio_scales = cast(
        list[torch.Tensor | None],
        encoded.audio_scales,
    )

    # Move any scale tensors back onto the CPU
    audio_scales = [
        scale.cpu()
        if scale is not None
        else None
        for scale in raw_audio_scales
    ]

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Store the encoded representation
    encoded_data = {
        "audio_codes": encoded.audio_codes.cpu(),
        "audio_scales": audio_scales,
        "padding_mask": padding_mask.cpu(),
        "last_frame_pad_length": encoded.last_frame_pad_length,
        "sample_rate": sample_rate,
    }

    # Save the encoded representation
    torch.save(
        encoded_data,
        output_path,
    )

    # Print the encoded file path
    print(
        "Encoded:",
        input_path,
    )

    # Print the token shape
    print(
        "Audio code shape:",
        tuple(
            encoded.audio_codes.shape
        ),
    )

    # Return the encoded file path
    return str(
        output_path
    )


# Encode every prepared training recording
def encode_training_audio(
    input_paths,
    output_folder="outputs/training_encoded",
):
    # Convert the output folder into a Path
    output_folder = Path(
        output_folder
    )

    # Create the output folder if necessary
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Store every encoded file path
    output_paths = []

    # Encode every prepared training recording
    for input_path in input_paths:
        # Convert the input path into a Path
        input_path = Path(
            input_path
        )

        # Replace the WAV extension with a PyTorch file extension
        output_path = (
            output_folder
            / f"{input_path.stem}.pt"
        )

        # Encode the current recording
        encoded_path = encode_audio(
            str(
                input_path
            ),
            str(
                output_path
            ),
        )

        # Remember the encoded representation
        output_paths.append(
            encoded_path
        )

    # Print the total number of encoded recordings
    print(
        "\nTraining files encoded:",
        len(
            output_paths
        ),
    )

    # Return all encoded representation paths
    return output_paths
