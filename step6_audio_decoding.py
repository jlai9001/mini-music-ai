# pyright: reportPrivateImportUsage=false

from pathlib import Path
from typing import cast

import soundfile as sf
import torch
from transformers import EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecDecoderOutput

from audio_length import (
    SAMPLE_RATE,
    seconds_to_samples,
)

MODEL_NAME = "facebook/encodec_32khz"

# Decode generated EnCodec tokens into a WAV file
def decode_generated_audio(
    generated_tokens_path,
    output_path="outputs/generated_audio.wav",
):
    # Choose the GPU when CUDA is available
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Print the decoding device
    print(
        "\nDecoding device:",
        device,
    )

    # Load the generated tokens created in Step 5
    generated_data = torch.load(
        generated_tokens_path,
        map_location="cpu",
        weights_only=False,
    )

    # Get the generated EnCodec token sequence
    generated_tokens = generated_data[
        "audio_codes"
    ]

    # Get the requested audio duration
    seconds = float(
        generated_data[
            "seconds"
        ]
    )

    # Load the pretrained EnCodec decoder
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

    # Move the generated tokens onto the selected device
    generated_tokens = generated_tokens.to(
        device
    )

    # Add the EnCodec frame dimension
    audio_codes = generated_tokens.unsqueeze(
        0
    )

    # Create one scale entry for the single EnCodec frame
    audio_scales = cast(
        torch.Tensor,
        [None],
    )

    # Disable gradients because we are only decoding
    with torch.no_grad():
        # Convert generated EnCodec tokens back into waveform audio
        decoded = cast(
            EncodecDecoderOutput,
            encodec_model.decode(
                audio_codes,
                audio_scales,
                padding_mask=None,
                return_dict=True,
                last_frame_pad_length=0,
            ),
        )

    # Make sure EnCodec returned waveform audio
    if decoded.audio_values is None:
        raise RuntimeError(
            "EnCodec did not return waveform audio"
        )

    # Get the decoded waveform
    generated_audio = decoded.audio_values

    # Remove the batch and channel dimensions
    generated_audio = generated_audio[
        0,
        0,
    ]

    # Move the waveform back onto the CPU
    generated_audio = generated_audio.detach().cpu()

    # Convert the requested duration into waveform samples
    target_sample_count = seconds_to_samples(
        seconds
    )

    # Trim any extra samples produced by the decoder
    generated_audio = generated_audio[
        :target_sample_count
    ]

    # Convert the waveform into a NumPy array
    generated_audio = generated_audio.numpy()

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the generated waveform as a WAV file
    sf.write(
        output_path,
        generated_audio,
        SAMPLE_RATE,
    )

    # Print the final waveform shape
    print(
        "Generated waveform shape:",
        generated_audio.shape,
    )

    # Print the final audio duration
    print(
        "Generated duration:",
        generated_audio.shape[
            0
        ]
        / SAMPLE_RATE,
        "seconds",
    )

    # Confirm that decoding finished
    print(
        "Step 6 complete:",
        output_path,
    )

    # Return the generated WAV path
    return str(
        output_path
    )
