import argparse

from step1_audio_preparation import resample_training_audio
from step2_audio_encoding import encode_training_audio
from step3_training_sequences import build_training_sequences
from step4_train_model import train_model
from step5_generate_tokens import generate_tokens
from step6_audio_decoding import decode_generated_audio


def main():
    # Create command-line arguments for the pipeline
    parser = argparse.ArgumentParser()

    # Allow the generated audio duration to be configured
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
    )

    # Read the command-line arguments
    args = parser.parse_args()

    # Step 1: prepare all training audio as 32 kHz mono WAV files
    resampled_audio_paths = resample_training_audio()

    # Step 2: convert the prepared audio into EnCodec tokens
    encoded_audio_paths = encode_training_audio(
        resampled_audio_paths
    )

    # Step 3: create next-token training sequences
    training_dataset_path = build_training_sequences(
        encoded_audio_paths
    )

    # Step 4: train the generative Transformer
    trained_model_path = train_model(
        training_dataset_path
    )

    # Step 5: generate new EnCodec tokens for the requested duration
    generated_tokens_path = generate_tokens(
        trained_model_path,
        seconds=args.seconds,
    )

    # Step 6: decode the generated tokens into a WAV file
    decode_generated_audio(
        generated_tokens_path
    )


if __name__ == "__main__":
    main()
