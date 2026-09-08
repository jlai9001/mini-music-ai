import argparse

from step1_audio_preparation import resample_training_audio
from step2_audio_encoding import encode_training_audio
from step3_training_sequences import build_training_sequences
from step3B_ending_training_sequences import build_ending_training_sequences
from step4_train_model import train_model
from step4B_train_ending_model import train_ending_model
from step5_generate_tokens import generate_tokens
from step5B_generate_ending import generate_ending
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

    # Allow the full training pipeline to be explicitly requested
    parser.add_argument(
        "--full-training",
        action="store_true",
    )

    # Read the command-line arguments
    args = parser.parse_args()

    # Run the full training pipeline only when explicitly requested
    if args.full_training:
        # Step 1: prepare all training audio as 32 kHz mono WAV files
        resampled_audio_paths = resample_training_audio()

        # Prepare the clean-ending training audio separately
        ending_resampled_audio_paths = resample_training_audio(
            input_folder="audio/ending_training",
            output_folder="audio/ending_training_resampled",
        )

        # Step 2: convert the prepared audio into EnCodec tokens
        encoded_audio_paths = encode_training_audio(
            resampled_audio_paths
        )

        # Convert the prepared clean-ending audio into EnCodec tokens separately
        ending_encoded_audio_paths = encode_training_audio(
            ending_resampled_audio_paths,
            output_folder="outputs/ending_encoded",
        )

        # Step 3: create next-token training sequences
        training_dataset_path = build_training_sequences(
            encoded_audio_paths
        )

        # Create clean-ending training sequences
        ending_training_dataset_path = build_ending_training_sequences(
            ending_encoded_audio_paths
        )

        # Step 4: train the generative Transformer
        trained_model_path = train_model(
            training_dataset_path
        )

        # Fine-tune a clean-ending specialist from the main model
        ending_model_path = train_ending_model(
            trained_model_path,
            ending_training_dataset_path,
        )

    # Otherwise, use the saved stable models
    else:
        trained_model_path = "models/codec_transformer.pt"
        ending_model_path = "models/ending_transformer.pt"

    # Step 5: generate new EnCodec tokens for the requested duration
    generated_tokens_path = generate_tokens(
        trained_model_path,
        seconds=args.seconds,
    )

    # Step 5B: generate a clean ending after the main audio
    generated_tokens_with_ending_path = generate_ending(
        generated_tokens_path,
        ending_model_path,
    )

    # Step 6: decode the generated audio including the clean ending
    decode_generated_audio(
        generated_tokens_with_ending_path
    )


if __name__ == "__main__":
    main()
