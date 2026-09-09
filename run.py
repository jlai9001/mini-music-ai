import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from step1_audio_preparation import resample_training_audio
from step2_audio_encoding import encode_training_audio
from step3_training_sequences import build_training_sequences
from step3B_ending_training_sequences import build_ending_training_sequences
from step4_train_model import train_model
from step4B_train_ending_model import train_ending_model
from step5_generate_tokens import generate_tokens
from step5B_generate_ending import generate_ending
from step6_audio_decoding import decode_generated_audio
from step7_collect_accepted_audio import collect_accepted_audio
from step8_train_accepted_model import train_accepted_model


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

    # Allow training on user-accepted generations
    parser.add_argument(
        "--train-accepted",
        action="store_true",
    )

    # Allow the accepted-trained model to become the new stable model
    parser.add_argument(
        "--promote-model",
        action="store_true",
    )

    # Allow the separate ending model to be added after generation
    parser.add_argument(
        "--with-ending",
        action="store_true",
    )

    # Read the command-line arguments
    args = parser.parse_args()

    # Promote the accepted-trained model when explicitly requested
    if args.promote_model:
        # Define the accepted-trained model path
        accepted_model_path = Path(
            "outputs/accepted_transformer.pt"
        )

        # Define the current stable model path
        stable_model_path = Path(
            "models/codec_transformer.pt"
        )

        # Make sure an accepted-trained model exists
        if not accepted_model_path.exists():
            raise RuntimeError(
                "No accepted-trained model found to promote."
            )

        # Create a folder for backups of previous stable models
        backup_folder = Path(
            "outputs/model_backups"
        )

        backup_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Create a timestamp for the backup
        backup_timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        # Create the backup path
        backup_model_path = (
            backup_folder
            / f"codec_transformer_{backup_timestamp}.pt"
        )

        # Back up the current stable model
        shutil.copy2(
            stable_model_path,
            backup_model_path,
        )

        # Promote the accepted-trained model
        shutil.copy2(
            accepted_model_path,
            stable_model_path,
        )

        # Confirm the promotion
        print(
            "Accepted-trained model promoted to stable model."
        )

        print(
            "Previous stable model backed up to:",
            backup_model_path,
        )

        # Stop after promotion
        return

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

    # Prepare user-accepted generations for accepted-model training
    elif args.train_accepted:
        # Collect only the generations marked as accepted
        accepted_training_folder = collect_accepted_audio()

        # Prepare the accepted WAV files as 32 kHz mono audio
        accepted_resampled_audio_paths = resample_training_audio(
            input_folder=accepted_training_folder,
            output_folder="audio/accepted_training_resampled",
        )

        # Convert the accepted audio into EnCodec tokens
        accepted_encoded_audio_paths = encode_training_audio(
            accepted_resampled_audio_paths,
            output_folder="outputs/accepted_encoded",
        )

        # Convert the accepted tokens into Transformer training sequences
        accepted_training_dataset_path = build_training_sequences(
            accepted_encoded_audio_paths,
            output_path="outputs/accepted_training_sequences.pt",
        )

        # Confirm that the accepted dataset is ready
        print(
            "Accepted training dataset ready:",
            accepted_training_dataset_path,
        )

        # Fine-tune a copy of the stable model on accepted complete clips
        accepted_model_path = train_accepted_model(
            base_model_path="models/codec_transformer.pt",
            training_dataset_path=accepted_training_dataset_path,
            output_path="outputs/accepted_transformer.pt",
        )

        # Confirm where the accepted-trained model was saved
        print(
            "Accepted-trained model ready:",
            accepted_model_path,
        )

        # Stop before the normal generation pipeline
        return


    # Otherwise, use the saved stable models
    else:
        trained_model_path = "models/codec_transformer.pt"
        ending_model_path = "models/ending_transformer.pt"

    # Step 5: generate new EnCodec tokens for the requested duration
    generated_tokens_path = generate_tokens(
        trained_model_path,
        seconds=args.seconds,
    )

    # Add the separate ending model only when explicitly requested
    if args.with_ending:
        generated_tokens_for_decoding_path = generate_ending(
            generated_tokens_path,
            ending_model_path,
        )

    # Otherwise, use the main model generation by itself
    else:
        generated_tokens_for_decoding_path = generated_tokens_path

    # Create a unique timestamp for this generation
    generation_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # Create a unique output path for this generation
    generated_audio_path = (
        f"outputs/generations/"
        f"generation_{generation_timestamp}.wav"
    )

    # Step 6: decode the final generated tokens
    decode_generated_audio(
        generated_tokens_for_decoding_path,
        output_path=generated_audio_path,
    )

    # Ask the user whether this generation should be accepted or rejected
    feedback = input(
        "Accept or reject this generation? [a/r]: "
    ).strip().lower()

    # Keep asking until the user enters a valid response
    while feedback not in ("a", "r"):
        feedback = input(
            "Please enter 'a' for accept or 'r' for reject: "
        ).strip().lower()

    # Convert the short response into a readable label
    rating = "accept" if feedback == "a" else "reject"

    # Define where generation feedback will be stored
    feedback_path = Path("outputs/feedback.json")

    # Load previous feedback if it already exists
    if feedback_path.exists():
        with feedback_path.open("r") as file:
            feedback_history = json.load(file)
    else:
        feedback_history = []

    # Add this generation to the feedback history
    feedback_history.append(
        {
            "generation": generated_audio_path,
            "rating": rating,
            "seconds": args.seconds,
            "timestamp": generation_timestamp,
        }
    )

    # Save the updated feedback history
    with feedback_path.open("w") as file:
        json.dump(
            feedback_history,
            file,
            indent=4,
        )

    # Confirm that the feedback was saved
    print(f"Feedback saved: {rating}")

    # Count the accepted generations
    accepted_count = sum(
        1
        for item in feedback_history
        if item["rating"] == "accept"
    )

    # Count all rated generations
    total_count = len(
        feedback_history
    )

    # Calculate the percentage of accepted generations
    acceptance_rate = (
        accepted_count
        / total_count
        * 100
    )

    # Print the total number of accepted generations
    print(
        f"Total accepted: {accepted_count}"
    )

    print(
        f"Acceptance rate: {acceptance_rate:.1f}%"
    )

if __name__ == "__main__":
    main()
