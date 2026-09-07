from step1_audio_preparation import resample_training_audio


def main():
    # Step 1: prepare all training audio as 32 kHz mono WAV files
    resampled_audio_paths = resample_training_audio()

    # Print the current final output
    print(
        "\nPipeline complete."
    )

    print(
        "Prepared audio files:",
        len(
            resampled_audio_paths
        ),
    )


if __name__ == "__main__":
    main()
