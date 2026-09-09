from pathlib import Path

import torch


CONTEXT_LENGTH = 150

START_TOKEN_ID = 2048


# Build Transformer training sequences from encoded EnCodec files
def build_training_sequences(
    encoded_paths,
    output_path="outputs/training_sequences.pt",
    stride=1,
):
    # Store every Transformer input sequence
    input_sequences = []

    # Store the correct next-token sequences
    target_sequences = []

    # Remember which recording each sequence came from
    source_paths = []

    # Process every encoded training recording
    for encoded_path in encoded_paths:
        # Load the EnCodec representation created in Step 2
        encoded_data = torch.load(
            encoded_path,
            map_location="cpu",
            weights_only=False,
        )

        # Get the discrete EnCodec token tensor
        audio_codes = encoded_data[
            "audio_codes"
        ]

        # Remove the batch and frame dimensions
        audio_codes = audio_codes[
            0,
            0,
        ]

        # Get the number of EnCodec codebooks
        num_codebooks = audio_codes.shape[
            0
        ]

        # Create one START token for every codebook
        start_tokens = torch.full(
            (
                num_codebooks,
                1,
            ),
            START_TOKEN_ID,
            dtype=audio_codes.dtype,
        )

        # Put the START token before the real audio tokens
        sequence = torch.cat(
            (
                start_tokens,
                audio_codes,
            ),
            dim=1,
        )

        # Get the complete sequence length
        token_length = sequence.shape[
            -1
        ]

        # Make sure the recording is long enough for one training example
        if token_length <= CONTEXT_LENGTH:
            print(
                "Skipping short recording:",
                encoded_path,
            )

            continue

        # Slide across the recording using the requested stride
        for start_index in range(
            0,
            token_length - CONTEXT_LENGTH,
            stride,
        ):
            # Calculate the end of the input sequence
            input_end = (
                start_index
                + CONTEXT_LENGTH
            )

            # Create the input sequence
            input_sequence = sequence[
                :,
                start_index:input_end,
            ]

            # Create the expected next-token sequence
            target_sequence = sequence[
                :,
                start_index + 1:input_end + 1,
            ]

            # Store the input sequence
            input_sequences.append(
                input_sequence
            )

            # Store the correct target sequence
            target_sequences.append(
                target_sequence
            )

            # Remember which recording produced this example
            source_paths.append(
                str(
                    encoded_path
                )
            )

    # Make sure at least one training sequence was created
    if not input_sequences:
        raise RuntimeError(
            "No training sequences were created"
        )

    # Combine all input sequences into one tensor
    inputs = torch.stack(
        input_sequences
    )

    # Combine all target sequences into one tensor
    targets = torch.stack(
        target_sequences
    )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build the complete training dataset
    training_dataset = {
        "inputs": inputs,
        "targets": targets,
        "context_length": CONTEXT_LENGTH,
        "num_codebooks": inputs.shape[
            1
        ],
        "start_token_id": START_TOKEN_ID,
        "source_paths": source_paths,
    }

    # Save the training dataset
    torch.save(
        training_dataset,
        output_path,
    )

    # Print the total number of training examples
    print(
        "\nTraining sequences created:",
        inputs.shape[
            0
        ],
    )

    # Print the input tensor shape
    print(
        "Input sequence shape:",
        tuple(
            inputs.shape
        ),
    )

    # Print the target tensor shape
    print(
        "Target sequence shape:",
        tuple(
            targets.shape
        ),
    )

    # Return the saved dataset path
    return str(
        output_path
    )
