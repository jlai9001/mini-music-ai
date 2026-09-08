from pathlib import Path

import torch


ENDING_SEQUENCE_LENGTH = 20

START_TOKEN_ID = 2048


# Build short next-token training sequences from clean-ending recordings
def build_ending_training_sequences(
    encoded_paths,
    output_path="outputs/ending_training_sequences.pt",
):
    # Store every ending input sequence
    all_inputs = []

    # Store every ending target sequence
    all_targets = []

    # Remember which recording produced each sequence
    source_paths = []

    # Process every encoded clean-ending recording
    for encoded_path in encoded_paths:
        # Load the EnCodec representation
        encoded_data = torch.load(
            encoded_path,
            map_location="cpu",
            weights_only=False,
        )

        # Get the EnCodec token sequence
        audio_codes = encoded_data[
            "audio_codes"
        ][
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
            dtype=torch.long,
        )

        # Add the START tokens before the clean-ending sequence
        token_sequence = torch.cat(
            (
                start_tokens,
                audio_codes,
            ),
            dim=-1,
        )

        # Skip recordings that are too short for one training example
        if (
            token_sequence.shape[
                -1
            ]
            <= ENDING_SEQUENCE_LENGTH
        ):
            print(
                "Skipped short ending:",
                encoded_path,
            )

            continue

        # Create overlapping next-token examples
        for start_index in range(
            token_sequence.shape[
                -1
            ]
            - ENDING_SEQUENCE_LENGTH
        ):
            # Select the current ending context
            input_sequence = token_sequence[
                :,
                start_index:
                start_index
                + ENDING_SEQUENCE_LENGTH,
            ]

            # Select the sequence shifted one position into the future
            target_sequence = token_sequence[
                :,
                start_index
                + 1:
                start_index
                + ENDING_SEQUENCE_LENGTH
                + 1,
            ]

            # Store the ending input
            all_inputs.append(
                input_sequence
            )

            # Store the expected continuation
            all_targets.append(
                target_sequence
            )

            # Remember which ending recording produced this example
            source_paths.append(
                str(
                    encoded_path
                )
            )

    # Make sure at least one ending sequence was created
    if not all_inputs:
        raise RuntimeError(
            "No ending training sequences were created"
        )

    # Combine every input sequence into one training tensor
    inputs = torch.stack(
        all_inputs
    )

    # Combine every target sequence into one training tensor
    targets = torch.stack(
        all_targets
    )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the ending training dataset
    torch.save(
        {
            "inputs": inputs,
            "targets": targets,
            "sequence_length": ENDING_SEQUENCE_LENGTH,
            "num_codebooks": inputs.shape[
                1
            ],
            "start_token_id": START_TOKEN_ID,
            "source_paths": source_paths,
        },
        output_path,
    )

    # Print how many ending examples were created
    print(
        "\nEnding training sequences created:",
        inputs.shape[
            0
        ],
    )

    # Print the ending input tensor shape
    print(
        "Ending input sequence shape:",
        tuple(
            inputs.shape
        ),
    )

    # Print the ending target tensor shape
    print(
        "Ending target sequence shape:",
        tuple(
            targets.shape
        ),
    )

    # Confirm that the ending dataset was saved
    print(
        "Ending sequence dataset:",
        output_path,
    )

    # Return the ending dataset path
    return str(
        output_path
    )
