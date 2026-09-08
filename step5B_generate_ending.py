from pathlib import Path

import torch

from audio_length import (
    ENDING_DURATION_SECONDS,
    SAMPLE_RATE,
    seconds_to_tokens,
)
from step4_train_model import CodecTransformer


ENDING_TEMPERATURE = 0.8

ENDING_TOP_K = 10


# Generate a clean-ending continuation using the ending-specialized Transformer
def generate_ending(
    generated_tokens_path,
    ending_model_path,
    output_path="outputs/generated_tokens_with_ending.pt",
):
    # Choose the GPU when CUDA is available
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Print the ending-generation device
    print(
        "\nEnding generation device:",
        device,
    )

    # Load the main generated token sequence
    generated_data = torch.load(
        generated_tokens_path,
        map_location="cpu",
        weights_only=False,
    )

    # Get the generated EnCodec tokens
    generated_tokens = generated_data[
        "audio_codes"
    ]

    # Get the duration originally requested by the user
    requested_seconds = float(
        generated_data[
            "requested_seconds"
        ]
    )

    # Load the ending-specialized Transformer checkpoint
    ending_checkpoint = torch.load(
        ending_model_path,
        map_location=device,
        weights_only=False,
    )

    # Get the ending-model configuration
    num_codebooks = ending_checkpoint[
        "num_codebooks"
    ]

    context_length = ending_checkpoint[
        "context_length"
    ]

    # Recreate the ending Transformer architecture
    ending_model = CodecTransformer(
        num_codebooks=num_codebooks,
        context_length=context_length,
        embedding_size=ending_checkpoint[
            "embedding_size"
        ],
        hidden_size=ending_checkpoint[
            "hidden_size"
        ],
        num_heads=ending_checkpoint[
            "num_heads"
        ],
        num_layers=ending_checkpoint[
            "num_layers"
        ],
    )

    # Load the ending-specialized learned weights
    ending_model.load_state_dict(
        ending_checkpoint[
            "model_state_dict"
        ]
    )

    # Move the ending model onto the selected device
    ending_model = ending_model.to(
        device
    )

    # Switch the ending model into generation mode
    ending_model.eval()

    # Move the main generated tokens onto the selected device
    combined_tokens = generated_tokens.to(
        device
    )

    # Remember how many main-generation tokens already exist
    main_token_length = combined_tokens.shape[
        -1
    ]

    # Convert the 1.5-second ending duration into token positions
    ending_token_length = seconds_to_tokens(
        ENDING_DURATION_SECONDS
    )

    # Disable gradient calculations during ending generation
    with torch.no_grad():
        # Generate one ending-token position at a time
        for _ in range(
            ending_token_length
        ):
            # Give the ending model the most recent generated context
            model_input = combined_tokens[
                :,
                :,
                -context_length:,
            ]

            # Predict the next ending token distribution
            predictions = ending_model(
                model_input
            )

            # Get the prediction from the final time position
            next_token_logits = predictions[
                :,
                :,
                -1,
                :,
            ]

            # Adjust how strongly the ending model favors likely choices
            next_token_logits = (
                next_token_logits
                / ENDING_TEMPERATURE
            )

            # Keep only the strongest ending-token candidates
            top_values, top_indices = torch.topk(
                next_token_logits,
                k=ENDING_TOP_K,
                dim=-1,
            )

            # Convert the selected logits into probabilities
            top_probabilities = torch.softmax(
                top_values,
                dim=-1,
            )

            # Flatten the batch and codebook dimensions for sampling
            flattened_probabilities = top_probabilities.reshape(
                -1,
                ENDING_TOP_K,
            )

            # Randomly choose among the strong ending candidates
            sampled_positions = torch.multinomial(
                flattened_probabilities,
                num_samples=1,
            )

            # Restore the batch and codebook dimensions
            sampled_positions = sampled_positions.reshape(
                1,
                num_codebooks,
                1,
            )

            # Convert the sampled positions into real EnCodec token IDs
            next_tokens = torch.gather(
                top_indices,
                dim=-1,
                index=sampled_positions,
            )

            # Append the new ending tokens
            combined_tokens = torch.cat(
                (
                    combined_tokens,
                    next_tokens,
                ),
                dim=-1,
            )

    # Separate the newly generated ending tokens for reporting
    ending_tokens = combined_tokens[
        :,
        :,
        main_token_length:,
    ]

    # Move the completed token sequence back onto the CPU
    combined_tokens = combined_tokens.cpu()

    # Move the ending-only tokens back onto the CPU
    ending_tokens = ending_tokens.cpu()

    # Calculate the final expected duration
    total_seconds = (
        requested_seconds
        + ENDING_DURATION_SECONDS
    )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the main generation and ending as one token sequence
    torch.save(
        {
            "audio_codes": combined_tokens,
            "requested_seconds": requested_seconds,
            "ending_seconds": ENDING_DURATION_SECONDS,
            "total_seconds": total_seconds,
            "sample_rate": SAMPLE_RATE,
        },
        output_path,
    )

    # Print the original generated-token shape
    print(
        "Main token shape:",
        tuple(
            generated_tokens.shape
        ),
    )

    # Print the ending-only token shape
    print(
        "Ending token shape:",
        tuple(
            ending_tokens.shape
        ),
    )

    # Print the combined token shape
    print(
        "Combined token shape:",
        tuple(
            combined_tokens.shape
        ),
    )

    # Print the ending duration
    print(
        "Ending duration:",
        ENDING_DURATION_SECONDS,
        "seconds",
    )

    # Print the expected total duration
    print(
        "Expected total duration:",
        total_seconds,
        "seconds",
    )

    # Confirm that ending generation finished
    print(
        "Step 5B complete:",
        output_path,
    )

    # Return the combined token path
    return str(
        output_path
    )
