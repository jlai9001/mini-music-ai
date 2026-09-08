from pathlib import Path

import torch

from step4_train_model import CodecTransformer

from audio_length import (
    CLEAN_END_EXTRA_SECONDS,
    SAMPLE_RATE,
    seconds_to_tokens,
)


TEMPERATURE = 0.9

TOP_K = 20


# Generate new EnCodec tokens using the trained Transformer
def generate_tokens(
    trained_model_path,
    seconds=5.0,
    output_path="outputs/generated_tokens.pt",
):
    # Choose the GPU when CUDA is available
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Print the generation device
    print(
        "\nGeneration device:",
        device,
    )

    # Load the trained Transformer checkpoint
    checkpoint = torch.load(
        trained_model_path,
        map_location=device,
        weights_only=False,
    )

    # Get the model configuration
    num_codebooks = checkpoint[
        "num_codebooks"
    ]

    context_length = checkpoint[
        "context_length"
    ]

    start_token_id = checkpoint[
        "start_token_id"
    ]

    # Recreate the trained Transformer architecture
    model = CodecTransformer(
        num_codebooks=num_codebooks,
        context_length=context_length,
        embedding_size=checkpoint[
            "embedding_size"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_heads=checkpoint[
            "num_heads"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
    )

    # Load the learned model weights
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # Move the model onto the selected device
    model = model.to(
        device
    )

    # Switch the Transformer into generation mode
    model.eval()

    # Add extra generation time so Step 6 can find a natural ending
    generation_seconds = (
        seconds
        + CLEAN_END_EXTRA_SECONDS
    )

    # Convert the extended duration into EnCodec token positions
    target_token_length = seconds_to_tokens(
        generation_seconds
    )

    # Begin with one START token for every EnCodec codebook
    generated_tokens = torch.full(
        (
            1,
            num_codebooks,
            1,
        ),
        start_token_id,
        dtype=torch.long,
        device=device,
    )

    # Disable gradient calculations during generation
    with torch.no_grad():
        # Generate one new audio-token position at a time
        for _ in range(
            target_token_length
        ):
            # Keep only the most recent context window
            model_input = generated_tokens[
                :,
                :,
                -context_length:,
            ]

            # Predict the next token distribution
            predictions = model(
                model_input
            )

            # Get predictions from the final time position
            next_token_logits = predictions[
                :,
                :,
                -1,
                :,
            ]

            # Adjust how strongly the model favors its most likely choices
            next_token_logits = (
                next_token_logits
                / TEMPERATURE
            )

            # Keep only the model's most likely token choices
            top_values, top_indices = torch.topk(
                next_token_logits,
                k=TOP_K,
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
                TOP_K,
            )

            # Randomly choose one of the strong candidate tokens
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

            # Convert the sampled positions back into real EnCodec token IDs
            next_tokens = torch.gather(
                top_indices,
                dim=-1,
                index=sampled_positions,
            )

            # Append the predicted tokens to the generated sequence
            generated_tokens = torch.cat(
                (
                    generated_tokens,
                    next_tokens,
                ),
                dim=-1,
            )

    # Remove the initial START token
    generated_tokens = generated_tokens[
        :,
        :,
        1:,
    ]

    # Move the generated tokens back onto the CPU
    generated_tokens = generated_tokens.cpu()

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the generated token sequence
    torch.save(
        {
            "audio_codes": generated_tokens,
            "requested_seconds": seconds,
            "generated_seconds": generation_seconds,
            "sample_rate": SAMPLE_RATE,
        },
        output_path,
    )

    # Print the generated token shape
    print(
        "Generated token shape:",
        tuple(
            generated_tokens.shape
        ),
    )

    # Print the requested duration
    print(
        "Requested duration:",
        seconds,
        "seconds",
    )

    # Print how much audio was generated for clean-ending detection
    print(
        "Generated search duration:",
        generation_seconds,
        "seconds",
    )

    # Print the sampling settings used for generation
    print(
        "Sampling temperature:",
        TEMPERATURE,
    )

    # Print how many candidate tokens were considered
    print(
        "Top-K sampling:",
        TOP_K,
    )

    # Confirm that token generation finished
    print(
        "Step 5 complete:",
        output_path,
    )

    # Return the generated token path
    return str(
        output_path
    )
