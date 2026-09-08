from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from step4_train_model import CodecTransformer


ENDING_EPOCHS = 50

ENDING_BATCH_SIZE = 32

ENDING_LEARNING_RATE = 0.0001


# Fine-tune the main Transformer so it specializes in clean endings
def train_ending_model(
    main_model_path,
    ending_training_dataset_path,
    output_path="outputs/ending_transformer.pt",
):
    # Choose the GPU when CUDA is available
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Print the ending-training device
    print(
        "\nEnding training device:",
        device,
    )

    # Load the already-trained main Transformer checkpoint
    main_checkpoint = torch.load(
        main_model_path,
        map_location=device,
        weights_only=False,
    )

    # Load the clean-ending training sequences
    ending_dataset = torch.load(
        ending_training_dataset_path,
        map_location="cpu",
        weights_only=False,
    )

    # Get the ending input sequences
    inputs = ending_dataset[
        "inputs"
    ]

    # Get the expected next-token sequences
    targets = ending_dataset[
        "targets"
    ]

    # Build the ending training dataset
    tensor_dataset = TensorDataset(
        inputs,
        targets,
    )

    # Create shuffled mini-batches
    data_loader = DataLoader(
        tensor_dataset,
        batch_size=ENDING_BATCH_SIZE,
        shuffle=True,
    )

    # Recreate the exact architecture used by the main Transformer
    ending_model = CodecTransformer(
        num_codebooks=main_checkpoint[
            "num_codebooks"
        ],
        context_length=main_checkpoint[
            "context_length"
        ],
        embedding_size=main_checkpoint[
            "embedding_size"
        ],
        hidden_size=main_checkpoint[
            "hidden_size"
        ],
        num_heads=main_checkpoint[
            "num_heads"
        ],
        num_layers=main_checkpoint[
            "num_layers"
        ],
    )

    # Copy the main Transformer's learned weights into the ending model
    ending_model.load_state_dict(
        main_checkpoint[
            "model_state_dict"
        ]
    )

    # Move the ending model onto the selected device
    ending_model = ending_model.to(
        device
    )

    # Switch the ending model into training mode
    ending_model.train()

    # Use a conservative optimizer for ending specialization
    optimizer = torch.optim.AdamW(
        ending_model.parameters(),
        lr=ENDING_LEARNING_RATE,
    )

    # Compare predicted EnCodec tokens against the correct tokens
    loss_function = nn.CrossEntropyLoss()

    # Fine-tune the model on the clean-ending examples
    for epoch in range(
        ENDING_EPOCHS
    ):
        # Store the total loss for this epoch
        total_loss = 0.0

        # Process every ending mini-batch
        for batch_inputs, batch_targets in data_loader:
            # Move the current inputs onto the selected device
            batch_inputs = batch_inputs.to(
                device
            )

            # Move the current targets onto the selected device
            batch_targets = batch_targets.to(
                device
            )

            # Clear gradients from the previous training step
            optimizer.zero_grad()

            # Predict the next EnCodec tokens
            predictions = ending_model(
                batch_inputs
            )

            # Flatten predictions for cross-entropy loss
            flattened_predictions = predictions.reshape(
                -1,
                predictions.shape[
                    -1
                ],
            )

            # Flatten the expected tokens
            flattened_targets = batch_targets.reshape(
                -1
            )

            # Measure how different the predictions are from the targets
            loss = loss_function(
                flattened_predictions,
                flattened_targets,
            )

            # Calculate gradients
            loss.backward()

            # Prevent unusually large gradient updates
            torch.nn.utils.clip_grad_norm_(
                ending_model.parameters(),
                max_norm=1.0,
            )

            # Update the ending model
            optimizer.step()

            # Add this batch's loss to the epoch total
            total_loss += loss.item()

        # Calculate the average loss for the epoch
        average_loss = (
            total_loss
            / len(
                data_loader
            )
        )

        # Print progress on the first epoch and every ten epochs
        if (
            epoch == 0
            or (
                epoch
                + 1
            )
            % 10
            == 0
            or epoch
            == ENDING_EPOCHS
            - 1
        ):
            print(
                f"Ending epoch {epoch + 1}/{ENDING_EPOCHS}",
                f"Loss: {average_loss:.4f}",
            )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the ending-specialized Transformer
    torch.save(
        {
            "model_state_dict": ending_model.state_dict(),
            "num_codebooks": main_checkpoint[
                "num_codebooks"
            ],
            "context_length": main_checkpoint[
                "context_length"
            ],
            "embedding_size": main_checkpoint[
                "embedding_size"
            ],
            "hidden_size": main_checkpoint[
                "hidden_size"
            ],
            "num_heads": main_checkpoint[
                "num_heads"
            ],
            "num_layers": main_checkpoint[
                "num_layers"
            ],
            "start_token_id": main_checkpoint[
                "start_token_id"
            ],
        },
        output_path,
    )

    # Confirm that ending-model training finished
    print(
        "Step 4B complete:",
        output_path,
    )

    # Return the ending Transformer checkpoint path
    return str(
        output_path
    )
