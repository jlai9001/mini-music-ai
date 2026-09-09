from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from step4_train_model import CODEBOOK_SIZE, CodecTransformer


# Fine-tune the stable model using user-accepted generations
def train_accepted_model(
    base_model_path,
    training_dataset_path,
    output_path="outputs/accepted_transformer.pt",
    epochs=5,
    batch_size=32,
):
    # Choose the GPU when CUDA is available
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Print the training device
    print(
        "\nAccepted-training device:",
        device,
    )

    # Load the accepted-generation training dataset
    dataset = torch.load(
        training_dataset_path,
        map_location="cpu",
        weights_only=False,
    )

    # Get the training inputs
    inputs = dataset[
        "inputs"
    ]

    # Get the expected next-token outputs
    targets = dataset[
        "targets"
    ]

    # Load the stable model checkpoint
    checkpoint = torch.load(
        base_model_path,
        map_location=device,
        weights_only=False,
    )

    # Create the same Transformer architecture as the stable model
    model = CodecTransformer(
        num_codebooks=checkpoint[
            "num_codebooks"
        ],
        context_length=checkpoint[
            "context_length"
        ],
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

    # Load the stable model's learned weights
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # Move the model onto the selected device
    model = model.to(
        device
    )

    # Create the accepted-generation training dataset
    training_data = TensorDataset(
        inputs,
        targets,
    )

    # Create training batches
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # Use a small learning rate so the stable model changes gradually
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
    )

    # Create the next-token prediction loss
    loss_function = nn.CrossEntropyLoss()

    # Switch the model into training mode
    model.train()

    # Fine-tune for a small number of epochs
    for epoch in range(
        1,
        epochs + 1,
    ):
        # Track the total loss for this epoch
        total_loss = 0.0

        # Track how many batches were processed
        batch_count = 0

        # Process every training batch
        for batch_inputs, batch_targets in training_loader:
            # Move the inputs onto the selected device
            batch_inputs = batch_inputs.to(
                device
            )

            # Move the targets onto the selected device
            batch_targets = batch_targets.to(
                device
            )

            # Clear gradients from the previous update
            optimizer.zero_grad()

            # Predict the next EnCodec tokens
            predictions = model(
                batch_inputs
            )

            # Flatten the predictions for cross-entropy
            prediction_values = predictions.reshape(
                -1,
                CODEBOOK_SIZE,
            )

            # Flatten the expected token IDs
            target_values = batch_targets.reshape(
                -1
            )

            # Calculate the prediction error
            loss = loss_function(
                prediction_values,
                target_values,
            )

            # Calculate gradients
            loss.backward()

            # Prevent unusually large gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # Update the model weights
            optimizer.step()

            # Add this batch's loss
            total_loss += float(
                loss.item()
            )

            # Count the batch
            batch_count += 1

        # Calculate the average loss
        average_loss = (
            total_loss
            / batch_count
        )

        # Show training progress
        print(
            f"Accepted epoch {epoch}/{epochs} "
            f"Loss: {average_loss:.4f}"
        )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build the candidate model checkpoint
    candidate_checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_codebooks": checkpoint["num_codebooks"],
        "context_length": checkpoint["context_length"],
        "embedding_size": checkpoint["embedding_size"],
        "hidden_size": checkpoint["hidden_size"],
        "num_heads": checkpoint["num_heads"],
        "num_layers": checkpoint["num_layers"],
        "codebook_size": checkpoint["codebook_size"],
        "start_token_id": checkpoint["start_token_id"],
    }

    # Save the candidate without overwriting the stable model
    torch.save(
        candidate_checkpoint,
        output_path,
    )

    # Confirm that accepted-generation training finished
    print(
        "Accepted model complete:",
        output_path,
    )

    # Return the candidate model path
    return str(
        output_path
    )
