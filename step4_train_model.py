from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CODEBOOK_SIZE = 2048

START_TOKEN_ID = 2048

VOCAB_SIZE = CODEBOOK_SIZE + 1


# Predict future EnCodec tokens from previous EnCodec tokens
class CodecTransformer(nn.Module):
    def __init__(
        self,
        num_codebooks,
        context_length,
        embedding_size=64,
        hidden_size=256,
        num_heads=8,
        num_layers=4,
    ):
        super().__init__()

        # Remember important model dimensions
        self.num_codebooks = num_codebooks
        self.context_length = context_length
        self.hidden_size = hidden_size

        # Create a separate token embedding for each EnCodec codebook
        self.codebook_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    VOCAB_SIZE,
                    embedding_size,
                )
                for _ in range(
                    num_codebooks
                )
            ]
        )

        # Combine all codebook embeddings into the Transformer hidden size
        self.input_projection = nn.Linear(
            num_codebooks
            * embedding_size,
            hidden_size,
        )

        # Learn the meaning of each position inside the sequence
        self.position_embedding = nn.Embedding(
            context_length,
            hidden_size,
        )

        # Create one Transformer layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )

        # Stack several Transformer layers together
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers,
        )

        # Predict one EnCodec token distribution for every codebook
        self.output_projection = nn.Linear(
            hidden_size,
            num_codebooks
            * CODEBOOK_SIZE,
        )

    def forward(
        self,
        token_sequences,
    ):
        # Get the batch size and token sequence length
        batch_size = token_sequences.shape[
            0
        ]

        sequence_length = token_sequences.shape[
            2
        ]

        # Store the embedding from every codebook
        embedded_codebooks = []

        # Embed each EnCodec codebook independently
        for codebook_index in range(
            self.num_codebooks
        ):
            # Get one codebook's token sequence
            codebook_tokens = token_sequences[
                :,
                codebook_index,
                :,
            ]

            # Convert token IDs into learned vectors
            codebook_embedding = self.codebook_embeddings[
                codebook_index
            ](
                codebook_tokens
            )

            # Remember the current codebook embedding
            embedded_codebooks.append(
                codebook_embedding
            )

        # Combine all codebook information at every time position
        combined_embeddings = torch.cat(
            embedded_codebooks,
            dim=-1,
        )

        # Convert the combined representation into Transformer features
        hidden_states = self.input_projection(
            combined_embeddings
        )

        # Create the position numbers for this sequence
        positions = torch.arange(
            sequence_length,
            device=token_sequences.device,
        )

        # Add a batch dimension to the position numbers
        positions = positions.unsqueeze(
            0
        )

        # Add learned position information
        hidden_states = (
            hidden_states
            + self.position_embedding(
                positions
            )
        )

        # Prevent the Transformer from seeing future positions
        causal_mask = torch.triu(
            torch.ones(
                (
                    sequence_length,
                    sequence_length,
                ),
                dtype=torch.bool,
                device=token_sequences.device,
            ),
            diagonal=1,
        )

        # Process the token history using self-attention
        hidden_states = self.transformer(
            hidden_states,
            mask=causal_mask,
        )

        # Predict EnCodec token scores
        logits = self.output_projection(
            hidden_states
        )

        # Separate the output back into individual codebooks
        logits = logits.view(
            batch_size,
            sequence_length,
            self.num_codebooks,
            CODEBOOK_SIZE,
        )

        # Move the codebook dimension before the time dimension
        logits = logits.permute(
            0,
            2,
            1,
            3,
        )

        # Return token predictions
        return logits


# Train the Transformer using the Step 3 dataset
def train_model(
    training_dataset_path,
    output_path="outputs/codec_transformer.pt",
    epochs=100,
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
        "\nTraining device:",
        device,
    )

    # Load the Step 3 training dataset
    dataset = torch.load(
        training_dataset_path,
        map_location="cpu",
        weights_only=False,
    )

    # Get the Transformer input sequences
    inputs = dataset[
        "inputs"
    ]

    # Get the expected next-token sequences
    targets = dataset[
        "targets"
    ]

    # Get the number of EnCodec codebooks
    num_codebooks = dataset[
        "num_codebooks"
    ]

    # Get the training context length
    context_length = dataset[
        "context_length"
    ]

    # Create a PyTorch dataset
    training_data = TensorDataset(
        inputs,
        targets,
    )

    # Create batches for training
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )

    # Create the Transformer
    model = CodecTransformer(
        num_codebooks=num_codebooks,
        context_length=context_length,
    )

    # Move the Transformer onto the selected device
    model = model.to(
        device
    )

    # Create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    # Create the next-token prediction loss
    loss_function = nn.CrossEntropyLoss()

    # Switch the model into training mode
    model.train()

    # Train for the requested number of epochs
    for epoch in range(
        1,
        epochs + 1,
    ):
        # Track the total loss for this epoch
        total_loss = 0.0

        # Track the number of processed batches
        batch_count = 0

        # Process every training batch
        for batch_inputs, batch_targets in training_loader:
            # Move the input sequences onto the GPU
            batch_inputs = batch_inputs.to(
                device
            )

            # Move the target sequences onto the GPU
            batch_targets = batch_targets.to(
                device
            )

            # Clear gradients from the previous update
            optimizer.zero_grad()

            # Predict the next EnCodec tokens
            predictions = model(
                batch_inputs
            )

            # Flatten the model predictions for cross-entropy
            prediction_values = predictions.reshape(
                -1,
                CODEBOOK_SIZE,
            )

            # Flatten the correct token IDs
            target_values = batch_targets.reshape(
                -1
            )

            # Calculate prediction error
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

            # Count this batch
            batch_count += 1

        # Calculate the average loss
        average_loss = (
            total_loss
            / batch_count
        )

        # Print progress every ten epochs
        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch}/{epochs} "
                f"Loss: {average_loss:.4f}"
            )

    # Create the output folder if necessary
    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save everything needed to recreate the model later
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_codebooks": num_codebooks,
        "context_length": context_length,
        "embedding_size": 64,
        "hidden_size": 256,
        "num_heads": 8,
        "num_layers": 4,
        "codebook_size": CODEBOOK_SIZE,
        "start_token_id": START_TOKEN_ID,
    }

    # Save the trained model
    torch.save(
        checkpoint,
        output_path,
    )

    # Confirm that training finished
    print(
        "Step 4 complete:",
        output_path,
    )

    # Return the trained model path
    return str(
        output_path
    )
