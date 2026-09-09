# Mini Music AI

Generate a snare drum clip using the trained model.

## Setup

Create and activate the virtual environment:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -e .
```

## Generate Audio - Normal Operation

Generate audio using the current stable model:

```bash
run --seconds <time_length_in_seconds>
```

Example:

```bash
run --seconds 15
```

The generated clip will be saved in:

```text
outputs/generations/
```

After generation, the program will ask whether the clip should be accepted or rejected.

```text
Accept or reject this generation? [a/r]:
```

The current acceptance rate will be displayed after feedback is recorded.

## Generate Audio With Separate Ending Model

Generate audio using the current stable model and then add the separate clean-ending model:

```bash
run --with-ending --seconds <time_length_in_seconds>
```

Example:

```bash
run --with-ending --seconds 15
```

This uses:

```text
models/codec_transformer.pt
```

followed by:

```text
models/ending_transformer.pt
```

## Train From User-Accepted Generations

Fine-tune a copy of the current stable model using previously accepted generated clips:

```bash
run --train-accepted
```

This process:

1. Collects generations marked as accepted.
2. Prepares the accepted WAV files for training.
3. Converts them into EnCodec tokens.
4. Builds Transformer training sequences.
5. Fine-tunes a copy of the current stable model.

The trained model is saved as:

```text
outputs/accepted_transformer.pt
```

The current stable model is not replaced during this step.

## Promote Accepted-Trained Model

Promote the accepted-trained model to become the new stable model:

```bash
run --promote-model
```

This replaces:

```text
models/codec_transformer.pt
```

with:

```text
outputs/accepted_transformer.pt
```

Before replacement, the previous stable model is backed up in:

```text
outputs/model_backups/
```

After promotion, normal generation:

```bash
run --seconds 15
```

will automatically use the newly promoted stable model.

## Full Retraining

Run the complete original training pipeline:

```bash
run --full-training --seconds <time_length_in_seconds>
```

Example:

```bash
run --full-training --seconds 15
```

This runs the full training pipeline before generating audio.
