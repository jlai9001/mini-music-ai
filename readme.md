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

To generate audio from the current model run this from the root of the project:

```bash
run --seconds <time_length_in_seconds>
```
Example:

```bash
run --seconds 15
```
## Retrain Model and Generate Audio

To run the full training pipeline and generate audio:

```bash
run --full-training --seconds <time_length_in_seconds>
```
Example:

```bash
run --full-training --seconds 15
```
