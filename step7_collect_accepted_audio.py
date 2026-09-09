import json
import shutil
from pathlib import Path


# Collect user-approved generations for future training
def collect_accepted_audio(
    feedback_path="outputs/feedback.json",
    output_folder="audio/accepted_training",
):
    # Convert the paths into Path objects
    feedback_path = Path(
        feedback_path
    )

    output_folder = Path(
        output_folder
    )

    # Make sure feedback exists before continuing
    if not feedback_path.exists():
        raise RuntimeError(
            f"Feedback file not found: {feedback_path}"
        )

    # Create the accepted-training folder if necessary
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load the generation feedback
    with feedback_path.open("r") as file:
        feedback_history = json.load(
            file
        )

    # Track how many accepted clips were collected
    accepted_count = 0

    # Check every rated generation
    for item in feedback_history:
        # Ignore rejected generations
        if item["rating"] != "accept":
            continue

        # Find the accepted generated WAV
        source_path = Path(
            item["generation"]
        )

        # Make sure the generated WAV still exists
        if not source_path.exists():
            print(
                f"Missing generation: {source_path}"
            )

            continue

        # Create the destination path
        destination_path = (
            output_folder
            / source_path.name
        )

        # Copy the accepted WAV into the training folder
        shutil.copy2(
            source_path,
            destination_path,
        )

        # Count the accepted example
        accepted_count += 1

    # Report how many accepted examples were collected
    print(
        f"Accepted training clips: {accepted_count}"
    )

    # Return the accepted-training folder
    return str(
        output_folder
    )

# Run the collector when this file is executed directly
if __name__ == "__main__":
    collect_accepted_audio()
