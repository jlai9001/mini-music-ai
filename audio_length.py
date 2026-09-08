import math

import numpy as np


TOKENS_PER_SECOND = 50

SAMPLE_RATE = 48000

CLEAN_END_EXTRA_SECONDS = 2.0

CLEAN_END_SEARCH_BEFORE_SECONDS = 2.0

ENERGY_WINDOW_SECONDS = 0.02

MIN_QUIET_SECONDS = 0.02

ZERO_CROSSING_SEARCH_SECONDS = 0.02


# Convert a requested duration into the number of EnCodec token positions
def seconds_to_tokens(
    seconds,
):
    return int(
        seconds
        * TOKENS_PER_SECOND
    )


# Convert a requested duration into the number of waveform samples
def seconds_to_samples(
    seconds,
):
    return int(
        seconds
        * SAMPLE_RATE
    )


# Find the nearest waveform zero crossing around a proposed cut point
def find_nearest_zero_crossing(
    audio,
    sample_index,
):
    # Convert the zero-crossing search range into samples
    search_radius = seconds_to_samples(
        ZERO_CROSSING_SEARCH_SECONDS
    )

    # Calculate the beginning of the search area
    search_start = max(
        1,
        sample_index
        - search_radius,
    )

    # Calculate the end of the search area
    search_end = min(
        len(
            audio
        )
        - 1,
        sample_index
        + search_radius,
    )

    # Get the waveform region around the proposed ending
    search_audio = audio[
        search_start - 1:
        search_end + 1
    ]

    # Find positions where the waveform changes sign
    zero_crossings = np.where(
        np.signbit(
            search_audio[
                :-1
            ]
        )
        != np.signbit(
            search_audio[
                1:
            ]
        )
    )[
        0
    ]

    # Use the original cut point when no zero crossing was found
    if len(
        zero_crossings
    ) == 0:
        return sample_index

    # Convert local zero-crossing positions into waveform positions
    zero_crossings = (
        zero_crossings
        + search_start
    )

    # Find the zero crossing closest to the proposed ending
    closest_index = int(
        zero_crossings[
            np.argmin(
                np.abs(
                    zero_crossings
                    - sample_index
                )
            )
        ]
    )

    # Return the cleaner waveform cut position
    return closest_index


# Find a natural quiet point near the requested audio duration
def find_clean_end(
    audio,
    requested_seconds,
):
    # Convert the requested duration into a waveform position
    requested_sample = seconds_to_samples(
        requested_seconds
    )

    # Calculate where the clean-ending search should begin
    search_start = max(
        0,
        requested_sample
        - seconds_to_samples(
            CLEAN_END_SEARCH_BEFORE_SECONDS
        ),
    )

    # Calculate where the clean-ending search should stop
    search_end = min(
        len(
            audio
        ),
        requested_sample
        + seconds_to_samples(
            CLEAN_END_EXTRA_SECONDS
        ),
    )

    # Extract the region where a natural ending should be found
    search_audio = audio[
        search_start:
        search_end
    ]

    # Convert the RMS analysis window into samples
    window_sample_count = seconds_to_samples(
        ENERGY_WINDOW_SECONDS
    )

    # Store the energy of every analysis window
    rms_values = []

    # Store the beginning of every analysis window
    window_starts = []

    # Measure waveform energy across the ending search region
    for start_index in range(
        0,
        len(
            search_audio
        )
        - window_sample_count
        + 1,
        window_sample_count,
    ):
        # Extract one short waveform window
        window = search_audio[
            start_index:
            start_index
            + window_sample_count
        ]

        # Measure the RMS energy of this window
        rms = np.sqrt(
            np.mean(
                window ** 2
            )
        )

        # Remember the measured energy
        rms_values.append(
            rms
        )

        # Remember where this window begins
        window_starts.append(
            start_index
        )

    # Fall back to the requested duration if no analysis windows exist
    if not rms_values:
        return min(
            requested_sample,
            len(
                audio
            ),
        )

    # Convert the measured energies into a NumPy array
    rms_values = np.asarray(
        rms_values
    )

    # Estimate the local background energy
    low_energy = np.percentile(
        rms_values,
        20,
    )

    # Estimate the louder activity level
    high_energy = np.percentile(
        rms_values,
        90,
    )

    # Build an adaptive threshold between quiet and active audio
    quiet_threshold = (
        low_energy
        + 0.15
        * (
            high_energy
            - low_energy
        )
    )

    # Make sure the threshold never becomes exactly zero
    quiet_threshold = max(
        float(
            quiet_threshold
        ),
        0.00001,
    )

    # Mark every analysis window that is quiet enough
    quiet_windows = (
        rms_values
        <= quiet_threshold
    )

    # Calculate how many consecutive windows count as a real pause
    minimum_quiet_windows = max(
        1,
        math.ceil(
            MIN_QUIET_SECONDS
            / ENERGY_WINDOW_SECONDS
        ),
    )

    # Store every detected natural pause
    pause_candidates = []

    # Track the beginning of the current quiet section
    quiet_run_start = None

    # Inspect every analysis window
    for index, is_quiet in enumerate(
        quiet_windows
    ):
        # Start tracking a new quiet section
        if (
            is_quiet
            and quiet_run_start is None
        ):
            quiet_run_start = index

        # Finish the current quiet section when sound resumes
        if (
            not is_quiet
            and quiet_run_start is not None
        ):
            # Calculate how many quiet windows were found
            quiet_run_length = (
                index
                - quiet_run_start
            )

            # Keep the section if it is long enough to be a pause
            if (
                quiet_run_length
                >= minimum_quiet_windows
            ):
                # Find the middle of the quiet section
                middle_window = (
                    quiet_run_start
                    + quiet_run_length
                    // 2
                )

                # Convert the quiet section into an absolute waveform position
                candidate_sample = (
                    search_start
                    + window_starts[
                        middle_window
                    ]
                    + window_sample_count
                    // 2
                )

                # Remember the natural pause
                pause_candidates.append(
                    candidate_sample
                )

            # Reset the quiet-section tracker
            quiet_run_start = None

    # Handle a quiet section that continues to the end of the search region
    if quiet_run_start is not None:
        # Calculate the final quiet-section length
        quiet_run_length = (
            len(
                quiet_windows
            )
            - quiet_run_start
        )

        # Keep the final section if it is long enough
        if (
            quiet_run_length
            >= minimum_quiet_windows
        ):
            # Find the middle of the quiet section
            middle_window = (
                quiet_run_start
                + quiet_run_length
                // 2
            )

            # Convert the quiet section into an absolute waveform position
            candidate_sample = (
                search_start
                + window_starts[
                    middle_window
                ]
                + window_sample_count
                // 2
            )

            # Remember the natural pause
            pause_candidates.append(
                candidate_sample
            )

    # Choose the detected pause closest to the requested duration
    if pause_candidates:
        clean_end_sample = min(
            pause_candidates,
            key=lambda sample: abs(
                sample
                - requested_sample
            ),
        )

    # Fall back to the lowest-energy window when no full pause exists
    else:
        quietest_window_index = int(
            np.argmin(
                rms_values
            )
        )

        clean_end_sample = (
            search_start
            + window_starts[
                quietest_window_index
            ]
            + window_sample_count
            // 2
        )

    # Move the ending onto the nearest waveform zero crossing
    clean_end_sample = find_nearest_zero_crossing(
        audio,
        clean_end_sample,
    )

    # Return the natural ending position
    return clean_end_sample
