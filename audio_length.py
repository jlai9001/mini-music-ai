TOKENS_PER_SECOND = 50

SAMPLE_RATE = 32000


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
