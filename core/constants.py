SAMPLE_RATE = 16_000

# Peak of audio returned by mix(); louder mixtures are scaled down to it.
MAX_PEAK = 0.99

# Peak at or above which audio counts as clipped. Below 1.0 so that PCM_16
# files clipped on save (max 32767/32768 when read back) are detected.
CLIP_THRESHOLD = 0.999
