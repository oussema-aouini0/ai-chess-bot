"""Shared OOD (out-of-distribution) eval-bucket FENs for nn_eval diagnostics.

The bucket = the real tactical FENs from chess_bot.EVALUATE_TACTICAL_FENS
(runtime, not a constant here) + OOD_SYNTH (41 synthetic sparse/extreme
endgames) + RANDOM_EXTREME (3 random extreme-material positions) = 57 total.
Same list used for the NN-eval OOD numbers and the training-coverage check.
"""

OOD_SYNTH = [
    "3k4/8/8/8/8/8/3Q4/3K4 w - - 0 1", "3k4/8/8/3Q4/8/8/8/K7 b - - 0 1",
    "8/2k5/8/8/8/8/2K5/Q7 w - - 0 1", "3k4/8/8/8/8/8/3R4/3K4 w - - 0 1",
    "k7/8/8/8/8/8/4R3/2K5 w - - 0 1", "3k4/8/8/8/8/8/2BB4/3K4 w - - 0 1",
    "3k4/8/8/8/8/8/2BN4/3K4 w - - 0 1", "k7/8/8/8/8/8/4R3/3K1R2 w - - 0 1",
    "k7/8/8/8/8/8/2Q5/3K3Q w - - 0 1", "k7/8/8/8/8/3q4/4r3/2K5 w - - 0 1",
    "k7/8/8/5q2/8/8/8/2KQ4 w - - 0 1", "8/8/8/8/3k4/4P3/4K3/8 w - - 0 1",
    "8/8/4k3/8/8/4P3/4K3/8 b - - 0 1", "8/4k3/4p3/8/8/4P3/4K3/8 w - - 0 1",
    "4k3/8/8/8/8/2R5/8/4KQ1R w - - 0 1", "3qk3/8/8/8/8/8/8/3RK3 w - - 0 1",
    "8/8/8/8/8/2k5/4P3/2K5 w - - 0 1", "8/3k4/8/8/8/8/6p1/5K2 b - - 0 1",
    "8/4k3/3R4/8/8/4p3/4K3/8 w - - 0 1", "8/3k4/3P4/8/8/4R3/8/7K w - - 0 1",
    "8/8/8/3k4/8/2Q5/2K5/8 w - - 0 1", "8/8/2k5/8/8/6K1/8/1b6 w - - 0 1",
    "8/8/8/3k4/8/3BKN2/8/8 w - - 0 1", "8/8/8/5k2/8/4BKN1/8/8 w - - 0 1",
    "8/8/2k5/8/8/6K1/8/B6B w - - 0 1", "8/8/8/3k4/8/8/4RB2/7K w - - 0 1",
    "8/8/8/4k3/8/8/4Q3/3K4 w - - 0 1", "8/8/8/8/3k4/8/4R3/3K4 w - - 0 1",
    "8/8/8/8/3k4/8/1B6/3K4 w - - 0 1", "8/8/2k5/8/8/8/2K5/1Q6 w - - 0 1",
    "8/1q1k4/8/8/8/8/8/2K5 w - - 0 1", "8/8/8/5k2/8/8/2R5/2K5 w - - 0 1",
    "k7/8/8/8/8/8/8/1RK2B2 w - - 0 1", "k7/8/8/8/8/8/1QK5/8 b - - 0 1",
    "8/8/8/1k6/8/8/2KB4/8 w - - 0 1", "8/8/8/8/1k6/8/2K5/3Q4 w - - 0 1",
    "4k3/8/8/8/8/3P4/4K3/8 w - - 0 1", "8/3k4/8/8/3P4/8/4K3/8 b - - 0 1",
    "8/8/8/8/8/3k4/4P3/4K1R1 w - - 0 1", "8/8/8/8/8/1k6/1Q6/1K6 b - - 0 1",
    "8/8/8/8/8/2k5/8/K2Q4 w - - 0 1",
]

RANDOM_EXTREME = [
    "7k/8/8/8/8/8/7P/7K w - - 0 1", "8/8/p1k5/8/8/P1K5/8/8 w - - 0 1",
    "8/8/8/3k4/8/8/R7/3K4 w - - 0 1",
]