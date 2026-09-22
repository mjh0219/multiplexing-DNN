#!/usr/bin/env python3
"""Verify vector-based MUX and DEMUX with random inputs and outputs."""

import argparse
import secrets
from itertools import combinations

import torch


def mux_inputs(x_vectors):
    """Combine x1, x2, ..., xS into one ordered vector X_mul."""
    return torch.cat(x_vectors, dim=0)


def demux_inputs(x_mul, number_of_slots, input_size):
    """Separate X_mul back into x1, x2, ..., xS."""
    expected_size = number_of_slots * input_size
    if x_mul.numel() != expected_size:
        raise ValueError("X_mul has an incorrect length")
    return list(x_mul.view(number_of_slots, input_size).unbind(dim=0))


def mux_outputs(y_vectors):
    """Combine y1, y2, ..., yS into one ordered vector Y_mul."""
    return torch.cat(y_vectors, dim=0)


def demux_outputs(y_mul, number_of_slots, output_size):
    """Separate Y_mul back into y1, y2, ..., yS."""
    expected_size = number_of_slots * output_size
    if y_mul.numel() != expected_size:
        raise ValueError("Y_mul has an incorrect length")
    return list(y_mul.view(number_of_slots, output_size).unbind(dim=0))


def maximum_recovery_error(original, recovered):
    return max(
        (source - result).abs().max().item()
        for source, result in zip(original, recovered)
    )


def compute_output_vector(x_vector, projection):
    """A deterministic toy mapping y=f(x) used to verify slot correspondence."""
    return torch.mv(projection, x_vector) / (x_vector.numel() ** 0.5)


def preview(vector, values=6):
    shown = vector.detach().cpu().flatten()[:values].tolist()
    return "[" + ", ".join("{:+.4f}".format(value) for value in shown) + "]"


def all_active_slot_sets(number_of_slots):
    """Return every nonempty subset of slot indices."""
    return [
        active
        for size in range(1, number_of_slots + 1)
        for active in combinations(range(number_of_slots), size)
    ]


def apply_active_slots(vectors, active_slots):
    """Keep active vectors and put zeros in inactive fixed-width slots."""
    active = set(active_slots)
    return [
        vector if slot in active else torch.zeros_like(vector)
        for slot, vector in enumerate(vectors)
    ]


def combination_name(active_slots):
    inputs = ", ".join("x{}".format(slot + 1) for slot in active_slots)
    outputs = ", ".join("y{}".format(slot + 1) for slot in active_slots)
    return "({}) -> ({})".format(inputs, outputs)


def run_trial(
    seed,
    input_size,
    output_size,
    number_of_slots,
    show_values,
    verify_all_combinations,
):
    torch.manual_seed(seed)

    # x1, x2, and x3 represent three randomly generated input vectors.
    x_vectors = [torch.randn(input_size) for _ in range(number_of_slots)]
    x_mul = mux_inputs(x_vectors)
    recovered_x = demux_inputs(x_mul, number_of_slots, input_size)
    input_error = maximum_recovery_error(x_vectors, recovered_x)

    # Use one shared deterministic function so that x1->y1, x2->y2, and x3->y3.
    # In the complete experiment, DNN_new learns this input-to-output mapping.
    projection = torch.randn(output_size, input_size)
    y_vectors = [compute_output_vector(x_vector, projection) for x_vector in x_vectors]
    y_mul = mux_outputs(y_vectors)
    recovered_y = demux_outputs(y_mul, number_of_slots, output_size)
    output_error = maximum_recovery_error(y_vectors, recovered_y)
    expected_from_recovered_x = [
        compute_output_vector(x_vector, projection) for x_vector in recovered_x
    ]
    correspondence_error = maximum_recovery_error(
        expected_from_recovered_x, recovered_y
    )

    if show_values:
        print("Random seed: {}".format(seed))
        for slot, vector in enumerate(x_vectors, start=1):
            print("x{} first values: {}".format(slot, preview(vector)))
        print("X_mul length: {}".format(x_mul.numel()))
        print("X_mul first values: {}".format(preview(x_mul)))
        for slot, vector in enumerate(y_vectors, start=1):
            print("y{} first values: {}".format(slot, preview(vector)))
        print("Y_mul length: {}".format(y_mul.numel()))
        print("Y_mul first values: {}".format(preview(y_mul)))
        print("Maximum input recovery error: {:.1f}".format(input_error))
        print("Maximum output recovery error: {:.1f}".format(output_error))
        print("Maximum x-to-y correspondence error: {:.1f}".format(correspondence_error))

    if input_error != 0.0:
        raise AssertionError("Input DEMUX did not exactly recover the input vectors")
    if output_error != 0.0:
        raise AssertionError("Output DEMUX did not exactly recover the output vectors")
    if correspondence_error != 0.0:
        raise AssertionError("An output vector was assigned to the wrong input slot")

    if verify_all_combinations:
        active_sets = all_active_slot_sets(number_of_slots)
        for active_slots in active_sets:
            masked_x = apply_active_slots(x_vectors, active_slots)
            masked_y = apply_active_slots(y_vectors, active_slots)

            recovered_masked_x = demux_inputs(
                mux_inputs(masked_x), number_of_slots, input_size
            )
            recovered_masked_y = demux_outputs(
                mux_outputs(masked_y), number_of_slots, output_size
            )

            if maximum_recovery_error(masked_x, recovered_masked_x) != 0.0:
                raise AssertionError("A selected input combination was not recovered")
            if maximum_recovery_error(masked_y, recovered_masked_y) != 0.0:
                raise AssertionError("A selected output combination was not recovered")

        if show_values:
            print("\nVerified variable combinations using fixed slot positions:")
            for active_slots in active_sets:
                mask = [
                    1 if slot in active_slots else 0
                    for slot in range(number_of_slots)
                ]
                print("  {}  active mask={}".format(combination_name(active_slots), mask))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--input-size",
        type=int,
        default=3 * 32 * 32,
        help="Length of each input vector; 3072 represents a CIFAR RGB image.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=10,
        help="Length of each output vector; 10 represents CIFAR-10 class logits.",
    )
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Verify every nonempty subset of the available slots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.input_size < 1 or args.output_size < 1 or args.slots < 2:
        raise ValueError("vector sizes must be positive and --slots must be at least 2")

    # A new unpredictable seed is selected when --seed is omitted.
    starting_seed = args.seed if args.seed is not None else secrets.randbits(31)
    for trial in range(args.trials):
        run_trial(
            seed=starting_seed + trial,
            input_size=args.input_size,
            output_size=args.output_size,
            number_of_slots=args.slots,
            show_values=(trial == 0),
            verify_all_combinations=args.all_combinations,
        )

    print("PASS: all {} random MUX/DEMUX trial(s) recovered exactly".format(args.trials))
    if args.all_combinations:
        combinations_per_trial = 2 ** args.slots - 1
        print(
            "PASS: verified {} variable input/output combinations".format(
                args.trials * combinations_per_trial
            )
        )


if __name__ == "__main__":
    main()
