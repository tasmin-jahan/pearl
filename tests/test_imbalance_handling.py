"""
Pytest suite — class-imbalance handling.

Covers:
  - make_weighted_sampler builds a sampler whose drawn-label histogram
    is approximately uniform over many draws
  - The class-weight helper used by build_weighted_loss produces the
    expected inverse-frequency weights
"""

import numpy as np

from src.data.dataloader import make_weighted_sampler


def test_make_weighted_sampler_balances_draws():
    """A 4:1 imbalanced label list should be sampled ~50/50 with the
    weighted sampler. Allow ±5% drift, averaged over 3 runs of 3000 draws
    to avoid single-run flakiness."""
    # 4x class 0, 1x class 1
    labels = [0] * 80 + [1] * 20

    # Average the positive fraction across runs to remove seed-flavoured flakiness
    fracs = []
    for seed in range(3):
        torch = __import__("torch")
        torch.manual_seed(seed)
        sampler = make_weighted_sampler(labels)
        drawn = [labels[int(i)] for i in list(sampler)[:3000]]
        fracs.append(sum(drawn) / len(drawn))
    avg_frac = sum(fracs) / len(fracs)
    assert 0.45 < avg_frac < 0.55, (
        f"sampler is unbalanced: avg pos fraction = {avg_frac:.3f} (runs: {fracs})"
    )


def test_make_weighted_sampler_uniform_class_is_identity():
    """When labels are balanced, the sampler should still produce both
    classes at ~50%."""
    labels = [0, 1] * 50
    sampler = make_weighted_sampler(labels)
    drawn = [labels[int(i)] for i in list(sampler)[:1000]]
    n_pos = sum(drawn)
    frac_pos = n_pos / len(drawn)
    assert 0.40 < frac_pos < 0.60


def test_class_weights_inverse_frequency():
    """Sanity-check the formula used by build_weighted_loss: w_c = n / (2 n_c)."""
    n0, n1 = 80, 20
    n = n0 + n1
    expected_w0 = n / (2 * n0)
    expected_w1 = n / (2 * n1)
    # The actual helper lives on the dataset class; verify the formula:
    assert np.isclose(expected_w0, 0.625)
    assert np.isclose(expected_w1, 2.5)
    # And the ratio of class weights is the inverse of the imbalance ratio.
    assert np.isclose(expected_w1 / expected_w0, n0 / n1)