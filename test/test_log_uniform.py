#!/usr/bin/env python3
"""
Simple test code for log uniform distribution.
"""

import pytest

import flaml.tune
import numpy as np
import numpy.typing as npt
import scipy.stats


def assert_is_uniform(arr: npt.NDArray) -> None:
    """Implements a few tests for uniformity."""
    _values, counts = np.unique(arr, return_counts=True)

    kurtosis = scipy.stats.kurtosis(arr)

    _chi_sq, p_value = scipy.stats.chisquare(counts)

    frequencies = counts / len(arr)
    assert np.isclose(frequencies.sum(), 1)
    _f_chi_sq, f_p_value = scipy.stats.chisquare(frequencies)

    assert np.isclose(kurtosis, -1.2, atol=0.1)
    assert p_value > 0.5
    assert f_p_value > 0.5


def assert_is_log_uniform(arr: npt.NDArray, base: float = np.e) -> None:
    """Checks whether an array is log uniformly distributed."""
    logs = np.log(arr) / np.log(base)
    assert_is_uniform(logs)


def test_is_uniform() -> None:
    """
    Test our uniform distribution check function using numpy to generate a
    uniform distribution.
    """
    uniform = np.random.uniform(1, 20, 1000)
    assert_is_uniform(uniform)


def test_is_log_uniform() -> None:
    """
    Test our log uniform distribution check function using numpy to generate a
    log uniform distribution.
    """
    log_uniform = np.exp(np.random.uniform(np.log(1), np.log(20), 1000))
    assert_is_log_uniform(log_uniform)


def test_flaml_uniform_int() -> None:
    """Check whether flaml's uniform int is uniform."""
    rand_int = flaml.tune.randint(1, 20)
    samples = rand_int.sample(size=1000)
    assert_is_uniform(samples)


def test_flaml_log_uniform_int() -> None:
    """Check whether flaml's log uniform int is log uniform."""
    base = 10
    rand_int = flaml.tune.lograndint(1, 20, base=base)
    samples = rand_int.sample(size=10000)
    assert_is_log_uniform(samples, base=base)  # FIXME: This fails


def test_flaml_uniform_float() -> None:
    """Check whether flaml's uniform int is uniform."""
    rand_float = flaml.tune.uniform(1, 20)
    samples = rand_float.sample(size=1000)
    assert_is_uniform(samples)


def test_flaml_log_uniform_float() -> None:
    """Check whether flaml's log uniform int is log uniform."""
    base = 10
    rand_float = flaml.tune.loguniform(1, 20, base=base)
    samples = rand_float.sample(size=10000)
    assert_is_log_uniform(samples, base=base)  # FIXME: This fails


if __name__ == "__main__":
    # For attaching debugger debugging:
    pytest.main(["-s", __file__])
