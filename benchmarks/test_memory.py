"""Tests for the memory helpers in ``conftest.py``.

No benchmark is run, so these are ordinary tests and run in a plain
``pytest benchmarks/``.
"""
import time

import numpy as np
import pytest
import torch

pytest.importorskip('psutil')

# Large enough to stand out from allocator and interpreter noise:
NBYTES = 200_000_000


def test_rss_sampler_sees_freed_allocation(peak_memory):
    """Memory freed before the call returns is seen only by the sampler."""
    def allocate_and_free():
        data = np.ones(NBYTES, dtype=np.uint8)
        # Many 1 ms sampling intervals:
        time.sleep(0.1)
        del data

    result = peak_memory(allocate_and_free)
    assert result['memory_metric'] == 'rss_delta'
    assert result['peak_mem_mb'] > 0.9 * NBYTES / 1e6


def test_rss_excludes_memory_held_before_the_call(peak_memory):
    held = np.ones(NBYTES, dtype=np.uint8)  # noqa: F841
    assert peak_memory(lambda: None)['peak_mem_mb'] < 0.1 * NBYTES / 1e6


def test_cuda_allocator_sees_freed_tensor(cuda_device, cuda_peak_memory):
    """A temporary tensor counts even though only a scalar is returned."""
    def allocate_and_reduce():
        return torch.ones(NBYTES // 4, device=cuda_device).sum()

    result = cuda_peak_memory(allocate_and_reduce)
    assert result['memory_metric'] == 'cuda_allocated_delta'
    assert result['peak_mem_mb'] == pytest.approx(NBYTES / 1e6, rel=0.01)
