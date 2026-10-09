"""Tests for the memory helpers in ``conftest.py``.

No benchmark is run, so these are ordinary tests and run in a plain
``pytest benchmarks/``.
"""
import sys

import numpy as np
import pytest
import torch

# Large enough to stand out from interpreter and allocator noise:
NBYTES = 200_000_000
MB = NBYTES / 1e6


def test_peak_sees_freed_numpy_allocation(peak_memory):
    """Memory freed before the call returns still counts toward the peak."""
    def allocate_and_free():
        data = np.ones(NBYTES, dtype=np.uint8)
        del data

    assert peak_memory(allocate_and_free)['peak_mem_mb'] == pytest.approx(
        MB, rel=0.05)


def test_peak_excludes_memory_held_before_the_call(peak_memory):
    held = np.ones(NBYTES, dtype=np.uint8)  # noqa: F841
    assert peak_memory(lambda: None)['peak_mem_mb'] < 0.01 * MB


@pytest.mark.skipif(sys.platform == 'win32',
                    reason='the Windows fallback, tracemalloc, does not see '
                           'Torch allocations')
def test_memray_sees_torch_cpu_allocation_every_time(peak_memory):
    """Native Torch memory is counted (#949), also when the allocator could
    reuse the block from the previous call."""
    for _ in range(2):
        result = peak_memory(torch.empty, NBYTES, dtype=torch.uint8)
        assert result['memory_metric'] == 'memray_heap_peak'
        assert result['peak_mem_mb'] == pytest.approx(MB, rel=0.05)


def test_cuda_allocator_sees_tensor(cuda_device, cuda_peak_memory):
    result = cuda_peak_memory(torch.empty, NBYTES, dtype=torch.uint8,
                              device=cuda_device)
    assert result['memory_metric'] == 'cuda_allocated_delta'
    assert result['peak_mem_mb'] == pytest.approx(MB, rel=0.01)
