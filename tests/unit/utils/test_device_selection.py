"""CPU requests must take precedence before CUDA initialization at import."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('cpu_request,visible,expected_cpu', [
    ('true', True, True), ('1', True, True), ('yes', True, True),
    ('on', True, True), ('false', True, False), (None, True, False),
    (None, False, True), ('false', False, True),
])
def test_import_selects_requested_device_before_cuda(cpu_request, visible, expected_cpu, tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('ACCELERATE_USE_CPU', None)
    if cpu_request is not None:
        env['ACCELERATE_USE_CPU'] = cpu_request
    code = '''
from types import SimpleNamespace
from unittest.mock import patch
import torch
calls = []
def state(*, cpu=False):
    calls.append(cpu)
    return SimpleNamespace(device=torch.device('cpu' if cpu else 'cuda'))
with patch('torch.cuda.is_available', return_value=VISIBLE), patch('accelerate.PartialState', side_effect=state), patch('torch.cuda.set_device', side_effect=AssertionError('CUDA initialized during CPU selection')):
    from perseus.utils import torch as module
assert calls and all(cpu == EXPECTED for cpu in calls), calls
assert module.device.type == ('cpu' if EXPECTED else 'cuda')
'''.replace('VISIBLE', repr(visible)).replace('EXPECTED', repr(expected_cpu))
    result = subprocess.run([sys.executable, '-c', code], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
