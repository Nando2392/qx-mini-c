import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / 'build/qxqxf.exe'


def command(*extra):
    return [str(BIN), 'prompt-state-loop-probe', '--in', 'missing.qxf',
            '--tokenizer', 'missing.qxt', '--text-file', 'missing.txt',
            '--generate', '1', '--layers', '48', '--ctx', '16', '--kv', 'int8',
            '--activation', 'f32', '--thread-policy', 'moe-pool', '--threads', '2',
            '--full-moe', '--final-head', *extra]


def test_moe_pool_advances_to_input_io():
    result = subprocess.run(command(), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'cannot open text file' in result.stderr
    assert 'unsupported thread policy' not in result.stderr


@pytest.mark.parametrize('args,message', [
    (['--threads', '1'], '2..64'),
    (['--threads', '65'], '2..64'),
    (['--activation', 'q8_k_compat'], 'F32'),
    (['--io-backend', 'mmap'], 'buffered'),
    (['--cuda-policy', 'final-head-f32'], 'CUDA'),
])
def test_moe_pool_rejects_before_io(args, message):
    result = subprocess.run(command(*args), cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert message in result.stderr
    assert 'text file read failed' not in result.stderr
