#!/usr/bin/env python3
"""
Test script to verify py3 conda environment is active.
Run with: python3 test_py3_env.py
Or with wrapper: ./python_py3_wrapper.sh test_py3_env.py
"""

import sys
import os
import platform

print("Python Environment Test")
print("=" * 60)

# Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Platform: {platform.platform()}")

# Conda environment
print(f"\nConda environment:")
print(f"  CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")
print(f"  CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")

# PATH
path = os.environ.get('PATH', '')
py3_path = '/home/bh692/.conda/envs/py3/bin'
if py3_path in path:
    print(f"  ✓ py3 bin directory is in PATH")
else:
    print(f"  ✗ py3 bin directory is NOT in PATH")

# Check if we're using the py3 environment Python
expected_prefix = '/home/bh692/.conda/envs/py3'
if sys.executable.startswith(expected_prefix):
    print(f"  ✓ Using py3 environment Python")
else:
    print(f"  ✗ NOT using py3 environment Python")
    print(f"    Expected prefix: {expected_prefix}")
    print(f"    Actual executable: {sys.executable}")

# Try to import common scientific packages
print(f"\nPackage availability:")
try:
    import numpy
    print(f"  numpy: {numpy.__version__}")
except ImportError:
    print(f"  numpy: Not available")

try:
    import scipy
    print(f"  scipy: {scipy.__version__}")
except ImportError:
    print(f"  scipy: Not available")

try:
    import matplotlib
    print(f"  matplotlib: {matplotlib.__version__}")
except ImportError:
    print(f"  matplotlib: Not available")

print("\n" + "=" * 60)
print("Test complete.")