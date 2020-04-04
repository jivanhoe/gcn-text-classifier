import pytest
import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path + '/../')

if __name__ == "__main__":
    pytest.main(["-x", "model_tests.py"])
