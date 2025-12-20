"""Simple script to check if GPU is available and working."""

from src.util import get_device

device = get_device()  # Just call to see if any errors occur
print(f"Using device: {device}")
