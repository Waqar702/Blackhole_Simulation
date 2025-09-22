#!/usr/bin/env python3
"""
Installation script for black hole simulation.

This script helps set up the Python environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with exit code {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"✗ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if not in_venv:
        print("Warning: Not in a virtual environment. Consider using one:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print()
    
    # Install dependencies
    if not run_command("pip install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def run_tests():
    """Run basic tests to verify installation."""
    print("\nRunning tests...")
    
    if not run_command("python test_setup.py", "Running setup tests"):
        return False
    
    return True

def main():
    """Main installation process."""
    print("Black Hole Simulation - Installation Script")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Run installation steps
    steps = [
        ("Check Python version", check_python_version),
        ("Install dependencies", install_dependencies),
        ("Run tests", run_tests),
    ]
    
    for description, func in steps:
        print(f"\n{description}...")
        if not func():
            print(f"\n✗ Installation failed at: {description}")
            return 1
    
    print("\n" + "=" * 50)
    print("✓ Installation completed successfully!")
    print("\nYou can now run the demos:")
    print("  python -m blackhole_python.app 2d    # 2D lensing demo")
    print("  python -m blackhole_python.app 3d    # 3D GPU simulation")
    print("\nOr run tests:")
    print("  python -m pytest tests/              # Run all tests")
    print("  python test_setup.py                 # Run setup tests")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
