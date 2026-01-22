#!/usr/bin/env python3
"""
Patch timm==0.3.2 for PyTorch 2.x compatibility.

This script fixes the `torch._six` import issue in timm 0.3.2,
which is required for this codebase but incompatible with PyTorch 2.x.

Changes:
- Replace `from torch._six import container_abcs` with direct imports from collections.abc
"""

import os
import sys
from pathlib import Path


def find_timm_installation():
    """Find the timm installation directory."""
    try:
        import timm
        timm_path = Path(timm.__file__).parent
        return timm_path
    except ImportError:
        print("ERROR: timm is not installed. Please install it first.")
        sys.exit(1)


def patch_helpers_file(timm_path):
    """Patch timm/models/layers/helpers.py to fix torch._six import."""
    helpers_file = timm_path / "models" / "layers" / "helpers.py"

    if not helpers_file.exists():
        print(f"WARNING: helpers.py not found at {helpers_file}")
        return False

    print(f"Patching {helpers_file}...")

    with open(helpers_file, 'r') as f:
        content = f.read()

    # Check if already patched
    if "from collections.abc import" in content:
        print("✓ File already patched, skipping")
        return True

    # Replace the problematic import
    old_import = "from torch._six import container_abcs"
    new_import = "from collections.abc import Iterable, Mapping"

    if old_import in content:
        content = content.replace(old_import, new_import)

        # Also need to replace container_abcs.Iterable -> Iterable, etc.
        content = content.replace("container_abcs.Iterable", "Iterable")
        content = content.replace("container_abcs.Mapping", "Mapping")

        with open(helpers_file, 'w') as f:
            f.write(content)

        print("✓ Successfully patched helpers.py")
        return True
    else:
        print(f"WARNING: Expected import not found in {helpers_file}")
        return False


def verify_patch():
    """Verify the patch was successful by trying to import timm."""
    try:
        import timm
        from timm.models.layers import to_2tuple
        print(f"✓ Patch verification successful (timm {timm.__version__})")
        return True
    except Exception as e:
        print(f"✗ Patch verification failed: {e}")
        return False


def main():
    print("=" * 60)
    print("timm 0.3.2 PyTorch 2.x Compatibility Patch")
    print("=" * 60)
    print()

    # Find timm installation
    timm_path = find_timm_installation()
    print(f"Found timm installation: {timm_path}")

    # Verify version
    import timm
    if timm.__version__ != "0.3.2":
        print(f"WARNING: Expected timm==0.3.2, found {timm.__version__}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)

    print()

    # Apply patch
    success = patch_helpers_file(timm_path)

    if not success:
        print("\n✗ Patching failed")
        sys.exit(1)

    print()

    # Verify
    if verify_patch():
        print("\n" + "=" * 60)
        print("Patching completed successfully!")
        print("=" * 60)
    else:
        print("\n✗ Verification failed - you may need to reinstall timm")
        sys.exit(1)


if __name__ == "__main__":
    main()
