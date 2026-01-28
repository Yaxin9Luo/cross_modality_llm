#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoints to standard PyTorch format.

This utility converts DeepSpeed ZeRO checkpoints (which store model states
across multiple files) into a single standard PyTorch checkpoint file that
can be loaded with torch.load().

Usage:
    # Convert DeepSpeed checkpoint to PyTorch format
    python convert_deepspeed_checkpoint.py \
        --checkpoint ./results/checkpoint-99 \
        --output ./converted_checkpoint.pth

    # Convert and optionally specify tag
    python convert_deepspeed_checkpoint.py \
        --checkpoint ./results \
        --tag checkpoint-99 \
        --output ./converted_checkpoint.pth

    # Convert for evaluation (model weights only)
    python convert_deepspeed_checkpoint.py \
        --checkpoint ./results/checkpoint-99 \
        --output ./model_weights.pth \
        --weights_only
"""

import argparse
import os
import torch
from pathlib import Path
import json


def convert_deepspeed_checkpoint(checkpoint_dir, output_path, tag=None, weights_only=False):
    """
    Convert DeepSpeed checkpoint to standard PyTorch format.

    Args:
        checkpoint_dir: Path to DeepSpeed checkpoint directory
        output_path: Path to save converted checkpoint
        tag: DeepSpeed checkpoint tag (optional)
        weights_only: If True, save only model weights (for evaluation)
    """
    try:
        import deepspeed
    except ImportError:
        print("ERROR: DeepSpeed is not installed. Please install it with:")
        print("  pip install deepspeed>=0.12.0")
        return False

    print(f"Converting DeepSpeed checkpoint from: {checkpoint_dir}")
    print(f"Output path: {output_path}")

    # Load DeepSpeed checkpoint metadata
    checkpoint_path = Path(checkpoint_dir)
    if tag is None:
        # Infer tag from directory name
        tag = checkpoint_path.name

    # Check if this is a DeepSpeed checkpoint
    if not (checkpoint_path / "zero_pp_rank_0_mp_rank_00_optim_states.pt").exists():
        # Try alternative structure
        if not any((checkpoint_path / tag).glob("*_optim_states.pt")):
            print(f"WARNING: {checkpoint_dir} doesn't appear to be a DeepSpeed checkpoint")
            print("Looking for standard PyTorch checkpoint instead...")

            # Try to load as standard checkpoint
            if (checkpoint_path / f"{tag}.pth").exists():
                print(f"Found standard PyTorch checkpoint: {tag}.pth")
                checkpoint = torch.load(checkpoint_path / f"{tag}.pth", map_location='cpu')
                torch.save(checkpoint, output_path)
                print(f"Copied checkpoint to: {output_path}")
                return True
            else:
                print(f"ERROR: Could not find checkpoint at {checkpoint_path}")
                return False

    print("Loading DeepSpeed checkpoint...")

    # Create a dummy model to load the checkpoint
    # Note: This requires the model architecture to be available
    from model_qwen import MAE_Qwen3_Classifier

    # Create a minimal args object
    class Args:
        def __init__(self):
            self.pretrained_model_path = "./Qwen3-8B"
            self.nb_classes = 1000  # Will be overridden
            self.linear_probe = False
            self.gradient_checkpointing = False

    args = Args()

    # Try to load checkpoint metadata to get nb_classes
    metadata_path = checkpoint_path / "latest"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            latest_tag = f.read().strip()
            print(f"Latest checkpoint tag: {latest_tag}")

    # Load client state if available
    client_state_path = checkpoint_path / tag / "client_state.pt"
    if client_state_path.exists():
        client_state = torch.load(client_state_path, map_location='cpu')
        print(f"Loaded client state: {client_state}")
        epoch = client_state.get('epoch', 0)
    else:
        print("WARNING: No client state found")
        epoch = 0

    print("\nWARNING: This conversion requires instantiating the model.")
    print("Please ensure the model configuration matches the checkpoint.")
    print("You may need to modify the nb_classes parameter.\n")

    # For now, create a simplified conversion that extracts model weights
    # This is a placeholder - full conversion requires model instantiation
    print("Creating simplified conversion (model weights only)...")

    # DeepSpeed ZeRO checkpoints store model states in multiple files
    # We need to consolidate them
    model_state = {}

    # Look for model state files
    state_files = list((checkpoint_path / tag).glob("*_model_states.pt"))
    if not state_files:
        state_files = list((checkpoint_path / tag).glob("mp_rank_*_model_states.pt"))

    if state_files:
        print(f"Found {len(state_files)} model state file(s)")
        for state_file in state_files:
            print(f"  Loading: {state_file.name}")
            state = torch.load(state_file, map_location='cpu')
            if 'module' in state:
                model_state.update(state['module'])
            else:
                model_state.update(state)
    else:
        print("ERROR: No model state files found in DeepSpeed checkpoint")
        return False

    # Create output checkpoint in standard format
    if weights_only:
        # Save only model weights
        output_checkpoint = model_state
    else:
        # Save full checkpoint with metadata
        output_checkpoint = {
            'model': model_state,
            'epoch': epoch,
            'args': None,  # Not available from DeepSpeed checkpoint
        }

    # Save converted checkpoint
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(output_checkpoint, output_path)

    print(f"\n✓ Conversion complete!")
    print(f"Saved to: {output_path}")
    print(f"Model parameters: {sum(p.numel() for p in model_state.values()) / 1e6:.2f}M")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeed checkpoints to standard PyTorch format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to DeepSpeed checkpoint directory'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output path for converted checkpoint'
    )
    parser.add_argument(
        '--tag', type=str, default=None,
        help='DeepSpeed checkpoint tag (default: infer from directory name)'
    )
    parser.add_argument(
        '--weights_only', action='store_true',
        help='Save only model weights (for evaluation)'
    )

    args = parser.parse_args()

    success = convert_deepspeed_checkpoint(
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
        tag=args.tag,
        weights_only=args.weights_only
    )

    if success:
        print("\nYou can now load this checkpoint with:")
        print(f"  checkpoint = torch.load('{args.output}', map_location='cpu')")
        if args.weights_only:
            print(f"  model.load_state_dict(checkpoint)")
        else:
            print(f"  model.load_state_dict(checkpoint['model'])")
    else:
        print("\nConversion failed. Please check the error messages above.")
        exit(1)


if __name__ == '__main__':
    main()
