"""Offline smoke-test via the exact Telegram image processing function."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot import process_image
from image_config import ImageConfig


def main() -> None:
    parser = argparse.ArgumentParser(description='AutoCrop offline: no Telegram token required')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rotation-only', action='store_true')
    parser.add_argument('--no-shadows', action='store_true')
    parser.add_argument('--no-contrast', action='store_true')
    parser.add_argument('--no-deskew', action='store_true')
    parser.add_argument('--enable-perspective', action='store_true', help='Opt-in projective warp')
    parser.add_argument('--legacy', action='store_true')
    args = parser.parse_args()
    opts = ImageConfig(rotation_only=not args.enable_perspective or args.rotation_only,
                       perspective_enabled=args.enable_perspective and not args.rotation_only,
                       deskew_enabled=not args.no_deskew,
                       shadow_enabled=not args.no_shadows,
                       contrast_enabled=not args.no_contrast, enabled=not args.legacy)
    data, name, caption = process_image(args.input.read_bytes(), args.input.name, (10,) * 4, opts)
    if args.output.suffix.lower() != Path(name).suffix.lower():
        parser.error(f'output suffix must be {Path(name).suffix} to match actual image format')
    args.output.write_bytes(data)
    print(caption)


if __name__ == '__main__':
    main()
