#!/usr/bin/env python3
"""
Download large fiction corpus to external disk.

Usage:
    python scripts/download_corpus_external.py --disk /Volumes/MacExt --limit 2000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.large_corpus import LargeCorpusDownloader


def main():
    parser = argparse.ArgumentParser(
        description="Download large fiction corpus to external disk"
    )
    parser.add_argument(
        "--disk",
        type=str,
        default="/Volumes/MacExt",
        help="External disk mount point"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum books to download from HuggingFace"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10000,
        help="Minimum word count per book"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        help="Local directory with additional files to import (e.g., torrents)"
    )

    args = parser.parse_args()

    # Validate disk
    disk_path = Path(args.disk)
    if not disk_path.exists():
        print(f"ERROR: Disk not found: {args.disk}")
        print("Available volumes:")
        for v in Path("/Volumes").iterdir():
            print(f"  - {v}")
        sys.exit(1)

    # Create corpus directory
    corpus_dir = disk_path / "narrative_corpus"
    corpus_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING LARGE FICTION CORPUS")
    print(f"Target: {corpus_dir}")
    print(f"Limit: {args.limit} books")
    print(f"Min words: {args.min_words}")
    print("=" * 60)

    # Initialize downloader with external disk cache
    downloader = LargeCorpusDownloader(cache_dir=str(corpus_dir))

    # Download from HuggingFace
    hf_count = downloader.download_huggingface(
        limit=args.limit,
        min_words=args.min_words
    )

    # Import local files if specified
    if args.local_dir:
        local_path = Path(args.local_dir)
        if local_path.exists():
            print(f"\nImporting from: {local_path}")
            local_count = downloader.import_local_files(
                str(local_path),
                source_name="local"
            )
        else:
            print(f"WARNING: Local directory not found: {args.local_dir}")

    # Print final statistics
    stats = downloader.get_statistics()

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print(f"\nTotal books: {stats['total_books']:,}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Average length: {stats['avg_word_count']:,} words")

    print(f"\nCorpus location: {corpus_dir}")
    print(f"Index file: {corpus_dir / 'index.json'}")
    print(f"Books directory: {corpus_dir / 'books'}")

    print("\nBy Genre:")
    for genre, count in sorted(stats.get('by_genre', {}).items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_books'] if stats['total_books'] else 0
        print(f"  {genre}: {count:,} ({pct:.1f}%)")

    # Create symlink in project for easy access
    project_link = Path("data/raw/external_corpus")
    if not project_link.exists():
        try:
            project_link.symlink_to(corpus_dir)
            print(f"\nSymlink created: {project_link} -> {corpus_dir}")
        except Exception as e:
            print(f"\nNote: Could not create symlink: {e}")
            print(f"Access corpus at: {corpus_dir}")


if __name__ == "__main__":
    main()
