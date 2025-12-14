#!/usr/bin/env python3
"""
Reclassify existing corpus with improved genre classifier
and optionally add more books from additional sources.

Usage:
    # Reclassify existing books
    python scripts/reclassify_and_expand.py --reclassify

    # Download more from HuggingFace (get ALL available)
    python scripts/reclassify_and_expand.py --download-more --limit 5000

    # Import local books (e.g., from torrents)
    python scripts/reclassify_and_expand.py --import-local /path/to/books

    # Do all
    python scripts/reclassify_and_expand.py --reclassify --download-more --limit 3000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus.large_corpus import (
    LargeCorpusDownloader,
    classify_genre,
    GENRE_KEYWORDS
)


def reclassify_corpus(corpus_dir: str):
    """Reclassify all books with improved genre classifier."""
    corpus_path = Path(corpus_dir)
    books_dir = corpus_path / "books"
    index_file = corpus_path / "index.json"

    if not books_dir.exists():
        print(f"ERROR: Books directory not found: {books_dir}")
        return

    print("=" * 60)
    print("RECLASSIFYING CORPUS WITH IMPROVED GENRE CLASSIFIER")
    print("=" * 60)

    # Load index
    if index_file.exists():
        index = json.loads(index_file.read_text(encoding='utf-8'))
    else:
        index = {"books": []}

    # Reclassify each book
    reclassified = 0
    genre_counts = {}

    for book_file in books_dir.glob("*.json"):
        try:
            data = json.loads(book_file.read_text(encoding='utf-8'))
            text = data.get("content", "")
            title = data.get("title", "")

            # Reclassify
            old_genre = data.get("genre", "unknown")
            new_genre = classify_genre(text, title)

            if new_genre != old_genre:
                data["genre"] = new_genre
                book_file.write_text(
                    json.dumps(data, ensure_ascii=False),
                    encoding='utf-8'
                )
                reclassified += 1

            genre_counts[new_genre] = genre_counts.get(new_genre, 0) + 1

        except Exception as e:
            print(f"  Error with {book_file.name}: {e}")

    # Update index
    for book_info in index.get("books", []):
        book_id = book_info.get("id")
        book_file = books_dir / f"{book_id}.json"
        if book_file.exists():
            try:
                data = json.loads(book_file.read_text(encoding='utf-8'))
                book_info["genre"] = data.get("genre", "general")
            except:
                pass

    # Save updated index
    index_file.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )

    print(f"\nReclassified {reclassified} books")
    print("\nNew genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / sum(genre_counts.values())
        print(f"  {genre}: {count} ({pct:.1f}%)")


def download_more(corpus_dir: str, limit: int, min_words: int = 10000):
    """Download more books from HuggingFace."""
    print("=" * 60)
    print(f"DOWNLOADING MORE BOOKS (limit={limit})")
    print("=" * 60)

    downloader = LargeCorpusDownloader(cache_dir=corpus_dir)

    # This will skip already downloaded books
    count = downloader.download_huggingface(
        limit=limit,
        min_words=min_words
    )

    print(f"\nDownloaded {count} new books")

    # Print updated stats
    stats = downloader.get_statistics()
    print(f"\nTotal corpus: {stats['total_books']} books")


def import_local_books(corpus_dir: str, local_dir: str):
    """Import books from local directory."""
    print("=" * 60)
    print(f"IMPORTING LOCAL BOOKS FROM: {local_dir}")
    print("=" * 60)

    downloader = LargeCorpusDownloader(cache_dir=corpus_dir)

    count = downloader.import_local_files(
        local_dir,
        source_name="local",
        extensions=[".txt", ".epub", ".html"]
    )

    print(f"\nImported {count} books")


def show_available_sources():
    """Show additional sources for fiction books."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              ADDITIONAL FICTION SOURCES                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║  FREE & LEGAL:                                                ║
║  ─────────────                                                ║
║  • Project Gutenberg: https://www.gutenberg.org/              ║
║    - 70,000+ free ebooks                                      ║
║    - Classic literature, public domain                        ║
║                                                               ║
║  • Standard Ebooks: https://standardebooks.org/               ║
║    - High-quality, well-formatted public domain books         ║
║                                                               ║
║  • Feedbooks: https://www.feedbooks.com/publicdomain          ║
║    - Public domain fiction                                    ║
║                                                               ║
║  • ManyBooks: https://manybooks.net/                          ║
║    - Free ebooks in multiple formats                          ║
║                                                               ║
║  • Open Library: https://openlibrary.org/                     ║
║    - Millions of books, some borrowable                       ║
║                                                               ║
║  HUGGINGFACE DATASETS:                                        ║
║  ─────────────────────                                        ║
║  • AlekseyKorshuk/fiction-books (4,737 books) - INCLUDED      ║
║  • bookcorpus (11,038 books) - Needs HF access                ║
║  • pg19 (Project Gutenberg 19) - Classic literature           ║
║                                                               ║
║  FOR RESEARCH (check copyright in your jurisdiction):         ║
║  ────────────────────────────────────────────────────         ║
║  • Archive.org has many scanned books                         ║
║  • LibGen mirrors (use responsibly)                           ║
║                                                               ║
║  TO IMPORT LOCAL FILES:                                       ║
║  ──────────────────────                                       ║
║  python scripts/reclassify_and_expand.py \\                   ║
║      --import-local /path/to/your/books                       ║
║                                                               ║
║  Supported formats: .txt, .epub, .html                        ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Reclassify and expand fiction corpus"
    )
    parser.add_argument(
        "--corpus-dir",
        default="/Volumes/MacExt/narrative_corpus",
        help="Corpus directory"
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="Reclassify existing books with improved classifier"
    )
    parser.add_argument(
        "--download-more",
        action="store_true",
        help="Download more books from HuggingFace"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Maximum books to download from HuggingFace"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10000,
        help="Minimum word count per book"
    )
    parser.add_argument(
        "--import-local",
        type=str,
        help="Import books from local directory"
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Show available sources for more books"
    )

    args = parser.parse_args()

    if args.show_sources:
        show_available_sources()
        return

    if args.reclassify:
        reclassify_corpus(args.corpus_dir)

    if args.download_more:
        download_more(args.corpus_dir, args.limit, args.min_words)

    if args.import_local:
        import_local_books(args.corpus_dir, args.import_local)

    if not any([args.reclassify, args.download_more, args.import_local]):
        print("No action specified. Use --help for options.")
        show_available_sources()


if __name__ == "__main__":
    main()
