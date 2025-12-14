"""
HuggingFace Datasets Corpus Loader

Provides access to large fiction corpora from HuggingFace Hub.

Available datasets:
- AlekseyKorshuk/fiction-books: 4,737 fiction books with full text
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass


@dataclass
class HFBook:
    """A book from HuggingFace dataset."""
    id: str
    url: str
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    word_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title or self.id,
            "author": self.author or "Unknown",
            "content": self.text,
            "word_count": self.word_count
        }


class HuggingFaceCorpus:
    """
    Load fiction corpus from HuggingFace Hub.

    Dataset: AlekseyKorshuk/fiction-books
    Size: 4,737 books, 254 MB
    """

    DATASET_NAME = "AlekseyKorshuk/fiction-books"
    CACHE_DIR = Path("data/raw/huggingface_cache")

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize corpus loader."""
        self.cache_dir = Path(cache_dir) if cache_dir else self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset = None

    def _load_dataset(self):
        """Load dataset using HuggingFace datasets library."""
        if self._dataset is not None:
            return

        try:
            from datasets import load_dataset
            print(f"Loading dataset: {self.DATASET_NAME}")
            print("  (This may take a few minutes on first download...)")
            self._dataset = load_dataset(self.DATASET_NAME, split="train")
            print(f"  Loaded {len(self._dataset)} books")
        except ImportError:
            raise ImportError(
                "HuggingFace datasets not installed. "
                "Run: pip install datasets"
            )

    def _extract_title_from_url(self, url: str) -> str:
        """Extract book title from URL."""
        # URL format: https://www.bookrix.com/-title-here
        if "bookrix.com" in url:
            parts = url.rstrip("/").split("/")
            if parts:
                title = parts[-1].lstrip("-").replace("-", " ").title()
                return title
        return "Unknown Title"

    def get_books(
        self,
        limit: Optional[int] = None,
        min_words: int = 5000,
        max_words: Optional[int] = None,
        sample_randomly: bool = False
    ) -> List[HFBook]:
        """
        Get books from the dataset.

        Args:
            limit: Maximum number of books to return
            min_words: Minimum word count filter
            max_words: Maximum word count filter (None = no limit)
            sample_randomly: If True, sample randomly; else take first N

        Returns:
            List of HFBook objects
        """
        self._load_dataset()

        books = []
        indices = range(len(self._dataset))

        if sample_randomly and limit:
            import random
            indices = random.sample(list(indices), min(limit * 3, len(indices)))

        for i in indices:
            row = self._dataset[i]
            text = row.get("text", "")
            word_count = len(text.split())

            # Apply filters
            if word_count < min_words:
                continue
            if max_words and word_count > max_words:
                continue

            url = row.get("url", "")
            title = self._extract_title_from_url(url)

            book = HFBook(
                id=f"hf_{i}",
                url=url,
                text=text,
                title=title,
                word_count=word_count
            )
            books.append(book)

            if limit and len(books) >= limit:
                break

        return books

    def stream_books(
        self,
        min_words: int = 5000
    ) -> Iterator[HFBook]:
        """
        Stream books one at a time (memory efficient).

        Args:
            min_words: Minimum word count filter

        Yields:
            HFBook objects
        """
        self._load_dataset()

        for i, row in enumerate(self._dataset):
            text = row.get("text", "")
            word_count = len(text.split())

            if word_count < min_words:
                continue

            url = row.get("url", "")
            title = self._extract_title_from_url(url)

            yield HFBook(
                id=f"hf_{i}",
                url=url,
                text=text,
                title=title,
                word_count=word_count
            )

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        self._load_dataset()

        word_counts = []
        for row in self._dataset:
            text = row.get("text", "")
            word_counts.append(len(text.split()))

        import numpy as np
        word_counts = np.array(word_counts)

        return {
            "total_books": len(self._dataset),
            "total_words": int(word_counts.sum()),
            "mean_words": float(word_counts.mean()),
            "median_words": float(np.median(word_counts)),
            "min_words": int(word_counts.min()),
            "max_words": int(word_counts.max()),
            "books_over_5k": int((word_counts >= 5000).sum()),
            "books_over_10k": int((word_counts >= 10000).sum()),
            "books_over_50k": int((word_counts >= 50000).sum()),
        }


def download_fiction_sample(n_books: int = 100, min_words: int = 10000) -> List[Dict]:
    """
    Download a sample of fiction books for analysis.

    Args:
        n_books: Number of books to download
        min_words: Minimum word count

    Returns:
        List of book dictionaries
    """
    corpus = HuggingFaceCorpus()

    print(f"Downloading {n_books} fiction books (min {min_words} words)...")
    books = corpus.get_books(
        limit=n_books,
        min_words=min_words,
        sample_randomly=True
    )

    print(f"Downloaded {len(books)} books")

    # Save to cache
    cache_file = corpus.cache_dir / "fiction_sample.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in books], f, indent=2, ensure_ascii=False)

    print(f"Saved to: {cache_file}")

    return [b.to_dict() for b in books]


if __name__ == "__main__":
    # Test the loader
    corpus = HuggingFaceCorpus()

    print("=" * 60)
    print("HUGGINGFACE FICTION CORPUS")
    print("=" * 60)

    # Get statistics
    print("\nDataset Statistics:")
    stats = corpus.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

    # Download sample
    print("\n" + "=" * 60)
    print("Downloading sample...")
    books = download_fiction_sample(n_books=50, min_words=10000)

    print(f"\nSample books:")
    for b in books[:5]:
        print(f"  - {b['title'][:50]}: {b['word_count']:,} words")
