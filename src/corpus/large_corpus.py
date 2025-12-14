"""
Large Diverse Fiction Corpus Downloader

Downloads and organizes fiction from multiple sources:
1. HuggingFace AlekseyKorshuk/fiction-books (4,737 books)
2. Project Gutenberg (classic literature)
3. Local files (user-provided, e.g., from torrents)

Supports genre classification and filtering.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class Book:
    """A book in the corpus."""
    id: str
    title: str
    author: str
    text: str
    source: str  # 'huggingface', 'gutenberg', 'local'
    genre: Optional[str] = None
    language: str = "english"
    word_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.text.split())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "author": self.author,
            "source": self.source,
            "genre": self.genre,
            "language": self.language,
            "word_count": self.word_count,
            "content": self.text,
            "metadata": self.metadata,
        }


# Genre keywords for classification - weighted by specificity
# Format: (keyword, weight) - higher weight = more genre-specific
GENRE_KEYWORDS = {
    "romance": [
        ("fell in love", 3), ("loved him deeply", 3), ("loved her deeply", 3),
        ("passionate kiss", 3), ("marriage proposal", 3), ("wedding day", 2),
        ("romantic", 1), ("passion", 1), ("desire", 1), ("embrace", 1),
        ("beloved", 1), ("heart raced", 2), ("love story", 3)
    ],
    "mystery": [
        ("detective", 3), ("murder investigation", 4), ("crime scene", 3),
        ("the murderer", 3), ("corpse", 2), ("inspector", 2), ("evidence", 1),
        ("alibi", 2), ("whodunit", 4), ("sleuth", 3), ("solved the case", 3),
        ("suspect", 1), ("clue", 1)
    ],
    "horror": [
        ("terrified", 2), ("nightmare", 1), ("haunted house", 3), ("ghost", 2),
        ("monster", 2), ("screamed in terror", 3), ("blood dripping", 3),
        ("horror", 2), ("demon", 2), ("supernatural", 2), ("evil presence", 3),
        ("creature", 1), ("darkness", 1)
    ],
    "scifi": [
        ("spaceship", 4), ("alien", 3), ("robot", 2), ("android", 3),
        ("laser", 2), ("galaxy", 2), ("starship", 4), ("cyborg", 4),
        ("spacecraft", 4), ("interstellar", 4), ("hyperspace", 4),
        ("space station", 4), ("artificial intelligence", 3)
    ],
    "fantasy": [
        ("wizard", 3), ("dragon", 3), ("elf", 2), ("dwarf", 2), ("spell", 2),
        ("sorcerer", 3), ("prophecy", 2), ("mythical", 2), ("enchanted", 2),
        ("magical powers", 3), ("cast a spell", 3), ("dark lord", 4),
        ("ancient magic", 3)
    ],
    "adventure": [
        ("expedition", 2), ("treasure hunt", 3), ("explorer", 2), ("jungle", 2),
        ("dangerous journey", 3), ("perilous", 2), ("voyage", 2), ("wilderness", 2),
        ("survived", 1), ("escape", 1), ("discovery", 1)
    ],
    "historical": [
        ("century ago", 2), ("in the year 18", 3), ("in the year 17", 3),
        ("ancient rome", 4), ("medieval", 2), ("victorian era", 3),
        ("civil war", 2), ("revolution of", 3), ("dynasty", 2),
        ("18th century", 3), ("19th century", 3), ("historical novel", 4)
    ],
    "literary": [
        ("contemplated", 1), ("existence", 1), ("meaning of life", 3),
        ("philosophical", 2), ("consciousness", 2), ("morality", 1),
        ("human condition", 3), ("introspection", 2), ("melancholy", 2),
        ("solitude", 1), ("profound", 1)
    ],
    "thriller": [
        ("chased", 1), ("escape", 1), ("spy", 2), ("secret agent", 3),
        ("conspiracy", 2), ("assassin", 3), ("dangerous", 1), ("threat", 1),
        ("kidnapped", 2), ("hostage", 2), ("time running out", 3),
        ("bomb", 2), ("terrorist", 3)
    ],
    "comedy": [
        ("laughed", 1), ("funny", 2), ("humorous", 2), ("joke", 1),
        ("hilarious", 3), ("comic", 2), ("amusing", 1), ("ridiculous", 1),
        ("absurd", 1), ("witty", 2), ("satirical", 2), ("comedy", 3)
    ],
}


def classify_genre(text: str, title: str = "") -> str:
    """Classify book genre based on weighted text content analysis."""
    text_lower = (text[:100000] + " " + title).lower()  # Sample first 100k chars

    genre_scores = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        score = 0
        for kw, weight in keywords:
            # Use regex for word boundary matching to avoid partial matches
            # (e.g., "elf" matching "self", "himself")
            pattern = r'\b' + re.escape(kw) + r'\b'
            count = len(re.findall(pattern, text_lower))
            score += count * weight
        genre_scores[genre] = score

    # Require minimum weighted score to classify
    max_score = max(genre_scores.values()) if genre_scores else 0
    if max_score >= 10:  # Higher threshold for weighted scores
        return max(genre_scores, key=genre_scores.get)
    return "general"


class LargeCorpusDownloader:
    """
    Downloads and manages a large diverse fiction corpus.
    """

    def __init__(self, cache_dir: str = "data/raw/large_corpus"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.books_dir = self.cache_dir / "books"
        self.books_dir.mkdir(exist_ok=True)

        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict:
        """Load corpus index."""
        if self.index_file.exists():
            return json.loads(self.index_file.read_text(encoding='utf-8'))
        return {"books": [], "stats": {}}

    def _save_index(self):
        """Save corpus index."""
        self.index_file.write_text(
            json.dumps(self.index, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )

    def _generate_id(self, text: str) -> str:
        """Generate unique ID from text hash."""
        return hashlib.md5(text[:1000].encode()).hexdigest()[:12]

    def download_huggingface(
        self,
        limit: Optional[int] = None,
        min_words: int = 10000,
        max_words: int = 500000
    ) -> int:
        """
        Download fiction from HuggingFace dataset.

        Args:
            limit: Maximum books to download (None = all)
            min_words: Minimum word count
            max_words: Maximum word count

        Returns:
            Number of books downloaded
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERROR: Install datasets: pip install datasets")
            return 0

        print("=" * 60)
        print("DOWNLOADING FROM HUGGINGFACE")
        print("Dataset: AlekseyKorshuk/fiction-books (4,737 books)")
        print("=" * 60)

        print("\nLoading dataset (this may take a few minutes)...")
        dataset = load_dataset("AlekseyKorshuk/fiction-books", split="train")

        downloaded = 0
        skipped = 0

        for i, row in enumerate(dataset):
            if limit and downloaded >= limit:
                break

            text = row.get("text", "")
            word_count = len(text.split())

            # Filter by length
            if word_count < min_words:
                skipped += 1
                continue
            if word_count > max_words:
                skipped += 1
                continue

            # Extract title from URL
            url = row.get("url", "")
            title = self._extract_title_from_url(url)

            # Generate ID
            book_id = f"hf_{self._generate_id(text)}"

            # Skip if already downloaded
            book_file = self.books_dir / f"{book_id}.json"
            if book_file.exists():
                downloaded += 1
                continue

            # Classify genre
            genre = classify_genre(text, title)

            # Create book
            book = Book(
                id=book_id,
                title=title,
                author="Unknown",
                text=text,
                source="huggingface",
                genre=genre,
                word_count=word_count,
                metadata={"url": url, "hf_index": i}
            )

            # Save book
            book_file.write_text(
                json.dumps(book.to_dict(), ensure_ascii=False),
                encoding='utf-8'
            )

            # Update index
            self.index["books"].append({
                "id": book_id,
                "title": title,
                "genre": genre,
                "word_count": word_count,
                "source": "huggingface"
            })

            downloaded += 1

            if downloaded % 100 == 0:
                print(f"  Downloaded: {downloaded} | Skipped: {skipped}")
                self._save_index()

        self._save_index()
        print(f"\nComplete: {downloaded} books downloaded, {skipped} skipped")

        return downloaded

    def _extract_title_from_url(self, url: str) -> str:
        """Extract book title from URL."""
        if "bookrix.com" in url:
            parts = url.rstrip("/").split("/")
            if parts:
                title = parts[-1].lstrip("-").replace("-", " ").title()
                return title[:100]  # Limit length
        return "Unknown Title"

    def import_local_files(
        self,
        directory: str,
        source_name: str = "local",
        extensions: List[str] = [".txt", ".epub", ".html"]
    ) -> int:
        """
        Import local text files (e.g., from torrents).

        Args:
            directory: Directory containing text files
            source_name: Name for this source
            extensions: File extensions to import

        Returns:
            Number of books imported
        """
        import_dir = Path(directory)
        if not import_dir.exists():
            print(f"ERROR: Directory not found: {directory}")
            return 0

        print("=" * 60)
        print(f"IMPORTING LOCAL FILES")
        print(f"Directory: {directory}")
        print("=" * 60)

        imported = 0

        for ext in extensions:
            for file_path in import_dir.rglob(f"*{ext}"):
                try:
                    # Read file
                    if ext == ".txt":
                        text = file_path.read_text(encoding='utf-8', errors='ignore')
                    elif ext == ".html":
                        text = self._extract_text_from_html(file_path)
                    elif ext == ".epub":
                        text = self._extract_text_from_epub(file_path)
                    else:
                        continue

                    if len(text) < 10000:
                        continue

                    # Generate ID
                    book_id = f"{source_name}_{self._generate_id(text)}"

                    # Skip if already imported
                    book_file = self.books_dir / f"{book_id}.json"
                    if book_file.exists():
                        continue

                    # Extract title from filename
                    title = file_path.stem.replace("_", " ").replace("-", " ").title()

                    # Classify genre
                    genre = classify_genre(text, title)
                    word_count = len(text.split())

                    book = Book(
                        id=book_id,
                        title=title,
                        author="Unknown",
                        text=text,
                        source=source_name,
                        genre=genre,
                        word_count=word_count,
                        metadata={"original_file": str(file_path)}
                    )

                    # Save
                    book_file.write_text(
                        json.dumps(book.to_dict(), ensure_ascii=False),
                        encoding='utf-8'
                    )

                    self.index["books"].append({
                        "id": book_id,
                        "title": title,
                        "genre": genre,
                        "word_count": word_count,
                        "source": source_name
                    })

                    imported += 1

                    if imported % 50 == 0:
                        print(f"  Imported: {imported}")

                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")

        self._save_index()
        print(f"\nComplete: {imported} books imported")
        return imported

    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML file."""
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.in_script = False

            def handle_starttag(self, tag, attrs):
                if tag in ['script', 'style']:
                    self.in_script = True

            def handle_endtag(self, tag):
                if tag in ['script', 'style']:
                    self.in_script = False

            def handle_data(self, data):
                if not self.in_script:
                    self.text.append(data)

        html = file_path.read_text(encoding='utf-8', errors='ignore')
        parser = TextExtractor()
        parser.feed(html)
        return ' '.join(parser.text)

    def _extract_text_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB file."""
        try:
            import zipfile
            from html.parser import HTMLParser

            text_parts = []

            with zipfile.ZipFile(file_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith(('.html', '.xhtml', '.htm')):
                        content = zf.read(name).decode('utf-8', errors='ignore')
                        # Simple HTML stripping
                        clean = re.sub(r'<[^>]+>', ' ', content)
                        text_parts.append(clean)

            return ' '.join(text_parts)
        except Exception as e:
            return ""

    def get_statistics(self) -> Dict:
        """Get corpus statistics."""
        books = self.index.get("books", [])

        if not books:
            return {"total": 0}

        # Count by source
        by_source = {}
        for b in books:
            src = b.get("source", "unknown")
            by_source[src] = by_source.get(src, 0) + 1

        # Count by genre
        by_genre = {}
        for b in books:
            genre = b.get("genre", "unknown")
            by_genre[genre] = by_genre.get(genre, 0) + 1

        # Word count stats
        word_counts = [b.get("word_count", 0) for b in books]

        return {
            "total_books": len(books),
            "total_words": sum(word_counts),
            "by_source": by_source,
            "by_genre": by_genre,
            "avg_word_count": sum(word_counts) // len(word_counts) if word_counts else 0,
            "min_word_count": min(word_counts) if word_counts else 0,
            "max_word_count": max(word_counts) if word_counts else 0,
        }

    def load_books(
        self,
        limit: Optional[int] = None,
        genre: Optional[str] = None,
        min_words: int = 0,
        source: Optional[str] = None
    ) -> Iterator[Book]:
        """
        Load books from corpus.

        Args:
            limit: Maximum books to return
            genre: Filter by genre
            min_words: Minimum word count
            source: Filter by source

        Yields:
            Book objects
        """
        count = 0

        for book_info in self.index.get("books", []):
            if limit and count >= limit:
                break

            # Apply filters
            if genre and book_info.get("genre") != genre:
                continue
            if min_words and book_info.get("word_count", 0) < min_words:
                continue
            if source and book_info.get("source") != source:
                continue

            # Load full book
            book_file = self.books_dir / f"{book_info['id']}.json"
            if not book_file.exists():
                continue

            try:
                data = json.loads(book_file.read_text(encoding='utf-8'))
                yield Book(
                    id=data["id"],
                    title=data["title"],
                    author=data.get("author", "Unknown"),
                    text=data.get("content", ""),
                    source=data.get("source", "unknown"),
                    genre=data.get("genre"),
                    word_count=data.get("word_count", 0),
                    metadata=data.get("metadata", {})
                )
                count += 1
            except Exception as e:
                print(f"Error loading {book_file}: {e}")


def download_diverse_corpus(
    limit: int = 1000,
    min_words: int = 15000,
    local_dir: Optional[str] = None
):
    """
    Download a large, diverse fiction corpus.

    Args:
        limit: Max books from HuggingFace
        min_words: Minimum word count
        local_dir: Optional directory with local files to import
    """
    downloader = LargeCorpusDownloader()

    # Download from HuggingFace
    print("\n" + "=" * 60)
    print("BUILDING LARGE DIVERSE CORPUS")
    print("=" * 60)

    hf_count = downloader.download_huggingface(
        limit=limit,
        min_words=min_words
    )

    # Import local files if provided
    local_count = 0
    if local_dir and Path(local_dir).exists():
        local_count = downloader.import_local_files(local_dir)

    # Print statistics
    stats = downloader.get_statistics()

    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)

    print(f"\nTotal books: {stats['total_books']:,}")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Average length: {stats['avg_word_count']:,} words")

    print("\nBy Source:")
    for src, count in stats.get('by_source', {}).items():
        print(f"  {src}: {count:,}")

    print("\nBy Genre:")
    for genre, count in sorted(stats.get('by_genre', {}).items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_books'] if stats['total_books'] else 0
        print(f"  {genre}: {count:,} ({pct:.1f}%)")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download large diverse fiction corpus")
    parser.add_argument("--limit", type=int, default=500, help="Max books from HuggingFace")
    parser.add_argument("--min-words", type=int, default=15000, help="Minimum word count")
    parser.add_argument("--local-dir", type=str, help="Directory with local files to import")

    args = parser.parse_args()

    download_diverse_corpus(
        limit=args.limit,
        min_words=args.min_words,
        local_dir=args.local_dir
    )
