"""
Project Gutenberg Corpus Collector

Downloads and processes fiction texts from Project Gutenberg for narrative analysis.
This is the primary corpus for replicating Reagan et al. (2016).
"""

import json
import re
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import requests
from tqdm import tqdm
import click
from rich.console import Console

console = Console()


@dataclass
class GutenbergText:
    """Represents a single text from Project Gutenberg."""
    id: str
    title: str
    author: str
    language: str
    text: str
    word_count: int
    source: str = "gutenberg"
    year: Optional[int] = None
    subjects: Optional[list] = None

    def to_dict(self) -> dict:
        return asdict(self)


class GutenbergCollector:
    """
    Collects fiction texts from Project Gutenberg.

    Uses the Gutendex API (https://gutendex.com/) to search and filter texts,
    then downloads raw text from Gutenberg mirrors.
    """

    GUTENDEX_API = "https://gutendex.com/books"
    GUTENBERG_MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

    # Fiction subject keywords
    FICTION_SUBJECTS = [
        "Fiction",
        "English fiction",
        "American fiction",
        "Short stories",
        "Adventure stories",
        "Love stories",
        "Science fiction",
        "Fantasy fiction",
        "Detective and mystery stories",
        "Gothic fiction",
        "Domestic fiction",
        "Historical fiction",
    ]

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def search_fiction(self, max_results: int = 1000) -> list[dict]:
        """
        Search for fiction texts using Gutendex API.

        Args:
            max_results: Maximum number of results to return

        Returns:
            List of book metadata dictionaries
        """
        console.print("[yellow]Searching Gutendex for fiction texts...[/yellow]")

        all_books = []
        seen_ids = set()

        for subject in tqdm(self.FICTION_SUBJECTS, desc="Searching subjects"):
            page_url = f"{self.GUTENDEX_API}?topic={subject}&languages=en"

            while page_url and len(all_books) < max_results * 2:  # Get extra for filtering
                try:
                    response = requests.get(page_url, timeout=30)
                    response.raise_for_status()
                    data = response.json()

                    for book in data.get("results", []):
                        book_id = book.get("id")
                        if book_id and book_id not in seen_ids:
                            seen_ids.add(book_id)
                            all_books.append(book)

                    page_url = data.get("next")

                    # Rate limiting
                    if page_url:
                        import time
                        time.sleep(0.5)

                except requests.RequestException as e:
                    console.print(f"[red]Error fetching {page_url}: {e}[/red]")
                    break

        console.print(f"[green]Found {len(all_books)} unique fiction texts[/green]")
        return all_books[:max_results * 2]

    def download_text(self, book_id: int) -> Optional[str]:
        """
        Download raw text for a book from Gutenberg.

        Args:
            book_id: Gutenberg book ID

        Returns:
            Raw text content or None if download fails
        """
        url = self.GUTENBERG_MIRROR.format(id=book_id)

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            # Try alternate format
            alt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            try:
                response = requests.get(alt_url, timeout=60)
                response.raise_for_status()
                return response.text
            except requests.RequestException:
                return None

    def strip_headers(self, text: str) -> str:
        """
        Remove Project Gutenberg headers and footers.

        Args:
            text: Raw text with headers/footers

        Returns:
            Cleaned text content
        """
        # Find start marker
        start_patterns = [
            r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK .+? \*\*\*",
            r"\*\*\*START OF (THE|THIS) PROJECT GUTENBERG EBOOK .+?\*\*\*",
            r"START OF (THE|THIS) PROJECT GUTENBERG EBOOK",
        ]

        start_pos = 0
        for pattern in start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.end()
                break

        # Find end marker
        end_patterns = [
            r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK .+? \*\*\*",
            r"\*\*\*END OF (THE|THIS) PROJECT GUTENBERG EBOOK .+?\*\*\*",
            r"END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
            r"End of Project Gutenberg",
        ]

        end_pos = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                end_pos = match.start()
                break

        cleaned = text[start_pos:end_pos].strip()

        # Remove any remaining boilerplate at the start
        lines = cleaned.split('\n')
        content_start = 0
        for i, line in enumerate(lines[:100]):  # Check first 100 lines
            if len(line.strip()) > 50 and not any(
                kw in line.lower() for kw in
                ['produced by', 'distributed proofreaders', 'transcriber', 'scanner', 'ebook']
            ):
                content_start = i
                break

        return '\n'.join(lines[content_start:]).strip()

    def process_book(self, book_meta: dict) -> Optional[GutenbergText]:
        """
        Download and process a single book.

        Args:
            book_meta: Book metadata from Gutendex

        Returns:
            GutenbergText object or None if processing fails
        """
        book_id = book_meta.get("id")

        # Download raw text
        raw_text = self.download_text(book_id)
        if not raw_text:
            return None

        # Strip headers
        text = self.strip_headers(raw_text)

        # Calculate word count
        word_count = len(text.split())

        # Filter by length (10K-500K words for novels)
        if word_count < 10000 or word_count > 500000:
            return None

        # Extract metadata
        authors = book_meta.get("authors", [])
        author_name = authors[0].get("name", "Unknown") if authors else "Unknown"

        # Extract year from author birth/death or title
        year = None
        if authors and authors[0].get("birth_year"):
            # Estimate publication as author age 30-50
            birth = authors[0].get("birth_year")
            death = authors[0].get("death_year")
            if death:
                year = (birth + death) // 2
            else:
                year = birth + 40

        return GutenbergText(
            id=f"pg{book_id}",
            title=book_meta.get("title", "Unknown"),
            author=author_name,
            language="en",
            text=text,
            word_count=word_count,
            year=year,
            subjects=book_meta.get("subjects", []),
        )

    def collect(
        self,
        sample_size: int = 1000,
        random_sample: bool = True,
        seed: int = 42
    ) -> list[GutenbergText]:
        """
        Collect fiction texts from Project Gutenberg.

        Args:
            sample_size: Number of texts to collect
            random_sample: Whether to randomly sample from available texts
            seed: Random seed for reproducibility

        Returns:
            List of GutenbergText objects
        """
        console.print(f"[bold blue]Collecting {sample_size} texts from Project Gutenberg[/bold blue]")

        # Search for fiction
        books = self.search_fiction(max_results=sample_size * 3)

        if random_sample:
            random.seed(seed)
            random.shuffle(books)

        # Process books
        collected = []
        pbar = tqdm(books, desc="Downloading texts")

        for book_meta in pbar:
            if len(collected) >= sample_size:
                break

            pbar.set_postfix({"collected": len(collected)})

            try:
                text = self.process_book(book_meta)
                if text:
                    collected.append(text)

                    # Save individual text
                    text_path = self.output_dir / f"{text.id}.json"
                    with open(text_path, 'w', encoding='utf-8') as f:
                        json.dump(text.to_dict(), f, ensure_ascii=False, indent=2)

            except Exception as e:
                console.print(f"[red]Error processing {book_meta.get('id')}: {e}[/red]")
                continue

            # Rate limiting
            import time
            time.sleep(0.3)

        # Save manifest
        manifest = {
            "source": "gutenberg",
            "count": len(collected),
            "sample_size": sample_size,
            "seed": seed,
            "texts": [{"id": t.id, "title": t.title, "author": t.author, "word_count": t.word_count}
                     for t in collected]
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        console.print(f"[bold green]✓ Collected {len(collected)} texts[/bold green]")
        console.print(f"[green]Saved to {self.output_dir}[/green]")

        return collected


@click.command()
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory')
@click.option('--sample-size', '-n', default=100, help='Number of texts to collect')
@click.option('--seed', default=42, help='Random seed for reproducibility')
def main(output: str, sample_size: int, seed: int):
    """Download fiction texts from Project Gutenberg."""
    collector = GutenbergCollector(Path(output))
    collector.collect(sample_size=sample_size, seed=seed)


if __name__ == "__main__":
    main()
