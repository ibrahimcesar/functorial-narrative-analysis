"""
Project Gutenberg Corpus Pipeline

Downloads and processes fiction texts from Project Gutenberg for narrative analysis.

Features:
    - Catalog parsing from Gutenberg RDF/CSV
    - Fiction filtering by Library of Congress classification
    - Header/footer stripping
    - Metadata extraction
    - Random and popularity-weighted sampling
"""

import json
import re
import csv
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import urllib.request
import urllib.error
from io import StringIO

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class GutenbergBook:
    """Represents a book from Project Gutenberg."""
    id: str
    title: str
    author: str
    year: Optional[int]
    language: str
    subjects: List[str] = field(default_factory=list)
    loc_class: Optional[str] = None  # Library of Congress classification
    downloads: int = 0
    text: Optional[str] = None
    word_count: Optional[int] = None
    source: str = "gutenberg"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GutenbergBook":
        return cls(**data)


# Fiction-related Library of Congress classifications
FICTION_LOC_CLASSES = {
    'PS': 'American literature',
    'PR': 'English literature', 
    'PZ': 'Fiction and juvenile belles lettres',
    'PT': 'German literature',
    'PQ': 'French, Italian, Spanish, Portuguese literature',
    'PN': 'General literature (includes drama/fiction)',
}

# Subject keywords indicating fiction
FICTION_SUBJECT_KEYWORDS = [
    'fiction', 'novel', 'stories', 'tale', 'romance',
    'adventure', 'mystery', 'detective', 'science fiction',
    'fantasy', 'horror', 'thriller', 'drama', 'plays',
]


class GutenbergPipeline:
    """
    Pipeline for downloading and processing Project Gutenberg texts.
    
    Usage:
        pipeline = GutenbergPipeline()
        pipeline.load_catalog()
        books = pipeline.filter_fiction(language='en')
        sample = pipeline.sample(books, n=100)
        pipeline.download_texts(sample)
        pipeline.save_corpus(sample, output_dir)
    """
    
    CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
    MIRROR_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
    ALT_MIRROR = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize pipeline.
        
        Args:
            cache_dir: Directory for caching catalog and texts
        """
        self.cache_dir = cache_dir or Path("data/cache/gutenberg")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog: List[GutenbergBook] = []
    
    def load_catalog(self, force_refresh: bool = False) -> List[GutenbergBook]:
        """
        Load the Gutenberg catalog.
        
        Args:
            force_refresh: Re-download catalog even if cached
            
        Returns:
            List of GutenbergBook entries
        """
        catalog_file = self.cache_dir / "pg_catalog.csv"
        
        # Download if needed
        if not catalog_file.exists() or force_refresh:
            console.print("[yellow]Downloading Gutenberg catalog...[/yellow]")
            try:
                urllib.request.urlretrieve(self.CATALOG_URL, catalog_file)
                console.print("[green]✓ Catalog downloaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed to download catalog: {e}[/red]")
                return []
        
        # Parse catalog
        console.print("[yellow]Parsing catalog...[/yellow]")
        self.catalog = []
        
        with open(catalog_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Extract book ID
                    book_id = row.get('Text#', '').strip()
                    if not book_id or not book_id.isdigit():
                        continue
                    
                    # Parse year from title or author dates
                    year = None
                    authors = row.get('Authors', '')
                    year_match = re.search(r'\b(1[4-9]\d{2}|20[0-2]\d)\b', authors)
                    if year_match:
                        year = int(year_match.group(1))
                    
                    # Parse subjects
                    subjects_str = row.get('Subjects', '')
                    subjects = [s.strip() for s in subjects_str.split(';') if s.strip()]
                    
                    # Parse LoC class
                    loc_class = row.get('LoCC', '').strip().split()[0] if row.get('LoCC') else None
                    
                    book = GutenbergBook(
                        id=f"pg{book_id}",
                        title=row.get('Title', 'Unknown').strip(),
                        author=row.get('Authors', 'Unknown').split(',')[0].strip(),
                        year=year,
                        language=row.get('Language', 'en').strip().lower(),
                        subjects=subjects,
                        loc_class=loc_class,
                    )
                    self.catalog.append(book)
                except Exception:
                    continue
        
        console.print(f"[green]✓ Loaded {len(self.catalog)} books from catalog[/green]")
        return self.catalog
    
    def filter_fiction(
        self, 
        language: str = 'en',
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> List[GutenbergBook]:
        """
        Filter catalog to fiction texts.
        
        Args:
            language: Language code (default: 'en')
            min_year: Minimum publication year
            max_year: Maximum publication year
            
        Returns:
            Filtered list of fiction books
        """
        fiction = []
        
        for book in self.catalog:
            # Language filter
            if book.language != language:
                continue
            
            # Year filter
            if min_year and book.year and book.year < min_year:
                continue
            if max_year and book.year and book.year > max_year:
                continue
            
            # Fiction filter: LoC class or subject keywords
            is_fiction = False
            
            # Check LoC classification
            if book.loc_class and book.loc_class[:2] in FICTION_LOC_CLASSES:
                is_fiction = True
            
            # Check subjects
            if not is_fiction:
                subjects_lower = ' '.join(book.subjects).lower()
                for keyword in FICTION_SUBJECT_KEYWORDS:
                    if keyword in subjects_lower:
                        is_fiction = True
                        break
            
            if is_fiction:
                fiction.append(book)
        
        console.print(f"[green]✓ Found {len(fiction)} fiction texts in {language}[/green]")
        return fiction
    
    def sample(
        self,
        books: List[GutenbergBook],
        n: int = 100,
        method: str = 'random',
        seed: int = 42,
    ) -> List[GutenbergBook]:
        """
        Sample books from filtered list.
        
        Args:
            books: List of books to sample from
            n: Number of books to sample
            method: 'random' or 'stratified' (by decade)
            seed: Random seed for reproducibility
            
        Returns:
            Sampled list of books
        """
        random.seed(seed)
        
        if len(books) <= n:
            return books
        
        if method == 'random':
            return random.sample(books, n)
        
        elif method == 'stratified':
            # Stratify by decade
            by_decade = {}
            for book in books:
                if book.year:
                    decade = (book.year // 10) * 10
                else:
                    decade = 0
                if decade not in by_decade:
                    by_decade[decade] = []
                by_decade[decade].append(book)
            
            # Sample proportionally from each decade
            sample = []
            for decade, decade_books in by_decade.items():
                k = max(1, int(n * len(decade_books) / len(books)))
                sample.extend(random.sample(decade_books, min(k, len(decade_books))))
            
            # Trim or pad to exactly n
            if len(sample) > n:
                sample = random.sample(sample, n)
            elif len(sample) < n:
                remaining = [b for b in books if b not in sample]
                sample.extend(random.sample(remaining, min(n - len(sample), len(remaining))))
            
            return sample
        
        return random.sample(books, n)
    
    def _strip_gutenberg_header_footer(self, text: str) -> str:
        """Remove Gutenberg boilerplate from text."""
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find start marker
        for i, line in enumerate(lines):
            if '*** START' in line.upper() or 'START OF' in line.upper():
                start_idx = i + 1
                break
        
        # Find end marker
        for i in range(len(lines) - 1, -1, -1):
            if '*** END' in lines[i].upper() or 'END OF' in lines[i].upper():
                end_idx = i
                break
        
        clean_text = '\n'.join(lines[start_idx:end_idx])
        
        # Fallback if markers not found
        if len(clean_text) < 1000:
            return text
        
        return clean_text
    
    def download_text(self, book: GutenbergBook) -> Optional[str]:
        """
        Download text for a single book.
        
        Args:
            book: Book to download
            
        Returns:
            Cleaned text or None if failed
        """
        book_num = book.id.replace('pg', '')
        
        # Try primary mirror
        urls = [
            self.MIRROR_URL.format(id=book_num),
            self.ALT_MIRROR.format(id=book_num),
        ]
        
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    text = response.read().decode('utf-8', errors='ignore')
                    clean_text = self._strip_gutenberg_header_footer(text)
                    return clean_text
            except Exception:
                continue
        
        return None
    
    def download_texts(
        self,
        books: List[GutenbergBook],
        min_words: int = 10000,
        max_words: int = 500000,
    ) -> List[GutenbergBook]:
        """
        Download texts for a list of books.
        
        Args:
            books: Books to download
            min_words: Minimum word count to include
            max_words: Maximum word count to include
            
        Returns:
            List of books with text successfully downloaded
        """
        successful = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading texts...", total=len(books))
            
            for book in books:
                progress.update(task, description=f"[cyan]{book.title[:40]}...")
                
                text = self.download_text(book)
                
                if text:
                    word_count = len(text.split())
                    
                    if min_words <= word_count <= max_words:
                        book.text = text
                        book.word_count = word_count
                        successful.append(book)
                
                progress.advance(task)
        
        console.print(f"[green]✓ Successfully downloaded {len(successful)}/{len(books)} texts[/green]")
        return successful
    
    def save_corpus(
        self,
        books: List[GutenbergBook],
        output_dir: Path,
        format: str = 'json',
    ) -> None:
        """
        Save corpus to disk.
        
        Args:
            books: Books to save
            output_dir: Output directory
            format: 'json' or 'txt'
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_books': len(books),
            'books': [
                {k: v for k, v in b.to_dict().items() if k != 'text'}
                for b in books
            ]
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save texts
        texts_dir = output_dir / 'texts'
        texts_dir.mkdir(exist_ok=True)
        
        for book in books:
            if book.text:
                if format == 'json':
                    with open(texts_dir / f'{book.id}.json', 'w') as f:
                        json.dump(book.to_dict(), f, indent=2)
                else:
                    with open(texts_dir / f'{book.id}.txt', 'w') as f:
                        f.write(book.text)
        
        console.print(f"[green]✓ Saved corpus to {output_dir}[/green]")


def main():
    """CLI entry point for Gutenberg pipeline."""
    import click
    
    @click.command()
    @click.option('--language', '-l', default='en', help='Language code')
    @click.option('--n-books', '-n', default=100, help='Number of books to download')
    @click.option('--output', '-o', default='data/raw/gutenberg', help='Output directory')
    @click.option('--method', '-m', default='random', help='Sampling method: random or stratified')
    @click.option('--seed', '-s', default=42, help='Random seed')
    def download_corpus(language, n_books, output, method, seed):
        """Download fiction corpus from Project Gutenberg."""
        pipeline = GutenbergPipeline()
        
        # Load and filter
        pipeline.load_catalog()
        fiction = pipeline.filter_fiction(language=language)
        
        # Sample
        sample = pipeline.sample(fiction, n=n_books, method=method, seed=seed)
        console.print(f"[cyan]Sampled {len(sample)} books[/cyan]")
        
        # Download
        successful = pipeline.download_texts(sample)
        
        # Save
        pipeline.save_corpus(successful, Path(output))
        
        console.print(f"\n[bold green]Done! Downloaded {len(successful)} books to {output}[/bold green]")
    
    download_corpus()


if __name__ == "__main__":
    main()
