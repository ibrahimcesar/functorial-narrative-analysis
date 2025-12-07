"""
Aozora Bunko Corpus Collector

Downloads and processes Japanese literature from Aozora Bunko (青空文庫),
a digital library of public domain Japanese texts.

This enables cross-cultural comparison with Western narrative structures.
"""

import json
import re
import random
import zipfile
import io
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, asdict

import requests
from tqdm import tqdm
import click
from rich.console import Console

console = Console()


@dataclass
class AozoraText:
    """Represents a single text from Aozora Bunko."""
    id: str
    title: str
    title_reading: str  # Furigana/reading
    author: str
    author_reading: str
    language: str
    text: str
    word_count: int
    char_count: int
    source: str = "aozora"
    year: Optional[int] = None
    category: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class AozoraCollector:
    """
    Collects Japanese texts from Aozora Bunko.

    Uses the Aozora Bunko GitHub mirror and CSV index for metadata,
    then downloads text files directly.
    """

    # Aozora Bunko index CSV (hosted on their website)
    INDEX_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
    # Alternative CSV URL
    INDEX_CSV_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_utf8.zip"
    TEXT_BASE_URL = "https://www.aozora.gr.jp/cards/{author_id}/files/{file_name}"

    # Alternative: Direct ZIP download
    ZIP_URL = "https://www.aozora.gr.jp/cards/{author_id}/files/{zip_name}"

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = None

    def load_index(self) -> List[dict]:
        """
        Load the Aozora Bunko index CSV.

        Returns:
            List of book metadata dictionaries
        """
        console.print("[yellow]Loading Aozora Bunko index...[/yellow]")

        import csv

        # Try downloading ZIP file containing CSV
        for url in [self.INDEX_URL, self.INDEX_CSV_URL]:
            try:
                console.print(f"[dim]Trying {url}[/dim]")
                response = requests.get(url, timeout=120)
                response.raise_for_status()

                # Extract CSV from ZIP
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    for name in zf.namelist():
                        if name.endswith('.csv'):
                            content = zf.read(name)
                            # Try UTF-8 first, then Shift-JIS
                            try:
                                csv_text = content.decode('utf-8')
                            except UnicodeDecodeError:
                                csv_text = content.decode('shift_jis')

                            lines = csv_text.split('\n')
                            reader = csv.DictReader(lines)

                            books = []
                            for row in reader:
                                if row.get('作品ID'):
                                    books.append(row)

                            if books:
                                console.print(f"[green]Loaded {len(books)} entries from index[/green]")
                                self.index = books
                                return books

            except Exception as e:
                console.print(f"[yellow]Failed with {url}: {e}[/yellow]")
                continue

        console.print("[red]Could not load index from any source[/red]")
        return []

    def filter_fiction(self, books: List[dict]) -> List[dict]:
        """
        Filter for fiction/literary works.

        Args:
            books: List of all book entries

        Returns:
            Filtered list of fiction works
        """
        fiction_keywords = [
            '小説', '物語', '短編', '長編',  # Novel, story, short, long
        ]

        # Known fiction authors (major Japanese novelists)
        fiction_authors = [
            '夏目', '芥川', '太宰', '川端', '三島', '谷崎',
            '森鷗外', '泉鏡花', '宮沢賢治', '江戸川乱歩',
            '坂口安吾', '堀辰雄', '横光利一', '梶井基次郎',
        ]

        filtered = []
        seen_works = set()

        for book in books:
            title = book.get('作品名', '')
            author = book.get('著者名', '')
            work_id = book.get('作品ID', '')
            status = book.get('状態', '')

            # Skip if not public or already seen
            if status != '公開' or work_id in seen_works:
                continue

            # Include if title contains fiction keywords
            if any(kw in title for kw in fiction_keywords):
                filtered.append(book)
                seen_works.add(work_id)
            # Or if by a known fiction author
            elif any(auth in author for auth in fiction_authors):
                filtered.append(book)
                seen_works.add(work_id)

        return filtered

    def download_text(self, book: dict) -> Optional[str]:
        """
        Download text content for a book.

        Args:
            book: Book metadata dictionary

        Returns:
            Text content or None if download fails
        """
        # Handle both field naming conventions - strip BOM and leading zeros for URL
        author_id_raw = book.get('人物ID', book.get('﻿人物ID', '')).lstrip('\ufeff')
        work_id_raw = book.get('作品ID', '')

        if not author_id_raw or not work_id_raw:
            return None

        # Aozora uses zero-padded IDs in URLs
        author_id = author_id_raw.zfill(6)
        # Work ID in card URL doesn't have leading zeros, but in file paths it might
        work_id = work_id_raw.lstrip('0') or work_id_raw

        # First, scrape the card page to find the actual download link
        card_url = f"https://www.aozora.gr.jp/cards/{author_id}/card{work_id}.html"

        try:
            response = requests.get(card_url, timeout=30)
            if response.status_code == 200:
                # Look for zip file links (they have version numbers like 773_ruby_5968.zip)
                zip_matches = re.findall(r'href=["\']\.?/?([^"\']*\.zip)["\']', response.text)

                for zip_file in zip_matches:
                    # Prefer ruby (with furigana) or txt versions
                    zip_url = f"https://www.aozora.gr.jp/cards/{author_id}/{zip_file}"
                    if zip_file.startswith('files/'):
                        zip_url = f"https://www.aozora.gr.jp/cards/{author_id}/{zip_file}"
                    else:
                        zip_url = f"https://www.aozora.gr.jp/cards/{author_id}/files/{zip_file}"

                    try:
                        zip_resp = requests.get(zip_url, timeout=30)
                        if zip_resp.status_code == 200:
                            with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
                                for name in zf.namelist():
                                    if name.endswith('.txt'):
                                        content = zf.read(name)
                                        for encoding in ['shift_jis', 'utf-8', 'cp932', 'euc-jp']:
                                            try:
                                                return content.decode(encoding)
                                            except UnicodeDecodeError:
                                                continue
                                        return content.decode('utf-8', errors='ignore')
                    except Exception:
                        continue

        except Exception:
            pass

        return None

    def clean_text(self, text: str) -> str:
        """
        Clean Aozora Bunko text format.

        Removes:
        - Ruby annotations (furigana): 《》
        - Notation markers: ［＃...］
        - Headers and footers

        Args:
            text: Raw Aozora text

        Returns:
            Cleaned text
        """
        # Remove ruby (furigana) annotations: 漢字《かんじ》 -> 漢字
        text = re.sub(r'《[^》]+》', '', text)

        # Remove notation marks: ［＃...］
        text = re.sub(r'［＃[^］]+］', '', text)

        # Remove gaiji (external character) notations
        text = re.sub(r'※［＃[^］]+］', '', text)

        # Find start of main text (after header)
        start_markers = [
            '-------------------------------------------------------',
            '底本：',
        ]

        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)

        # Find content start (skip header)
        for i, line in enumerate(lines):
            if any(marker in line for marker in start_markers):
                # Skip a few lines after marker
                start_idx = i + 3
                break

        # Find content end (before footer)
        for i in range(len(lines) - 1, 0, -1):
            if '底本：' in lines[i] or '青空文庫' in lines[i]:
                end_idx = i
                break

        # Extract main content
        content_lines = lines[start_idx:end_idx]

        # Join and clean whitespace
        text = '\n'.join(content_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines

        return text.strip()

    def count_japanese_chars(self, text: str) -> int:
        """
        Count Japanese characters (kanji, hiragana, katakana).

        This is more meaningful than word count for Japanese.
        """
        # Match Japanese characters
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        return len(japanese_pattern.findall(text))

    def estimate_word_count(self, text: str) -> int:
        """
        Estimate word count for Japanese text.

        Japanese doesn't use spaces, so we estimate based on character count.
        Average Japanese word is ~2-3 characters.
        """
        char_count = self.count_japanese_chars(text)
        return char_count // 2  # Rough estimate

    def process_book(self, book: dict) -> Optional[AozoraText]:
        """
        Download and process a single book.

        Args:
            book: Book metadata from index

        Returns:
            AozoraText object or None if processing fails
        """
        # Download text
        raw_text = self.download_text(book)
        if not raw_text:
            return None

        # Clean text
        text = self.clean_text(raw_text)

        # Calculate counts
        char_count = self.count_japanese_chars(text)
        word_count = self.estimate_word_count(text)

        # Filter by length (equivalent to 10K-500K words in English)
        # Japanese chars: ~5K-250K chars (2-3 chars per word)
        if char_count < 5000 or char_count > 300000:
            return None

        # Extract year from metadata
        year = None
        year_str = book.get('状態の開始日', '') or book.get('底本名', '')
        year_match = re.search(r'(\d{4})', year_str)
        if year_match:
            year = int(year_match.group(1))

        return AozoraText(
            id=f"aozora_{book.get('作品ID', 'unknown')}",
            title=book.get('作品名', 'Unknown'),
            title_reading=book.get('作品名読み', ''),
            author=book.get('著者名', 'Unknown'),
            author_reading=book.get('著者名読み', ''),
            language="ja",
            text=text,
            word_count=word_count,
            char_count=char_count,
            year=year,
            category=book.get('仮名遣い種別', ''),
        )

    def collect(
        self,
        sample_size: int = 100,
        random_sample: bool = True,
        seed: int = 42
    ) -> List[AozoraText]:
        """
        Collect Japanese texts from Aozora Bunko.

        Args:
            sample_size: Number of texts to collect
            random_sample: Whether to randomly sample
            seed: Random seed for reproducibility

        Returns:
            List of AozoraText objects
        """
        console.print(f"[bold blue]Collecting {sample_size} texts from Aozora Bunko[/bold blue]")

        # Load index if not already loaded
        if self.index is None:
            self.load_index()

        if not self.index:
            console.print("[red]Failed to load index[/red]")
            return []

        # Filter for fiction
        fiction_books = self.filter_fiction(self.index)
        console.print(f"[blue]Found {len(fiction_books)} fiction entries[/blue]")

        if random_sample:
            random.seed(seed)
            random.shuffle(fiction_books)

        # Process books
        collected = []
        pbar = tqdm(fiction_books, desc="Downloading texts")

        for book in pbar:
            if len(collected) >= sample_size:
                break

            pbar.set_postfix({"collected": len(collected)})

            try:
                aozora_text = self.process_book(book)
                if aozora_text:
                    collected.append(aozora_text)

                    # Save individual text
                    text_path = self.output_dir / f"{aozora_text.id}.json"
                    with open(text_path, 'w', encoding='utf-8') as f:
                        json.dump(aozora_text.to_dict(), f, ensure_ascii=False, indent=2)

            except Exception as e:
                console.print(f"[red]Error processing {book.get('作品名', 'unknown')}: {e}[/red]")
                continue

            # Rate limiting
            import time
            time.sleep(0.5)

        # Save manifest
        manifest = {
            "source": "aozora",
            "language": "ja",
            "count": len(collected),
            "sample_size": sample_size,
            "seed": seed,
            "texts": [{
                "id": t.id,
                "title": t.title,
                "author": t.author,
                "char_count": t.char_count,
                "word_count": t.word_count,
            } for t in collected]
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        console.print(f"[bold green]✓ Collected {len(collected)} texts[/bold green]")
        console.print(f"[green]Saved to {self.output_dir}[/green]")

        return collected


def create_windows_japanese(
    text: str,
    window_size: int = 500,  # Characters, not words
    overlap: int = 250
) -> List[str]:
    """
    Create overlapping windows from Japanese text.

    Since Japanese doesn't use spaces, we window by character count.

    Args:
        text: Input text
        window_size: Characters per window
        overlap: Character overlap between windows

    Returns:
        List of text windows
    """
    # Remove whitespace for character counting
    chars = re.sub(r'\s+', '', text)

    step = window_size - overlap
    windows = []

    for i in range(0, len(chars), step):
        window = chars[i:i + window_size]
        # Only include windows that are at least half the target size
        if len(window) >= window_size // 2:
            windows.append(window)

    return windows if windows else [text]


@click.command()
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory')
@click.option('--sample-size', '-n', default=50, help='Number of texts to collect')
@click.option('--seed', default=42, help='Random seed for reproducibility')
def main(output: str, sample_size: int, seed: int):
    """Download Japanese fiction texts from Aozora Bunko."""
    collector = AozoraCollector(Path(output))
    collector.collect(sample_size=sample_size, seed=seed)


if __name__ == "__main__":
    main()
