"""
Aozora Bunko (青空文庫) Corpus Pipeline

Downloads public domain Japanese literature from Aozora Bunko.
These are classic works (mostly pre-1953) that are excellent for
testing kishōtenketsu hypothesis on traditional Japanese narrative.

Features:
    - Catalog parsing from GitHub mirror
    - Author/work filtering
    - Ruby text handling (furigana)
    - Japanese text encoding (Shift-JIS → UTF-8)
"""

import json
import csv
import random
import urllib.request
import zipfile
import io
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class AozoraBook:
    """Represents a book from Aozora Bunko."""
    book_id: str
    title: str
    author: str
    author_id: str
    first_name: str = ""
    last_name: str = ""
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    category: str = ""
    subcategory: str = ""
    text_url: str = ""
    html_url: str = ""
    text: Optional[str] = None
    word_count: Optional[int] = None
    source: str = "aozora"
    
    def to_dict(self) -> dict:
        return asdict(self)


# Notable Japanese authors for sampling
NOTABLE_AUTHORS = [
    "夏目漱石",  # Natsume Soseki
    "芥川龍之介",  # Akutagawa Ryunosuke
    "太宰治",  # Dazai Osamu
    "宮沢賢治",  # Miyazawa Kenji
    "森鷗外",  # Mori Ogai
    "泉鏡花",  # Izumi Kyoka
    "樋口一葉",  # Higuchi Ichiyo
    "坪内逍遥",  # Tsubouchi Shoyo
    "島崎藤村",  # Shimazaki Toson
    "谷崎潤一郎",  # Tanizaki Junichiro
    "川端康成",  # Kawabata Yasunari
    "志賀直哉",  # Shiga Naoya
    "横光利一",  # Yokomitsu Riichi
]


class AozoraPipeline:
    """
    Pipeline for downloading Japanese literature from Aozora Bunko.
    
    Usage:
        pipeline = AozoraPipeline()
        pipeline.load_catalog()
        books = pipeline.filter_by_category('小説')  # Novels
        sample = pipeline.sample(books, n=50)
        pipeline.download_texts(sample)
        pipeline.save_corpus(sample, output_dir)
    """
    
    # GitHub mirror of Aozora catalog
    CATALOG_URL = "https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/index_pages/list_person_all_extended_utf8.csv"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache/aozora")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog: List[AozoraBook] = []
    
    def load_catalog(self, force_refresh: bool = False) -> List[AozoraBook]:
        """Load Aozora Bunko catalog."""
        catalog_file = self.cache_dir / "aozora_catalog.csv"
        
        if not catalog_file.exists() or force_refresh:
            console.print("[yellow]Downloading Aozora catalog...[/yellow]")
            try:
                urllib.request.urlretrieve(self.CATALOG_URL, catalog_file)
                console.print("[green]✓ Catalog downloaded[/green]")
            except Exception as e:
                console.print(f"[red]Failed: {e}[/red]")
                # Try alternative
                return self._load_fallback_catalog()
        
        console.print("[yellow]Parsing catalog...[/yellow]")
        self.catalog = []
        
        try:
            with open(catalog_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        book = AozoraBook(
                            book_id=row.get('作品ID', ''),
                            title=row.get('作品名', ''),
                            author=row.get('姓') + row.get('名', ''),
                            author_id=row.get('人物ID', ''),
                            first_name=row.get('名', ''),
                            last_name=row.get('姓', ''),
                            category=row.get('分類番号', ''),
                            text_url=row.get('テキストファイルURL', ''),
                            html_url=row.get('XHTML/HTMLファイルURL', ''),
                        )
                        if book.title and book.text_url:
                            self.catalog.append(book)
                    except Exception:
                        continue
        except Exception as e:
            console.print(f"[red]Error parsing: {e}[/red]")
            return self._load_fallback_catalog()
        
        console.print(f"[green]✓ Loaded {len(self.catalog)} books[/green]")
        return self.catalog
    
    def _load_fallback_catalog(self) -> List[AozoraBook]:
        """Load a curated list of notable works."""
        console.print("[yellow]Using curated catalog...[/yellow]")

        # Curated list of well-known works with verified HTML URLs
        # (ZIP URLs often return 404, HTML pages are more stable)
        curated = [
            # Natsume Soseki (夏目漱石)
            ("夏目漱石", "吾輩は猫である", "https://www.aozora.gr.jp/cards/000148/files/789_14547.html"),
            ("夏目漱石", "坊っちゃん", "https://www.aozora.gr.jp/cards/000148/files/752_14964.html"),
            ("夏目漱石", "こころ", "https://www.aozora.gr.jp/cards/000148/files/773_14560.html"),
            ("夏目漱石", "三四郎", "https://www.aozora.gr.jp/cards/000148/files/794_14946.html"),
            # Akutagawa Ryunosuke (芥川龍之介)
            ("芥川龍之介", "羅生門", "https://www.aozora.gr.jp/cards/000879/files/127_15260.html"),
            ("芥川龍之介", "鼻", "https://www.aozora.gr.jp/cards/000879/files/42_15228.html"),
            ("芥川龍之介", "藪の中", "https://www.aozora.gr.jp/cards/000879/files/179_15255.html"),
            ("芥川龍之介", "河童", "https://www.aozora.gr.jp/cards/000879/files/69_14933.html"),
            ("芥川龍之介", "蜘蛛の糸", "https://www.aozora.gr.jp/cards/000879/files/92_490.html"),
            # Dazai Osamu (太宰治)
            ("太宰治", "人間失格", "https://www.aozora.gr.jp/cards/000035/files/301_14912.html"),
            ("太宰治", "走れメロス", "https://www.aozora.gr.jp/cards/000035/files/1567_14913.html"),
            # Miyazawa Kenji (宮沢賢治)
            ("宮沢賢治", "銀河鉄道の夜", "https://www.aozora.gr.jp/cards/000081/files/456_15050.html"),
            ("宮沢賢治", "風の又三郎", "https://www.aozora.gr.jp/cards/000081/files/462_15405.html"),
            ("宮沢賢治", "セロ弾きのゴーシュ", "https://www.aozora.gr.jp/cards/000081/files/470_15407.html"),
            ("宮沢賢治", "注文の多い料理店", "https://www.aozora.gr.jp/cards/000081/files/43754_17659.html"),
            # Nakajima Atsushi (中島敦)
            ("中島敦", "山月記", "https://www.aozora.gr.jp/cards/000119/files/624_14544.html"),
            # Kajii Motojiro (梶井基次郎)
            ("梶井基次郎", "檸檬", "https://www.aozora.gr.jp/cards/000074/files/424_19826.html"),
            # Izumi Kyoka (泉鏡花)
            ("泉鏡花", "高野聖", "https://www.aozora.gr.jp/cards/000050/files/521_19518.html"),
            # Higuchi Ichiyo (樋口一葉)
            ("樋口一葉", "たけくらべ", "https://www.aozora.gr.jp/cards/000064/files/389_15253.html"),
        ]
        
        self.catalog = []
        for i, (author, title, url) in enumerate(curated):
            book = AozoraBook(
                book_id=str(i+1),
                title=title,
                author=author,
                author_id=str(i),
                text_url=url,
            )
            self.catalog.append(book)
        
        console.print(f"[green]✓ Loaded {len(self.catalog)} curated works[/green]")
        return self.catalog
    
    def filter_by_author(self, author_name: str) -> List[AozoraBook]:
        """Filter by author name."""
        return [b for b in self.catalog if author_name in b.author]
    
    def filter_by_authors(self, authors: List[str]) -> List[AozoraBook]:
        """Filter by multiple authors."""
        result = []
        for author in authors:
            result.extend(self.filter_by_author(author))
        return result
    
    def filter_notable_authors(self) -> List[AozoraBook]:
        """Filter to notable Japanese authors."""
        return self.filter_by_authors(NOTABLE_AUTHORS)
    
    def sample(
        self,
        books: List[AozoraBook],
        n: int = 50,
        seed: int = 42,
    ) -> List[AozoraBook]:
        """Sample books."""
        random.seed(seed)
        if len(books) <= n:
            return books
        return random.sample(books, n)
    
    def _clean_aozora_text(self, text: str) -> str:
        """Clean Aozora text format."""
        # Remove ruby annotations ｜word《reading》
        text = re.sub(r'｜([^《]+)《[^》]+》', r'\1', text)
        text = re.sub(r'《[^》]+》', '', text)
        
        # Remove other annotations
        text = re.sub(r'［＃[^］]+］', '', text)
        text = re.sub(r'【[^】]+】', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text.strip()
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract main text content from Aozora HTML page."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Find main text div (Aozora uses class="main_text")
            main_text = soup.find('div', class_='main_text')
            if main_text:
                text = main_text.get_text(separator='\n')
            else:
                # Fallback: get body text
                body = soup.find('body')
                if body:
                    # Remove navigation, headers, etc.
                    for tag in body.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    text = body.get_text(separator='\n')
                else:
                    text = soup.get_text(separator='\n')

            return text
        except ImportError:
            # If BeautifulSoup not available, use regex
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html)
            return text

    def _download_text(self, book: AozoraBook) -> Optional[str]:
        """Download and extract text for a book."""
        if not book.text_url:
            return None

        try:
            req = urllib.request.Request(book.text_url, headers={
                'User-Agent': 'FunctorialNarrativeAnalysis/1.0'
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

                # Handle zip files
                if book.text_url.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(data)) as zf:
                        for name in zf.namelist():
                            if name.endswith('.txt'):
                                # Try Shift-JIS encoding (common for Aozora)
                                try:
                                    text = zf.read(name).decode('shift_jis')
                                except:
                                    try:
                                        text = zf.read(name).decode('utf-8')
                                    except:
                                        text = zf.read(name).decode('cp932', errors='ignore')
                                return self._clean_aozora_text(text)
                elif book.text_url.endswith('.html'):
                    # HTML page - extract text
                    try:
                        html = data.decode('shift_jis')
                    except:
                        try:
                            html = data.decode('utf-8')
                        except:
                            html = data.decode('cp932', errors='ignore')
                    text = self._extract_text_from_html(html)
                    return self._clean_aozora_text(text)
                else:
                    # Plain text
                    try:
                        text = data.decode('shift_jis')
                    except:
                        text = data.decode('utf-8', errors='ignore')
                    return self._clean_aozora_text(text)
        except Exception as e:
            console.print(f"[red]Download error for {book.title}: {e}[/red]")
            return None

        return None
    
    def download_texts(
        self,
        books: List[AozoraBook],
        min_chars: int = 5000,
    ) -> List[AozoraBook]:
        """Download texts for books."""
        successful = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=len(books))
            
            for book in books:
                progress.update(task, description=f"[cyan]{book.title[:20]}...")
                
                text = self._download_text(book)
                
                if text and len(text) >= min_chars:
                    book.text = text
                    book.word_count = len(text)
                    successful.append(book)
                
                progress.advance(task)
        
        console.print(f"[green]✓ Downloaded {len(successful)}/{len(books)} books[/green]")
        return successful
    
    def save_corpus(self, books: List[AozoraBook], output_dir: Path) -> None:
        """Save corpus to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_books': len(books),
            'source': 'aozora',
            'books': [{k: v for k, v in b.to_dict().items() if k != 'text'} for b in books]
        }
        
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        texts_dir = output_dir / 'texts'
        texts_dir.mkdir(exist_ok=True)
        
        for book in books:
            if book.text:
                with open(texts_dir / f'{book.book_id}.json', 'w', encoding='utf-8') as f:
                    json.dump(book.to_dict(), f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Saved corpus to {output_dir}[/green]")


def main():
    """CLI entry point."""
    import click
    
    @click.command()
    @click.option('--n-books', '-n', default=20, help='Number of books')
    @click.option('--output', '-o', default='data/raw/aozora', help='Output directory')
    @click.option('--seed', '-s', default=42, help='Random seed')
    def download_corpus(n_books, output, seed):
        """Download Japanese literature from Aozora Bunko."""
        pipeline = AozoraPipeline()
        pipeline.load_catalog()
        
        # Get all books (or notable authors)
        books = pipeline.catalog if len(pipeline.catalog) > 0 else []
        
        # Sample
        sample = pipeline.sample(books, n=n_books, seed=seed)
        console.print(f"[cyan]Sampled {len(sample)} books[/cyan]")
        
        # Download
        successful = pipeline.download_texts(sample)
        
        # Save
        pipeline.save_corpus(successful, Path(output))
        
        console.print(f"\n[bold green]Done! Downloaded {len(successful)} books[/bold green]")
    
    download_corpus()


if __name__ == "__main__":
    main()
