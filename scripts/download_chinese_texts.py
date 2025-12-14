#!/usr/bin/env python3
"""
Chinese Classical Literature Downloader

Downloads Chinese classical texts from the Chinese Text Project (ctext.org)
using their public API for narrative analysis.

Key texts included:
- Confucian classics (Analects, Mencius, Great Learning, Doctrine of the Mean)
- Daoist classics (Dao De Jing, Zhuangzi, Liezi)
- Historical narratives (Zuo Zhuan, Shi Ji selections)
- Literary fiction (Strange Tales from a Chinese Studio, Dream of the Red Chamber excerpts)
- Ming/Qing novels (Romance of Three Kingdoms, Journey to the West, Water Margin, Jin Ping Mei)

Usage:
    python scripts/download_chinese_texts.py --output /Volumes/MacExt/narrative_corpus/chinese
"""

import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# CTP API base URL
CTP_API_BASE = "https://api.ctext.org"


@dataclass
class ChineseText:
    """Represents a Chinese classical text."""
    id: str
    title: str
    title_en: str
    author: str
    dynasty: str
    genre: str
    urn: str  # CTP URN identifier
    text: Optional[str] = None
    char_count: Optional[int] = None
    source: str = "ctext"
    language: str = "zh"

    def to_dict(self) -> dict:
        return asdict(self)


# Curated list of Chinese classical texts with narrative content
# Format: (title_zh, title_en, author, dynasty, genre, urn)
CHINESE_CLASSICS = [
    # === Confucian Classics ===
    ("論語", "Analects", "孔子弟子", "Spring and Autumn", "philosophy",
     "ctp:analects"),
    ("孟子", "Mencius", "孟子", "Warring States", "philosophy",
     "ctp:mengzi"),
    ("大學", "Great Learning", "曾子", "Zhou", "philosophy",
     "ctp:daxue"),
    ("中庸", "Doctrine of the Mean", "子思", "Zhou", "philosophy",
     "ctp:zhongyong"),

    # === Daoist Classics ===
    ("道德經", "Dao De Jing", "老子", "Spring and Autumn", "philosophy",
     "ctp:dao-de-jing"),
    ("莊子", "Zhuangzi", "莊子", "Warring States", "philosophy",
     "ctp:zhuangzi"),
    ("列子", "Liezi", "列子", "Warring States", "philosophy",
     "ctp:liezi"),

    # === Historical Narratives ===
    ("左傳", "Zuo Zhuan", "左丘明", "Spring and Autumn", "history",
     "ctp:chun-qiu-zuo-zhuan"),
    ("戰國策", "Strategies of the Warring States", "劉向編", "Han", "history",
     "ctp:zhan-guo-ce"),
    ("史記", "Records of the Grand Historian", "司馬遷", "Han", "history",
     "ctp:shiji"),

    # === Literary Collections ===
    ("世說新語", "A New Account of Tales of the World", "劉義慶", "Liu Song", "fiction",
     "ctp:shishuo-xinyu"),
    ("搜神記", "In Search of the Supernatural", "干寶", "Jin", "fiction",
     "ctp:soushenji"),

    # === Tang Tales ===
    ("太平廣記", "Extensive Records of the Taiping Era", "李昉等", "Song", "fiction",
     "ctp:taiping-guangji"),

    # === Buddhist Literature ===
    ("六祖壇經", "Platform Sutra of the Sixth Patriarch", "惠能", "Tang", "religion",
     "ctp:liuzu-tanjing"),

    # === Pre-Qin Narratives ===
    ("山海經", "Classic of Mountains and Seas", "佚名", "Warring States", "mythology",
     "ctp:shan-hai-jing"),
    ("穆天子傳", "Biography of the Son of Heaven Mu", "佚名", "Warring States", "mythology",
     "ctp:mu-tianzi-zhuan"),

    # === Han Narratives ===
    ("漢書", "Book of Han", "班固", "Han", "history",
     "ctp:han-shu"),
    ("後漢書", "Book of Later Han", "范曄", "Liu Song", "history",
     "ctp:hou-han-shu"),

    # === Wei-Jin Period ===
    ("三國志", "Records of the Three Kingdoms", "陳壽", "Jin", "history",
     "ctp:sanguo-zhi"),

    # === Poetry Collections (for comparative analysis) ===
    ("詩經", "Classic of Poetry", "Various", "Zhou", "poetry",
     "ctp:book-of-poetry"),
    ("楚辭", "Songs of Chu", "屈原等", "Warring States", "poetry",
     "ctp:chuci"),

    # === Military/Strategy Texts ===
    ("孫子兵法", "Art of War", "孫武", "Spring and Autumn", "military",
     "ctp:art-of-war"),

    # === Legalist Texts ===
    ("韓非子", "Han Feizi", "韓非", "Warring States", "philosophy",
     "ctp:hanfeizi"),

    # === Later Fiction/Narratives ===
    ("聊齋志異", "Strange Tales from a Chinese Studio", "蒲松齡", "Qing", "fiction",
     "ctp:liaozhai-zhiyi"),
]


class CtextDownloader:
    """
    Downloads Chinese classical texts from ctext.org API.

    Respects rate limits and caches responses.
    """

    def __init__(self, cache_dir: Optional[Path] = None, delay: float = 1.0):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory for caching API responses
            delay: Delay between API calls (seconds)
        """
        self.cache_dir = cache_dir or Path("data/cache/ctext")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.catalog: List[ChineseText] = []

    def _api_request(self, endpoint: str, params: Dict[str, str] = None) -> Optional[Dict]:
        """
        Make API request to ctext.org.

        Args:
            endpoint: API endpoint (e.g., 'gettext')
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        url = f"{CTP_API_BASE}/{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'FunctorialNarrativeAnalysis/1.0 (Academic Research)',
                'Accept': 'application/json',
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode('utf-8')
                return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                console.print("[yellow]Rate limited, waiting...[/yellow]")
                time.sleep(10)
                return self._api_request(endpoint, params)
            console.print(f"[red]HTTP Error {e.code}: {e.reason}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]API Error: {e}[/red]")
            return None

    def _get_text_content(self, urn: str) -> Optional[str]:
        """
        Get full text content for a URN.

        The API returns 'fulltext' as a list of paragraphs.
        For works with subsections, we need to recursively fetch.
        """
        cache_file = self.cache_dir / f"{urn.replace(':', '_').replace('/', '_')}.json"

        # Check cache
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                    if 'text' in cached:
                        return cached['text']
            except:
                pass

        # Fetch from API
        response = self._api_request("gettext", {"urn": urn})

        if not response:
            return None

        text_parts = []

        # Get fulltext if available
        if 'fulltext' in response and response['fulltext']:
            text_parts.extend(response['fulltext'])

        # Get subsections if available
        if 'subsections' in response and response['subsections']:
            for sub_urn in response['subsections']:
                time.sleep(self.delay)  # Rate limit
                sub_text = self._get_text_content(sub_urn)
                if sub_text:
                    text_parts.append(sub_text)

        full_text = '\n\n'.join(text_parts) if text_parts else None

        # Cache result
        if full_text:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'urn': urn, 'text': full_text}, f, ensure_ascii=False)

        return full_text

    def _get_chapters(self, urn: str) -> List[str]:
        """
        Get chapter URNs for a work.
        """
        response = self._api_request("gettext", {"urn": urn})

        if response and 'subsections' in response:
            return response['subsections']
        return []

    def load_catalog(self) -> List[ChineseText]:
        """Load the curated catalog of Chinese texts."""
        self.catalog = []

        for i, (title_zh, title_en, author, dynasty, genre, urn) in enumerate(CHINESE_CLASSICS):
            text = ChineseText(
                id=f"ctext_{i+1:04d}",
                title=title_zh,
                title_en=title_en,
                author=author,
                dynasty=dynasty,
                genre=genre,
                urn=urn,
            )
            self.catalog.append(text)

        console.print(f"[green]✓ Loaded {len(self.catalog)} texts in catalog[/green]")
        return self.catalog

    def download_text(self, text: ChineseText, max_chapters: int = 50) -> Optional[ChineseText]:
        """
        Download a single text.

        Args:
            text: ChineseText object with URN
            max_chapters: Maximum chapters to download (for large works)

        Returns:
            ChineseText with content filled in, or None on failure
        """
        console.print(f"[cyan]Downloading: {text.title} ({text.title_en})[/cyan]")

        # First get the structure
        chapters = self._get_chapters(text.urn)

        if chapters:
            # Multi-chapter work
            console.print(f"  Found {len(chapters)} chapters")

            if len(chapters) > max_chapters:
                console.print(f"  [yellow]Limiting to first {max_chapters} chapters[/yellow]")
                chapters = chapters[:max_chapters]

            all_text = []
            for i, chapter_urn in enumerate(chapters):
                time.sleep(self.delay)
                chapter_text = self._get_text_content(chapter_urn)
                if chapter_text:
                    all_text.append(chapter_text)

                if (i + 1) % 10 == 0:
                    console.print(f"  Progress: {i+1}/{len(chapters)} chapters")

            text.text = '\n\n'.join(all_text)
        else:
            # Single text or direct content
            text.text = self._get_text_content(text.urn)

        if text.text:
            text.char_count = len(text.text)
            console.print(f"  [green]✓ Downloaded {text.char_count:,} characters[/green]")
            return text
        else:
            console.print(f"  [red]✗ Failed to download[/red]")
            return None

    def download_all(
        self,
        output_dir: Path,
        min_chars: int = 1000,
        max_chapters: int = 50,
    ) -> List[ChineseText]:
        """
        Download all texts in catalog.

        Args:
            output_dir: Output directory
            min_chars: Minimum character count
            max_chapters: Maximum chapters per work

        Returns:
            List of successfully downloaded texts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        texts_dir = output_dir / "texts"
        texts_dir.mkdir(exist_ok=True)

        successful = []

        console.print(f"\n[bold blue]Downloading {len(self.catalog)} Chinese classical texts...[/bold blue]")
        console.print(f"Output: {output_dir}\n")

        for text in self.catalog:
            result = self.download_text(text, max_chapters)

            if result and result.text and len(result.text) >= min_chars:
                # Save individual text
                text_file = texts_dir / f"{result.id}.json"
                with open(text_file, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

                successful.append(result)

            time.sleep(self.delay)  # Rate limit between texts

        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_texts': len(successful),
            'source': 'ctext.org',
            'language': 'zh',
            'total_characters': sum(t.char_count or 0 for t in successful),
            'texts': [
                {k: v for k, v in t.to_dict().items() if k != 'text'}
                for t in successful
            ]
        }

        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Save manifest for analysis pipeline
        manifest = {
            'corpus': 'chinese_classics',
            'language': 'zh',
            'count': len(successful),
            'texts': [
                {
                    'id': t.id,
                    'title': t.title,
                    'title_en': t.title_en,
                    'author': t.author,
                    'dynasty': t.dynasty,
                    'genre': t.genre,
                    'char_count': t.char_count,
                    'file': f"texts/{t.id}.json"
                }
                for t in successful
            ]
        }

        with open(output_dir / 'manifest.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        console.print(f"\n[bold green]✓ Downloaded {len(successful)}/{len(self.catalog)} texts[/bold green]")
        console.print(f"Total characters: {sum(t.char_count or 0 for t in successful):,}")

        return successful


def create_symlink(target: Path, link: Path) -> bool:
    """Create symlink from link to target."""
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)
        return True
    except Exception as e:
        console.print(f"[yellow]Could not create symlink: {e}[/yellow]")
        return False


@click.command()
@click.option('--output', '-o', default='/Volumes/MacExt/narrative_corpus/chinese',
              type=click.Path(), help='Output directory')
@click.option('--min-chars', '-m', default=1000, help='Minimum characters per text')
@click.option('--max-chapters', '-c', default=50, help='Maximum chapters per work')
@click.option('--delay', '-d', default=1.0, help='Delay between API calls (seconds)')
@click.option('--cache-dir', type=click.Path(), default='data/cache/ctext',
              help='Cache directory for API responses')
def main(output: str, min_chars: int, max_chapters: int, delay: float, cache_dir: str):
    """
    Download Chinese classical literature from ctext.org.

    Downloads a curated selection of Chinese classical texts for
    narrative structure analysis, including:

    - Confucian classics (Analects, Mencius)
    - Daoist texts (Dao De Jing, Zhuangzi)
    - Historical narratives (Zuo Zhuan, Shi Ji)
    - Classical fiction (Strange Tales, World Tales)
    """
    output_path = Path(output)

    # Check if output location is accessible
    if output_path.parts[1] == 'Volumes':
        volume = Path('/') / output_path.parts[1] / output_path.parts[2]
        if not volume.exists():
            console.print(f"[red]Volume not found: {volume}[/red]")
            console.print("Available volumes:")
            for v in Path("/Volumes").iterdir():
                console.print(f"  - {v}")
            return

    console.print("=" * 60)
    console.print("[bold]CHINESE CLASSICAL LITERATURE DOWNLOADER[/bold]")
    console.print(f"Source: ctext.org (Chinese Text Project)")
    console.print(f"Output: {output_path}")
    console.print("=" * 60)

    # Initialize downloader
    downloader = CtextDownloader(
        cache_dir=Path(cache_dir),
        delay=delay
    )

    # Load catalog
    downloader.load_catalog()

    # Download texts
    texts = downloader.download_all(
        output_dir=output_path,
        min_chars=min_chars,
        max_chapters=max_chapters
    )

    # Create symlink in project
    project_link = Path("data/raw/chinese")
    if create_symlink(output_path, project_link):
        console.print(f"\n[green]Symlink created: {project_link} -> {output_path}[/green]")

    # Print summary by genre
    console.print("\n[bold]Summary by Genre:[/bold]")
    genres = {}
    for t in texts:
        genres[t.genre] = genres.get(t.genre, 0) + 1
    for genre, count in sorted(genres.items(), key=lambda x: -x[1]):
        console.print(f"  {genre}: {count}")

    # Print summary by dynasty
    console.print("\n[bold]Summary by Dynasty:[/bold]")
    dynasties = {}
    for t in texts:
        dynasties[t.dynasty] = dynasties.get(t.dynasty, 0) + 1
    for dynasty, count in sorted(dynasties.items(), key=lambda x: -x[1]):
        console.print(f"  {dynasty}: {count}")

    console.print(f"\n[bold green]Done! Corpus ready at: {output_path}[/bold green]")


if __name__ == "__main__":
    main()
