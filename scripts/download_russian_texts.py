#!/usr/bin/env python3
"""
Download Russian Classical Texts

Downloads Russian literary texts from lib.ru and other sources.
Includes original Russian texts for cross-linguistic comparison.

Usage:
    python scripts/download_russian_texts.py --output data/raw/russian
"""

import json
import re
import time
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urljoin
import urllib.request
import urllib.error

import click
from rich.console import Console
from rich.progress import Progress

console = Console()

# Russian texts to download - using lib.ru and other sources
RUSSIAN_TEXTS = [
    {
        "id": "anna_karenina_ru",
        "title": "Анна Каренина",
        "title_en": "Anna Karenina",
        "author": "Лев Толстой",
        "author_en": "Leo Tolstoy",
        "year": 1877,
        "url": "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0080.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "war_and_peace_ru",
        "title": "Война и мир",
        "title_en": "War and Peace",
        "author": "Лев Толстой",
        "author_en": "Leo Tolstoy",
        "year": 1869,
        "url": "http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0040.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "crime_and_punishment_ru",
        "title": "Преступление и наказание",
        "title_en": "Crime and Punishment",
        "author": "Фёдор Достоевский",
        "author_en": "Fyodor Dostoevsky",
        "year": 1866,
        "url": "http://az.lib.ru/d/dostoewskij_f_m/text_0060.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "brothers_karamazov_ru",
        "title": "Братья Карамазовы",
        "title_en": "The Brothers Karamazov",
        "author": "Фёдор Достоевский",
        "author_en": "Fyodor Dostoevsky",
        "year": 1880,
        "url": "http://az.lib.ru/d/dostoewskij_f_m/text_0100.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "idiot_ru",
        "title": "Идиот",
        "title_en": "The Idiot",
        "author": "Фёдор Достоевский",
        "author_en": "Fyodor Dostoevsky",
        "year": 1869,
        "url": "http://az.lib.ru/d/dostoewskij_f_m/text_0070.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "dead_souls_ru",
        "title": "Мёртвые души",
        "title_en": "Dead Souls",
        "author": "Николай Гоголь",
        "author_en": "Nikolai Gogol",
        "year": 1842,
        "url": "http://az.lib.ru/g/gogolx_n_w/text_0140.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "eugene_onegin_ru",
        "title": "Евгений Онегин",
        "title_en": "Eugene Onegin",
        "author": "Александр Пушкин",
        "author_en": "Alexander Pushkin",
        "year": 1833,
        "url": "http://az.lib.ru/p/pushkin_a_s/text_0170.shtml",
        "encoding": "cp1251",
        "genre": "verse_novel"
    },
    {
        "id": "fathers_and_sons_ru",
        "title": "Отцы и дети",
        "title_en": "Fathers and Sons",
        "author": "Иван Тургенев",
        "author_en": "Ivan Turgenev",
        "year": 1862,
        "url": "http://az.lib.ru/t/turgenew_i_s/text_0040.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "oblomov_ru",
        "title": "Обломов",
        "title_en": "Oblomov",
        "author": "Иван Гончаров",
        "author_en": "Ivan Goncharov",
        "year": 1859,
        "url": "http://az.lib.ru/g/goncharow_i_a/text_0020.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
    {
        "id": "master_and_margarita_ru",
        "title": "Мастер и Маргарита",
        "title_en": "The Master and Margarita",
        "author": "Михаил Булгаков",
        "author_en": "Mikhail Bulgakov",
        "year": 1967,
        "url": "http://az.lib.ru/b/bulgakow_m_a/text_0060.shtml",
        "encoding": "cp1251",
        "genre": "novel"
    },
]


def clean_html(html: str) -> str:
    """Extract text content from HTML."""
    # Remove HTML tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '\n', text)

    # Decode HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&mdash;', '—')
    text = text.replace('&ndash;', '–')
    text = text.replace('&laquo;', '«')
    text = text.replace('&raquo;', '»')

    # Normalize whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def download_text(url: str, encoding: str = 'utf-8') -> Optional[str]:
    """Download and decode text from URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) FunctorialNarrativeAnalysis/1.0'
        }
        request = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()

        # Try specified encoding first
        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            # Try common Russian encodings
            for enc in ['utf-8', 'koi8-r', 'cp1251', 'iso-8859-5']:
                try:
                    text = content.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = content.decode('utf-8', errors='ignore')

        return clean_html(text)

    except Exception as e:
        console.print(f"[red]Error downloading {url}: {e}[/red]")
        return None


def download_russian_corpus(output_dir: Path) -> int:
    """
    Download Russian texts corpus.

    Args:
        output_dir: Output directory

    Returns:
        Number of texts downloaded
    """
    output_dir = Path(output_dir)
    texts_dir = output_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0

    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading Russian texts...", total=len(RUSSIAN_TEXTS))

        for text_info in RUSSIAN_TEXTS:
            text_id = text_info["id"]
            out_file = texts_dir / f"{text_id}.json"

            # Skip if already downloaded
            if out_file.exists():
                console.print(f"[dim]Skipping {text_info['title_en']} (already exists)[/dim]")
                progress.update(task, advance=1)
                downloaded += 1
                continue

            console.print(f"[blue]Downloading: {text_info['title']} ({text_info['title_en']})[/blue]")

            text = download_text(text_info["url"], text_info.get("encoding", "utf-8"))

            if text and len(text) > 1000:
                # Save with metadata
                data = {
                    "id": text_id,
                    "title": text_info["title"],
                    "title_en": text_info["title_en"],
                    "author": text_info["author"],
                    "author_en": text_info["author_en"],
                    "year": text_info["year"],
                    "genre": text_info["genre"],
                    "language": "ru",
                    "source_url": text_info["url"],
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "text": text
                }

                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                console.print(f"[green]✓ Saved: {text_info['title_en']} ({len(text):,} chars)[/green]")
                downloaded += 1
            else:
                console.print(f"[yellow]⚠ Failed or too short: {text_info['title_en']}[/yellow]")

            progress.update(task, advance=1)
            time.sleep(2)  # Be polite to server

    # Create manifest
    manifest = {
        "corpus": "russian_classical",
        "total_texts": downloaded,
        "texts": [t for t in RUSSIAN_TEXTS if (texts_dir / f"{t['id']}.json").exists()]
    }

    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return downloaded


@click.command()
@click.option('--output', '-o', default='data/raw/russian',
              type=click.Path(), help='Output directory')
def main(output: str):
    """
    Download Russian classical literature corpus.

    Downloads major Russian literary works from lib.ru for
    cross-linguistic narrative analysis.
    """
    console.print("=" * 60)
    console.print("[bold]RUSSIAN CLASSICAL LITERATURE DOWNLOADER[/bold]")
    console.print(f"Output: {output}")
    console.print("=" * 60)

    count = download_russian_corpus(Path(output))

    console.print(f"\n[bold green]✓ Downloaded {count} Russian texts[/bold green]")
    console.print(f"Texts saved to: {output}/texts/")


if __name__ == "__main__":
    main()
