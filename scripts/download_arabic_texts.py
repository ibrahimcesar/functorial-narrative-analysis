#!/usr/bin/env python3
"""
Arabic/Islamic Classical Literature Downloader

Downloads Arabic classical texts from the OpenITI corpus (sourced from Shamela.ws).
The OpenITI corpus contains 6,858 unique books from 2,854 authors spanning
14 centuries of Arabic literature.

Sources:
- OpenITI GitHub: https://github.com/OpenITI
- Zenodo: https://zenodo.org/records/10007820
- Original source: Al-Maktaba Al-Shamela (shamela.ws)

Usage:
    python scripts/download_arabic_texts.py --output /Volumes/MacExt/narrative_corpus/arabic
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class ArabicText:
    """Represents an Arabic classical text."""
    id: str
    title: str
    title_transliterated: str
    author: str
    author_death_ah: Optional[int]
    author_death_ce: Optional[int]
    genre: str
    text: Optional[str] = None
    word_count: Optional[int] = None
    source: str = "openiti"
    language: str = "ar"

    def to_dict(self) -> dict:
        return asdict(self)


# Curated selection of major Arabic classical texts
# Format: (repo, author_folder, book_folder, title_ar, title_en, author, death_ah, genre)
CURATED_ARABIC_TEXTS = [
    # === Hadith Collections ===
    ("0275AH", "0256Bukhari", "0256Bukhari.Sahih", "صحيح البخاري", "Sahih al-Bukhari", "al-Bukhari", 256, "hadith"),
    ("0275AH", "0261Muslim", "0261Muslim.Sahih", "صحيح مسلم", "Sahih Muslim", "Muslim ibn al-Hajjaj", 261, "hadith"),
    ("0275AH", "0275AbuDawworkoSijistani", "0275AbuDawworkoSijistani.Sunan", "سنن أبي داود", "Sunan Abu Dawud", "Abu Dawud", 275, "hadith"),
    ("0300AH", "0279Tirmidhi", "0279Tirmidhi.Sunan", "سنن الترمذي", "Jami at-Tirmidhi", "al-Tirmidhi", 279, "hadith"),
    ("0275AH", "0273IbnMaja", "0273IbnMaja.Sunan", "سنن ابن ماجه", "Sunan Ibn Majah", "Ibn Majah", 273, "hadith"),
    ("0325AH", "0303Nasai", "0303Nasai.Sunan", "سنن النسائي", "Sunan an-Nasa'i", "al-Nasa'i", 303, "hadith"),

    # === Classical Arabic Literature (Adab) ===
    ("0275AH", "0255Jahiz", "0255Jahiz.Hayawan", "الحيوان", "Book of Animals", "al-Jahiz", 255, "adab"),
    ("0275AH", "0255Jahiz", "0255Jahiz.Bukhala", "البخلاء", "Book of Misers", "al-Jahiz", 255, "adab"),
    ("0275AH", "0276IbnQutayba", "0276IbnQutayba.CuyunAkhbar", "عيون الأخبار", "Fountains of Information", "Ibn Qutaybah", 276, "adab"),

    # === Maqamat ===
    ("0400AH", "0398BadicZamanHamadhani", "0398BadicZamanHamadhani.Maqamat", "مقامات بديع الزمان", "Maqamat", "al-Hamadhani", 398, "maqama"),
    ("0525AH", "0516Hariri", "0516Hariri.Maqamat", "مقامات الحريري", "Maqamat al-Hariri", "al-Hariri", 516, "maqama"),

    # === History ===
    ("0325AH", "0310Tabari", "0310Tabari.Tarikh", "تاريخ الطبري", "History of al-Tabari", "al-Tabari", 310, "history"),
    ("0650AH", "0630IbnAthir", "0630IbnAthir.Kamil", "الكامل في التاريخ", "The Complete History", "Ibn al-Athir", 630, "history"),
    ("0850AH", "0808IbnKhaldun", "0808IbnKhaldun.Muqaddima", "مقدمة ابن خلدون", "Muqaddimah", "Ibn Khaldun", 808, "history"),

    # === Sira (Biography) ===
    ("0225AH", "0213IbnHisham", "0213IbnHisham.Sira", "السيرة النبوية", "Prophetic Biography", "Ibn Hisham", 218, "sira"),

    # === Philosophy ===
    ("0350AH", "0339Farabi", "0339Farabi.AraAhlMadinaFadila", "آراء أهل المدينة الفاضلة", "Virtuous City", "al-Farabi", 339, "philosophy"),
    ("0450AH", "0428IbnSina", "0428IbnSina.Shifa", "الشفاء", "The Book of Healing", "Ibn Sina", 428, "philosophy"),
    ("0600AH", "0595IbnRushd", "0595IbnRushd.TahafutTahafut", "تهافت التهافت", "Incoherence of the Incoherence", "Ibn Rushd", 595, "philosophy"),

    # === Sufism ===
    ("0525AH", "0505Ghazali", "0505Ghazali.IhsacCalumDin", "إحياء علوم الدين", "Revival of Religious Sciences", "al-Ghazali", 505, "sufism"),
    ("0650AH", "0638IbnArabi", "0638IbnArabi.FutuhatMakkiyya", "الفتوحات المكية", "Meccan Revelations", "Ibn Arabi", 638, "sufism"),

    # === Fiqh (Jurisprudence) ===
    ("0225AH", "0204Shafici", "0204Shafici.Umm", "الأم", "The Mother", "al-Shafi'i", 204, "fiqh"),
    ("0250AH", "0241IbnHanbal", "0241IbnHanbal.Musnad", "مسند أحمد", "Musnad Ahmad", "Ahmad ibn Hanbal", 241, "hadith"),

    # === Grammar ===
    ("0200AH", "0180Sibawayh", "0180Sibawayh.Kitab", "الكتاب", "The Book (Grammar)", "Sibawayh", 180, "grammar"),
    ("0400AH", "0392IbnJinni", "0392IbnJinni.Khasais", "الخصائص", "The Characteristics", "Ibn Jinni", 392, "grammar"),

    # === Geography/Travel ===
    ("0350AH", "0346Masudi", "0346Masudi.MurujDhahab", "مروج الذهب", "Meadows of Gold", "al-Masudi", 346, "geography"),
    ("0800AH", "0779IbnBattworkouta", "0779IbnBattworkouta.Rihla", "رحلة ابن بطوطة", "Travels of Ibn Battuta", "Ibn Battuta", 779, "travel"),

    # === Tafsir ===
    ("0325AH", "0310Tabari", "0310Tabari.JamicBayan", "جامع البيان", "Tafsir al-Tabari", "al-Tabari", 310, "tafsir"),
    ("0775AH", "0774IbnKathir", "0774IbnKathir.TafsirQuran", "تفسير ابن كثير", "Tafsir Ibn Kathir", "Ibn Kathir", 774, "tafsir"),

    # === Medicine ===
    ("0450AH", "0428IbnSina", "0428IbnSina.QanunFiTibb", "القانون في الطب", "Canon of Medicine", "Ibn Sina", 428, "medicine"),
]


def parse_openiti_file(file_path: Path) -> Tuple[str, Dict]:
    """Parse an OpenITI mARkdown file."""
    metadata = {}
    text_lines = []

    try:
        for encoding in ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            return "", {}

        lines = content.split('\n')

        for line in lines:
            if line.startswith('#META#') or line.startswith('######'):
                if '#META#' in line:
                    parts = line.replace('#META#', '').strip().split('::')
                    if len(parts) >= 2:
                        metadata[parts[0].strip()] = '::'.join(parts[1:]).strip()
                continue

            if re.match(r'^PageV\d+P\d+', line):
                continue
            if line.startswith('ms'):
                continue

            line = re.sub(r'@[A-Z]+@', '', line)
            line = re.sub(r'~~', '', line)
            line = line.strip()

            if line:
                text_lines.append(line)

        text = '\n'.join(text_lines)
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        metadata['word_count'] = len(arabic_words)

        return text, metadata

    except Exception as e:
        console.print(f"[red]Error parsing {file_path}: {e}[/red]")
        return "", {}


def clone_repo(repo_name: str, output_dir: Path) -> Optional[Path]:
    """Clone an OpenITI repository."""
    repo_url = f"https://github.com/OpenITI/{repo_name}.git"
    repo_path = output_dir / "repos" / repo_name

    if repo_path.exists():
        console.print(f"[yellow]Repository {repo_name} exists[/yellow]")
        return repo_path

    try:
        console.print(f"[cyan]Cloning {repo_url}...[/cyan]")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            capture_output=True,
            timeout=300
        )
        if result.returncode == 0:
            return repo_path
        else:
            console.print(f"[red]Clone failed: {result.stderr.decode()}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Failed to clone {repo_name}: {e}[/red]")
        return None


def find_best_text_file(book_path: Path) -> Optional[Path]:
    """Find the best text file in a book directory."""
    if not book_path.exists():
        return None

    # Priority: .completed > .mARkdown > no extension (Shamela)
    for suffix in ['.completed', '.mARkdown', '']:
        files = list(book_path.glob(f"*{suffix}")) if suffix else [
            f for f in book_path.iterdir()
            if f.is_file() and not f.suffix and 'Shamela' in f.name
        ]
        if files:
            # Prefer larger files
            files.sort(key=lambda f: f.stat().st_size, reverse=True)
            return files[0]

    # Fallback: any text-like file
    for f in book_path.iterdir():
        if f.is_file() and not f.suffix.startswith('.yml'):
            return f

    return None


def download_curated_texts(output_dir: Path) -> List[ArabicText]:
    """Download curated selection of Arabic texts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = output_dir / "texts"
    texts_dir.mkdir(exist_ok=True)

    results = []
    repos_cloned = set()

    console.print(f"\n[bold blue]Downloading {len(CURATED_ARABIC_TEXTS)} curated Arabic texts...[/bold blue]")

    for i, (repo, author, book, title_ar, title_en, author_name, death_ah, genre) in enumerate(CURATED_ARABIC_TEXTS):
        console.print(f"\n[cyan][{i+1}/{len(CURATED_ARABIC_TEXTS)}] {title_en} ({title_ar})[/cyan]")

        # Clone repository
        if repo not in repos_cloned:
            repo_path = clone_repo(repo, output_dir)
            if repo_path:
                repos_cloned.add(repo)
        else:
            repo_path = output_dir / "repos" / repo

        if not repo_path or not repo_path.exists():
            console.print(f"[yellow]  Skipping: repository not available[/yellow]")
            continue

        # Find book directory
        book_path = repo_path / "data" / author / book
        if not book_path.exists():
            # Try alternate paths
            alt_paths = list((repo_path / "data").glob(f"*{author.split('AH')[0]}*/{book}*"))
            if alt_paths:
                book_path = alt_paths[0]
            else:
                console.print(f"[yellow]  Skipping: book not found at {book_path}[/yellow]")
                continue

        # Find text file
        text_file = find_best_text_file(book_path)
        if not text_file:
            console.print(f"[yellow]  Skipping: no text file found[/yellow]")
            continue

        # Parse file
        text, metadata = parse_openiti_file(text_file)
        if not text or len(text) < 1000:
            console.print(f"[yellow]  Skipping: text too short ({len(text)} chars)[/yellow]")
            continue

        # Calculate CE date
        ce_year = int(death_ah * 0.97 + 622)

        # Create record
        record = ArabicText(
            id=f"ar_{i+1:04d}_{book.replace('.', '_')}",
            title=title_ar,
            title_transliterated=title_en,
            author=author_name,
            author_death_ah=death_ah,
            author_death_ce=ce_year,
            genre=genre,
            text=text,
            word_count=metadata.get('word_count', len(text.split())),
        )

        # Save
        out_file = texts_dir / f"{record.id}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, ensure_ascii=False, indent=2)

        results.append(record)
        console.print(f"[green]  ✓ Downloaded: {record.word_count:,} words from {text_file.name}[/green]")

    return results


def save_corpus_metadata(results: List[ArabicText], output_dir: Path):
    """Save corpus metadata and manifest."""
    output_dir = Path(output_dir)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source": "OpenITI (via Al-Maktaba Al-Shamela)",
        "source_url": "https://shamela.ws",
        "corpus_url": "https://github.com/OpenITI",
        "license": "CC-BY-NC-SA 4.0",
        "language": "ar",
        "total_texts": len(results),
        "total_words": sum(r.word_count or 0 for r in results),
        "texts": [{k: v for k, v in r.to_dict().items() if k != 'text'} for r in results]
    }

    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    manifest = {
        "corpus": "arabic_classical",
        "language": "ar",
        "count": len(results),
        "texts": [
            {
                "id": r.id,
                "title": r.title,
                "title_en": r.title_transliterated,
                "author": r.author,
                "author_death_ah": r.author_death_ah,
                "author_death_ce": r.author_death_ce,
                "genre": r.genre,
                "word_count": r.word_count,
                "file": f"texts/{r.id}.json"
            }
            for r in results
        ]
    }

    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


@click.command()
@click.option('--output', '-o', default='/Volumes/MacExt/narrative_corpus/arabic',
              type=click.Path(), help='Output directory')
def main(output: str):
    """Download Arabic classical literature from OpenITI corpus."""
    output_path = Path(output)

    if output_path.parts[1] == 'Volumes':
        volume = Path('/') / output_path.parts[1] / output_path.parts[2]
        if not volume.exists():
            console.print(f"[red]Volume not found: {volume}[/red]")
            return

    console.print("=" * 60)
    console.print("[bold]ARABIC CLASSICAL LITERATURE DOWNLOADER[/bold]")
    console.print("Source: OpenITI Corpus (from Al-Maktaba Al-Shamela)")
    console.print(f"Output: {output_path}")
    console.print("=" * 60)

    results = download_curated_texts(output_path)

    if not results:
        console.print("[red]No texts downloaded![/red]")
        return

    save_corpus_metadata(results, output_path)

    # Create symlink
    project_link = Path("data/raw/arabic")
    try:
        if project_link.exists() or project_link.is_symlink():
            project_link.unlink()
        project_link.symlink_to(output_path)
        console.print(f"\n[green]Symlink created: {project_link} -> {output_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not create symlink: {e}[/yellow]")

    console.print(f"\n[bold green]✓ Downloaded {len(results)} texts[/bold green]")
    console.print(f"Total words: {sum(r.word_count or 0 for r in results):,}")

    # Summary by genre
    console.print("\n[bold]Summary by Genre:[/bold]")
    genres = {}
    for r in results:
        genres[r.genre] = genres.get(r.genre, 0) + 1
    for genre, count in sorted(genres.items(), key=lambda x: -x[1]):
        console.print(f"  {genre}: {count}")

    console.print(f"\n[bold green]Done! Corpus ready at: {output_path}[/bold green]")


if __name__ == "__main__":
    main()
