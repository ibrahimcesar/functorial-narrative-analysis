#!/usr/bin/env python3
"""
Convert Early Chinese Text (ECT-KRP) corpus to standard JSON format.

Converts the JSONL files from the ect-krp repository to our standard
text JSON format for narrative analysis.

Usage:
    python scripts/convert_chinese_ect.py --input /Volumes/MacExt/narrative_corpus/chinese_ect
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import click
from rich.console import Console

console = Console()

# Metadata mapping: KR ID -> (title_zh, title_en, author, dynasty, genre)
TEXT_METADATA = {
    # Confucian Classics
    "KR1h0004": ("論語", "Analects", "孔子弟子", "Spring and Autumn", "philosophy"),
    "KR1h0001": ("孟子", "Mencius", "孟子", "Warring States", "philosophy"),
    "KR1a0001": ("易經", "I Ching (Book of Changes)", "傳說伏羲", "Zhou", "divination"),
    "KR1c0001": ("詩經", "Classic of Poetry", "Various", "Zhou", "poetry"),
    "KR1b0001": ("尚書", "Book of Documents", "Various", "Zhou", "history"),
    "KR1d0052": ("禮記", "Book of Rites", "Various", "Han", "ritual"),
    "KR1d0002": ("周禮", "Rites of Zhou", "Unknown", "Zhou/Han", "ritual"),
    "KR1d0026": ("儀禮", "Book of Etiquette and Ceremonies", "Unknown", "Zhou", "ritual"),
    "KR1f0001": ("孝經", "Classic of Filial Piety", "曾子", "Zhou", "philosophy"),
    "KR1d0076": ("大戴禮記", "Record of Rites by Dai the Elder", "戴德", "Han", "ritual"),

    # Spring and Autumn Commentaries
    "KR1e0001": ("春秋左傳", "Zuo Zhuan (Commentary on Spring and Autumn)", "左丘明", "Spring and Autumn", "history"),
    "KR1e0007": ("春秋公羊傳", "Gongyang Commentary", "公羊高", "Han", "commentary"),
    "KR1e0008": ("春秋穀梁傳", "Guliang Commentary", "穀梁赤", "Han", "commentary"),
    "KR1e0122": ("春秋繁露", "Luxuriant Dew of Spring and Autumn", "董仲舒", "Han", "philosophy"),

    # Historical Works
    "KR2a0001": ("史記", "Records of the Grand Historian", "司馬遷", "Han", "history"),
    "KR2a0007": ("漢書", "Book of Han", "班固", "Han", "history"),
    "KR2b0001": ("竹書紀年", "Bamboo Annals", "Unknown", "Warring States", "history"),
    "KR2b0003": ("漢記", "Han Records", "荀悅", "Han", "history"),
    "KR2d0001": ("逸周書", "Lost Book of Zhou", "Unknown", "Zhou", "history"),
    "KR2d0002": ("東觀漢記", "Dongguan Records of Han", "Various", "Han", "history"),
    "KR2e0001": ("國語", "Discourses of the States", "左丘明", "Spring and Autumn", "history"),
    "KR2e0003": ("戰國策", "Strategies of the Warring States", "劉向編", "Han", "history"),
    "KR2g0003": ("晏子春秋", "Spring and Autumn of Master Yan", "晏嬰", "Spring and Autumn", "narrative"),
    "KR2i0001": ("吳越春秋", "Spring and Autumn of Wu and Yue", "趙曄", "Han", "narrative"),
    "KR2i0002": ("越絕書", "Records of the Extraordinary States", "Unknown", "Han", "narrative"),

    # Daoist Classics
    "KR5c0057": ("老子道德經", "Dao De Jing", "老子", "Spring and Autumn", "philosophy"),
    "KR5c0126": ("莊子", "Zhuangzi", "莊子", "Warring States", "philosophy"),
    "KR5c0124": ("列子", "Liezi", "列子", "Warring States", "philosophy"),

    # Legalist/Political Philosophy
    "KR3c0005": ("韓非子", "Han Feizi", "韓非", "Warring States", "philosophy"),
    "KR3c0001": ("管子", "Guanzi", "管仲", "Warring States", "philosophy"),
    "KR3c0004": ("商君書", "Book of Lord Shang", "商鞅", "Warring States", "philosophy"),

    # Confucian Philosophy
    "KR3a0001": ("孔子家語", "School Sayings of Confucius", "王肅", "Wei", "philosophy"),
    "KR3a0002": ("荀子", "Xunzi", "荀子", "Warring States", "philosophy"),
    "KR3a0004": ("新語", "New Sayings", "陸賈", "Han", "philosophy"),
    "KR3a0005": ("新書", "New Book", "賈誼", "Han", "philosophy"),
    "KR3a0006": ("鹽鐵論", "Discourses on Salt and Iron", "桓寬", "Han", "philosophy"),
    "KR3a0007": ("說苑", "Garden of Persuasions", "劉向", "Han", "narrative"),
    "KR3a0008": ("新序", "New Prefaces", "劉向", "Han", "narrative"),
    "KR3a0009": ("法言", "Model Sayings", "揚雄", "Han", "philosophy"),
    "KR3a0010": ("潛夫論", "Comments of a Recluse", "王符", "Han", "philosophy"),
    "KR3a0011": ("申鑒", "Extended Reflections", "荀悅", "Han", "philosophy"),
    "KR3a0012": ("中論", "Treatise on the Mean", "徐幹", "Han", "philosophy"),

    # Mohist
    "KR3j0002": ("墨子", "Mozi", "墨子", "Warring States", "philosophy"),

    # Military
    "KR3b0003": ("孫子兵法", "Art of War", "孫武", "Spring and Autumn", "military"),

    # Cosmological/Divinatory
    "KR3g0001": ("太玄經", "Canon of Supreme Mystery", "揚雄", "Han", "divination"),

    # Medical
    "KR3e0001": ("黃帝內經", "Yellow Emperor's Classic of Medicine", "Unknown", "Han", "medicine"),

    # Mathematical
    "KR3f0001": ("周髀算經", "Zhou Gnomon Calculation Classic", "Unknown", "Han", "mathematics"),
    "KR3f0032": ("九章算術", "Nine Chapters on Mathematical Art", "Various", "Han", "mathematics"),

    # Encyclopedic/Miscellaneous
    "KR3j0006": ("鶡冠子", "Master Heguanzi", "Unknown", "Warring States", "philosophy"),
    "KR3j0007": ("公孫龍子", "Gongsun Longzi", "公孫龍", "Warring States", "philosophy"),
    "KR3j0009": ("呂氏春秋", "Spring and Autumn of Lu Buwei", "呂不韋", "Warring States", "encyclopedia"),
    "KR3j0010": ("淮南子", "Huainanzi", "劉安", "Han", "philosophy"),
    "KR3j0023": ("白虎通", "White Tiger Hall", "班固", "Han", "encyclopedia"),
    "KR3j0024": ("獨斷", "Solitary Judgments", "蔡邕", "Han", "encyclopedia"),
    "KR3j0080": ("論衡", "Critical Essays", "王充", "Han", "philosophy"),
    "KR3j0081": ("風俗通義", "Comprehensive Meanings of Customs", "應劭", "Han", "encyclopedia"),
    "KR3j0192": ("新論", "New Treatises", "桓譚", "Han", "philosophy"),

    # Lexicography
    "KR1j0002": ("爾雅", "Erya (Dictionary)", "Unknown", "Han", "lexicon"),
    "KR1j0007": ("釋名", "Explaining Names", "劉熙", "Han", "lexicon"),
    "KR1j0018": ("說文解字", "Explaining Graphs and Analyzing Characters", "許慎", "Han", "lexicon"),

    # Mythology/Geography
    "KR3l0090": ("山海經", "Classic of Mountains and Seas", "Unknown", "Warring States", "mythology"),
    "KR3l0092": ("穆天子傳", "Biography of King Mu", "Unknown", "Warring States", "mythology"),

    # Poetry
    "KR4a0002": ("楚辭", "Songs of Chu", "屈原等", "Warring States", "poetry"),
    "KR1c0066": ("韓詩外傳", "Han School Outer Commentary on Poetry", "韓嬰", "Han", "commentary"),
}


def load_jsonl(path: Path) -> str:
    """Load JSONL file and concatenate all text."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    texts.append(data['text'])
            except:
                continue
    return ''.join(texts)


def convert_corpus(
    input_dir: Path,
    output_dir: Path,
    min_chars: int = 1000,
) -> List[Dict]:
    """
    Convert ECT-KRP corpus to standard JSON format.

    Args:
        input_dir: ECT-KRP repository directory
        output_dir: Output directory

    Returns:
        List of converted text records
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = output_dir / "texts"
    texts_dir.mkdir(exist_ok=True)

    jsonl_dir = input_dir / "jsonl"
    if not jsonl_dir.exists():
        console.print(f"[red]JSONL directory not found: {jsonl_dir}[/red]")
        return []

    results = []
    console.print(f"[blue]Converting ECT-KRP corpus from {input_dir}...[/blue]")

    for jsonl_file in sorted(jsonl_dir.glob("*.jsonl")):
        kr_id = jsonl_file.stem

        # Get metadata
        if kr_id in TEXT_METADATA:
            title_zh, title_en, author, dynasty, genre = TEXT_METADATA[kr_id]
        else:
            console.print(f"[yellow]Unknown text: {kr_id}[/yellow]")
            title_zh = kr_id
            title_en = kr_id
            author = "Unknown"
            dynasty = "Unknown"
            genre = "unknown"

        # Load text
        text = load_jsonl(jsonl_file)

        if len(text) < min_chars:
            console.print(f"[yellow]Skipping {title_zh}: too short ({len(text)} chars)[/yellow]")
            continue

        # Create record
        record = {
            "id": f"ect_{kr_id}",
            "kr_id": kr_id,
            "title": title_zh,
            "title_en": title_en,
            "author": author,
            "dynasty": dynasty,
            "genre": genre,
            "text": text,
            "char_count": len(text),
            "source": "ect-krp",
            "language": "zh",
        }

        # Save
        out_file = texts_dir / f"ect_{kr_id}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        results.append({k: v for k, v in record.items() if k != 'text'})
        console.print(f"[green]✓ {title_zh} ({title_en}): {len(text):,} chars[/green]")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source": "ect-krp (Early Chinese Text Corpus)",
        "license": "CC-BY-SA 4.0",
        "language": "zh",
        "total_texts": len(results),
        "total_characters": sum(r.get("char_count", 0) for r in results),
        "texts": results,
    }

    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save manifest
    manifest = {
        "corpus": "ect-krp",
        "language": "zh",
        "count": len(results),
        "texts": [
            {
                "id": r["id"],
                "title": r["title"],
                "title_en": r["title_en"],
                "author": r["author"],
                "dynasty": r["dynasty"],
                "genre": r["genre"],
                "char_count": r["char_count"],
                "file": f"texts/{r['id']}.json"
            }
            for r in results
        ]
    }

    with open(output_dir / "manifest.json", 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]✓ Converted {len(results)} texts[/bold green]")
    console.print(f"Total characters: {sum(r.get('char_count', 0) for r in results):,}")

    return results


@click.command()
@click.option('--input', '-i', 'input_dir',
              default='/Volumes/MacExt/narrative_corpus/chinese_ect',
              type=click.Path(exists=True), help='ECT-KRP repository directory')
@click.option('--output', '-o', 'output_dir',
              default='/Volumes/MacExt/narrative_corpus/chinese_classical',
              type=click.Path(), help='Output directory')
@click.option('--min-chars', '-m', default=1000, help='Minimum characters per text')
def main(input_dir: str, output_dir: str, min_chars: int):
    """
    Convert ECT-KRP corpus to standard JSON format.

    Converts the Early Chinese Text corpus from the Kanseki Repository
    to our standard JSON format for narrative analysis.
    """
    console.print("=" * 60)
    console.print("[bold]EARLY CHINESE TEXT CORPUS CONVERTER[/bold]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print("=" * 60)

    results = convert_corpus(
        Path(input_dir),
        Path(output_dir),
        min_chars=min_chars
    )

    # Print summary by genre
    console.print("\n[bold]Summary by Genre:[/bold]")
    genres = {}
    for r in results:
        genres[r["genre"]] = genres.get(r["genre"], 0) + 1
    for genre, count in sorted(genres.items(), key=lambda x: -x[1]):
        console.print(f"  {genre}: {count}")

    # Print summary by dynasty
    console.print("\n[bold]Summary by Dynasty:[/bold]")
    dynasties = {}
    for r in results:
        dynasties[r["dynasty"]] = dynasties.get(r["dynasty"], 0) + 1
    for dynasty, count in sorted(dynasties.items(), key=lambda x: -x[1]):
        console.print(f"  {dynasty}: {count}")

    console.print(f"\n[bold green]Done! Corpus ready at: {output_dir}[/bold green]")


if __name__ == "__main__":
    main()
