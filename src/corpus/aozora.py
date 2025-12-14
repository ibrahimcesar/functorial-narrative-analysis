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

        # Expanded curated list of Japanese literature
        # Organized by author for easy reference
        curated = [
            # === Natsume Soseki (夏目漱石) - 10 works ===
            ("夏目漱石", "吾輩は猫である", "https://www.aozora.gr.jp/cards/000148/files/789_14547.html"),
            ("夏目漱石", "坊っちゃん", "https://www.aozora.gr.jp/cards/000148/files/752_14964.html"),
            ("夏目漱石", "こころ", "https://www.aozora.gr.jp/cards/000148/files/773_14560.html"),
            ("夏目漱石", "三四郎", "https://www.aozora.gr.jp/cards/000148/files/794_14946.html"),
            ("夏目漱石", "それから", "https://www.aozora.gr.jp/cards/000148/files/1746_15061.html"),
            ("夏目漱石", "門", "https://www.aozora.gr.jp/cards/000148/files/783_14958.html"),
            ("夏目漱石", "草枕", "https://www.aozora.gr.jp/cards/000148/files/776_14941.html"),
            ("夏目漱石", "明暗", "https://www.aozora.gr.jp/cards/000148/files/780_14959.html"),
            ("夏目漱石", "虞美人草", "https://www.aozora.gr.jp/cards/000148/files/761_14938.html"),
            ("夏目漱石", "行人", "https://www.aozora.gr.jp/cards/000148/files/775_14957.html"),

            # === Akutagawa Ryunosuke (芥川龍之介) - 15 works ===
            ("芥川龍之介", "羅生門", "https://www.aozora.gr.jp/cards/000879/files/127_15260.html"),
            ("芥川龍之介", "鼻", "https://www.aozora.gr.jp/cards/000879/files/42_15228.html"),
            ("芥川龍之介", "藪の中", "https://www.aozora.gr.jp/cards/000879/files/179_15255.html"),
            ("芥川龍之介", "河童", "https://www.aozora.gr.jp/cards/000879/files/69_14933.html"),
            ("芥川龍之介", "蜘蛛の糸", "https://www.aozora.gr.jp/cards/000879/files/92_490.html"),
            ("芥川龍之介", "地獄変", "https://www.aozora.gr.jp/cards/000879/files/60_14935.html"),
            ("芥川龍之介", "杜子春", "https://www.aozora.gr.jp/cards/000879/files/170_15144.html"),
            ("芥川龍之介", "トロッコ", "https://www.aozora.gr.jp/cards/000879/files/43016_17447.html"),
            ("芥川龍之介", "歯車", "https://www.aozora.gr.jp/cards/000879/files/42377_15461.html"),
            ("芥川龍之介", "蜜柑", "https://www.aozora.gr.jp/cards/000879/files/99_491.html"),
            ("芥川龍之介", "奉教人の死", "https://www.aozora.gr.jp/cards/000879/files/68_14934.html"),
            ("芥川龍之介", "戯作三昧", "https://www.aozora.gr.jp/cards/000879/files/124_15257.html"),
            ("芥川龍之介", "或日の大石内蔵助", "https://www.aozora.gr.jp/cards/000879/files/108_15075.html"),
            ("芥川龍之介", "舞踏会", "https://www.aozora.gr.jp/cards/000879/files/102_492.html"),
            ("芥川龍之介", "秋", "https://www.aozora.gr.jp/cards/000879/files/123_15256.html"),

            # === Dazai Osamu (太宰治) - 10 works ===
            ("太宰治", "人間失格", "https://www.aozora.gr.jp/cards/000035/files/301_14912.html"),
            ("太宰治", "走れメロス", "https://www.aozora.gr.jp/cards/000035/files/1567_14913.html"),
            ("太宰治", "斜陽", "https://www.aozora.gr.jp/cards/000035/files/1565_8559.html"),
            ("太宰治", "津軽", "https://www.aozora.gr.jp/cards/000035/files/2282_15074.html"),
            ("太宰治", "ヴィヨンの妻", "https://www.aozora.gr.jp/cards/000035/files/2253_13081.html"),
            ("太宰治", "富嶽百景", "https://www.aozora.gr.jp/cards/000035/files/270_14914.html"),
            ("太宰治", "女生徒", "https://www.aozora.gr.jp/cards/000035/files/275_13903.html"),
            ("太宰治", "お伽草紙", "https://www.aozora.gr.jp/cards/000035/files/307_14915.html"),
            ("太宰治", "パンドラの匣", "https://www.aozora.gr.jp/cards/000035/files/1566_8558.html"),
            ("太宰治", "グッド・バイ", "https://www.aozora.gr.jp/cards/000035/files/1563_8556.html"),

            # === Miyazawa Kenji (宮沢賢治) - 8 works ===
            ("宮沢賢治", "銀河鉄道の夜", "https://www.aozora.gr.jp/cards/000081/files/456_15050.html"),
            ("宮沢賢治", "風の又三郎", "https://www.aozora.gr.jp/cards/000081/files/462_15405.html"),
            ("宮沢賢治", "セロ弾きのゴーシュ", "https://www.aozora.gr.jp/cards/000081/files/470_15407.html"),
            ("宮沢賢治", "注文の多い料理店", "https://www.aozora.gr.jp/cards/000081/files/43754_17659.html"),
            ("宮沢賢治", "よだかの星", "https://www.aozora.gr.jp/cards/000081/files/473_15406.html"),
            ("宮沢賢治", "どんぐりと山猫", "https://www.aozora.gr.jp/cards/000081/files/44921_19093.html"),
            ("宮沢賢治", "やまなし", "https://www.aozora.gr.jp/cards/000081/files/46605_23911.html"),
            ("宮沢賢治", "ポラーノの広場", "https://www.aozora.gr.jp/cards/000081/files/1935_18387.html"),

            # === Mori Ogai (森鷗外) - 8 works ===
            ("森鷗外", "舞姫", "https://www.aozora.gr.jp/cards/000129/files/2078_15963.html"),
            ("森鷗外", "高瀬舟", "https://www.aozora.gr.jp/cards/000129/files/691_15352.html"),
            ("森鷗外", "山椒大夫", "https://www.aozora.gr.jp/cards/000129/files/689_15353.html"),
            ("森鷗外", "雁", "https://www.aozora.gr.jp/cards/000129/files/673_15348.html"),
            ("森鷗外", "阿部一族", "https://www.aozora.gr.jp/cards/000129/files/692_15354.html"),
            ("森鷗外", "ヰタ・セクスアリス", "https://www.aozora.gr.jp/cards/000129/files/695_15355.html"),
            ("森鷗外", "青年", "https://www.aozora.gr.jp/cards/000129/files/688_15347.html"),
            ("森鷗外", "渋江抽斎", "https://www.aozora.gr.jp/cards/000129/files/694_15356.html"),

            # === Tanizaki Junichiro (谷崎潤一郎) - 6 works ===
            ("谷崎潤一郎", "痴人の愛", "https://www.aozora.gr.jp/cards/001383/files/56646_60018.html"),
            ("谷崎潤一郎", "刺青", "https://www.aozora.gr.jp/cards/001383/files/56622_59377.html"),
            ("谷崎潤一郎", "春琴抄", "https://www.aozora.gr.jp/cards/001383/files/56866_64227.html"),
            ("谷崎潤一郎", "卍", "https://www.aozora.gr.jp/cards/001383/files/56649_59802.html"),
            ("谷崎潤一郎", "秘密", "https://www.aozora.gr.jp/cards/001383/files/56619_58025.html"),
            ("谷崎潤一郎", "蓼喰ふ虫", "https://www.aozora.gr.jp/cards/001383/files/56640_59799.html"),

            # === Izumi Kyoka (泉鏡花) - 5 works ===
            ("泉鏡花", "高野聖", "https://www.aozora.gr.jp/cards/000050/files/521_19518.html"),
            ("泉鏡花", "外科室", "https://www.aozora.gr.jp/cards/000050/files/348_19545.html"),
            ("泉鏡花", "夜行巡査", "https://www.aozora.gr.jp/cards/000050/files/4578_12623.html"),
            ("泉鏡花", "草迷宮", "https://www.aozora.gr.jp/cards/000050/files/49669_39422.html"),
            ("泉鏡花", "婦系図", "https://www.aozora.gr.jp/cards/000050/files/3530_10923.html"),

            # === Nakajima Atsushi (中島敦) - 4 works ===
            ("中島敦", "山月記", "https://www.aozora.gr.jp/cards/000119/files/624_14544.html"),
            ("中島敦", "李陵", "https://www.aozora.gr.jp/cards/000119/files/621_14498.html"),
            ("中島敦", "弟子", "https://www.aozora.gr.jp/cards/000119/files/622_14501.html"),
            ("中島敦", "名人伝", "https://www.aozora.gr.jp/cards/000119/files/626_14497.html"),

            # === Kajii Motojiro (梶井基次郎) - 3 works ===
            ("梶井基次郎", "檸檬", "https://www.aozora.gr.jp/cards/000074/files/424_19826.html"),
            ("梶井基次郎", "桜の樹の下には", "https://www.aozora.gr.jp/cards/000074/files/427_19793.html"),
            ("梶井基次郎", "城のある町にて", "https://www.aozora.gr.jp/cards/000074/files/4314_14903.html"),

            # === Higuchi Ichiyo (樋口一葉) - 4 works ===
            ("樋口一葉", "たけくらべ", "https://www.aozora.gr.jp/cards/000064/files/389_15253.html"),
            ("樋口一葉", "にごりえ", "https://www.aozora.gr.jp/cards/000064/files/390_15254.html"),
            ("樋口一葉", "大つごもり", "https://www.aozora.gr.jp/cards/000064/files/388_15252.html"),
            ("樋口一葉", "十三夜", "https://www.aozora.gr.jp/cards/000064/files/391_20794.html"),

            # === Shimazaki Toson (島崎藤村) - 4 works ===
            ("島崎藤村", "破戒", "https://www.aozora.gr.jp/cards/000158/files/1498_26232.html"),
            ("島崎藤村", "夜明け前", "https://www.aozora.gr.jp/cards/000158/files/1506_26077.html"),
            ("島崎藤村", "春", "https://www.aozora.gr.jp/cards/000158/files/1501_49896.html"),
            ("島崎藤村", "家", "https://www.aozora.gr.jp/cards/000158/files/1499_49898.html"),

            # === Shiga Naoya (志賀直哉) - 5 works ===
            ("志賀直哉", "暗夜行路", "https://www.aozora.gr.jp/cards/000023/files/241_26229.html"),
            ("志賀直哉", "城の崎にて", "https://www.aozora.gr.jp/cards/000023/files/234_15342.html"),
            ("志賀直哉", "小僧の神様", "https://www.aozora.gr.jp/cards/000023/files/238_18584.html"),
            ("志賀直哉", "和解", "https://www.aozora.gr.jp/cards/000023/files/239_26094.html"),
            ("志賀直哉", "清兵衛と瓢箪", "https://www.aozora.gr.jp/cards/000023/files/1686_26295.html"),

            # === Edogawa Ranpo (江戸川乱歩) - 5 works ===
            ("江戸川乱歩", "人間椅子", "https://www.aozora.gr.jp/cards/001779/files/56648_58218.html"),
            ("江戸川乱歩", "D坂の殺人事件", "https://www.aozora.gr.jp/cards/001779/files/56632_57195.html"),
            ("江戸川乱歩", "二銭銅貨", "https://www.aozora.gr.jp/cards/001779/files/56628_57191.html"),
            ("江戸川乱歩", "屋根裏の散歩者", "https://www.aozora.gr.jp/cards/001779/files/56638_57199.html"),
            ("江戸川乱歩", "芋虫", "https://www.aozora.gr.jp/cards/001779/files/56652_58252.html"),

            # === Yokomitsu Riichi (横光利一) - 3 works ===
            ("横光利一", "機械", "https://www.aozora.gr.jp/cards/000168/files/2159_19986.html"),
            ("横光利一", "蠅", "https://www.aozora.gr.jp/cards/000168/files/2158_19983.html"),
            ("横光利一", "春は馬車に乗って", "https://www.aozora.gr.jp/cards/000168/files/2152_19884.html"),

            # === Ango Sakaguchi (坂口安吾) - 4 works ===
            ("坂口安吾", "堕落論", "https://www.aozora.gr.jp/cards/001095/files/42620_21407.html"),
            ("坂口安吾", "桜の森の満開の下", "https://www.aozora.gr.jp/cards/001095/files/42618_21410.html"),
            ("坂口安吾", "白痴", "https://www.aozora.gr.jp/cards/001095/files/42617_21406.html"),
            ("坂口安吾", "風博士", "https://www.aozora.gr.jp/cards/001095/files/42615_21404.html"),

            # === Osamu Dazai (additional) / Nagai Kafu (永井荷風) - 3 works ===
            ("永井荷風", "濹東綺譚", "https://www.aozora.gr.jp/cards/001341/files/49675_67451.html"),
            ("永井荷風", "つゆのあとさき", "https://www.aozora.gr.jp/cards/001341/files/49632_36851.html"),
            ("永井荷風", "腕くらべ", "https://www.aozora.gr.jp/cards/001341/files/55228_47045.html"),
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
