"""
Syosetu (小説家になろう) Corpus Pipeline

Downloads and processes Japanese web novels from Syosetu for narrative analysis.
Critical for testing kishōtenketsu hypothesis on native Japanese fiction.

API Documentation: https://dev.syosetu.com/man/api/

Features:
    - Novel search via Syosetu API
    - Genre-stratified sampling
    - Popularity-weighted sampling (by points/bookmarks)
    - Japanese text handling (UTF-8)
    - Rate limiting compliance
"""

import json
import time
import random
import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


@dataclass
class SyosetuNovel:
    """Represents a novel from Syosetu."""
    ncode: str  # Novel code (unique identifier)
    title: str
    author: str  # writer
    author_id: Optional[int] = None  # userid
    genre: int = 0
    genre_name: str = ""
    keyword: str = ""  # Tags/keywords
    story: str = ""  # Synopsis
    general_firstup: str = ""  # First upload date
    general_lastup: str = ""  # Last update date
    novel_type: int = 1  # 1=serial, 2=short
    end_flag: int = 0  # 0=ongoing, 1=complete
    general_all_no: int = 0  # Total chapters
    length: int = 0  # Character count
    time: int = 0  # Reading time (minutes)
    global_point: int = 0  # Total points
    daily_point: int = 0
    weekly_point: int = 0
    monthly_point: int = 0
    fav_novel_cnt: int = 0  # Bookmark count
    review_cnt: int = 0
    all_point: int = 0  # Rating points
    all_hyoka_cnt: int = 0  # Rating count
    text: Optional[str] = None
    chapters: List[Dict] = field(default_factory=list)
    source: str = "syosetu"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_api_response(cls, data: dict) -> "SyosetuNovel":
        """Create from Syosetu API response."""
        return cls(
            ncode=data.get('ncode', ''),
            title=data.get('title', ''),
            author=data.get('writer', 'Unknown'),
            author_id=data.get('userid'),
            genre=data.get('genre', 0),
            keyword=data.get('keyword', ''),
            story=data.get('story', ''),
            general_firstup=data.get('general_firstup', ''),
            general_lastup=data.get('general_lastup', ''),
            novel_type=data.get('novel_type', 1),
            end_flag=data.get('end', 0),
            general_all_no=data.get('general_all_no', 0),
            length=data.get('length', 0),
            time=data.get('time', 0),
            global_point=data.get('global_point', 0),
            daily_point=data.get('daily_point', 0),
            weekly_point=data.get('weekly_point', 0),
            monthly_point=data.get('monthly_point', 0),
            fav_novel_cnt=data.get('fav_novel_cnt', 0),
            review_cnt=data.get('review_cnt', 0),
            all_point=data.get('all_point', 0),
            all_hyoka_cnt=data.get('all_hyoka_cnt', 0),
        )


# Syosetu genre mappings
SYOSETU_GENRES = {
    # Big Genres
    1: "恋愛 (Romance)",
    2: "ファンタジー (Fantasy)",
    3: "文芸 (Literature)",
    4: "SF (Science Fiction)",
    99: "その他 (Other)",
    98: "ノンジャンル (Non-genre)",
    
    # Sub-genres (selected)
    101: "異世界〔恋愛〕(Isekai Romance)",
    102: "現実世界〔恋愛〕(Real World Romance)",
    201: "ハイファンタジー (High Fantasy)",
    202: "ローファンタジー (Low Fantasy)",
    301: "純文学 (Pure Literature)",
    302: "ヒューマンドラマ (Human Drama)",
    303: "歴史 (Historical)",
    304: "推理 (Mystery)",
    305: "ホラー (Horror)",
    306: "アクション (Action)",
    307: "コメディー (Comedy)",
    401: "VRゲーム (VR Game)",
    402: "宇宙 (Space)",
    403: "空想科学 (Speculative Science)",
    404: "パニック (Panic)",
}


class SyosetuPipeline:
    """
    Pipeline for downloading Japanese web novels from Syosetu.
    
    Usage:
        pipeline = SyosetuPipeline()
        novels = pipeline.search(genre=201, limit=100)  # High Fantasy
        sample = pipeline.sample(novels, n=50, method='popularity')
        pipeline.download_texts(sample)
        pipeline.save_corpus(sample, output_dir)
    """
    
    API_URL = "https://api.syosetu.com/novelapi/api/"
    NOVEL_URL = "https://ncode.syosetu.com/{ncode}/"
    CHAPTER_URL = "https://ncode.syosetu.com/{ncode}/{chapter}/"
    
    # Rate limiting (conservative)
    REQUEST_DELAY = 1.0  # seconds between API requests
    DOWNLOAD_DELAY = 2.0  # seconds between novel downloads
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize pipeline.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = cache_dir or Path("data/cache/syosetu")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
    
    def _rate_limit(self, delay: float = None):
        """Enforce rate limiting."""
        delay = delay or self.REQUEST_DELAY
        elapsed = time.time() - self.last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request_time = time.time()
    
    def _api_request(self, params: dict) -> List[dict]:
        """Make API request to Syosetu."""
        self._rate_limit()
        
        # Set output format to JSON
        params['out'] = 'json'
        
        url = self.API_URL + '?' + urllib.parse.urlencode(params)
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'FunctorialNarrativeAnalysis/1.0 (Research)',
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                
                # First element is count metadata
                if data and isinstance(data, list) and len(data) > 1:
                    return data[1:]  # Skip count metadata
                return []
        except Exception as e:
            console.print(f"[red]API error: {e}[/red]")
            return []
    
    def search(
        self,
        genre: Optional[int] = None,
        biggenre: Optional[int] = None,
        keyword: Optional[str] = None,
        novel_type: Optional[int] = None,  # 1=serial, 2=short
        end_flag: Optional[int] = None,  # 0=ongoing, 1=complete
        min_length: int = 10000,  # Minimum character count
        max_length: int = 500000,
        order: str = 'weekly',  # weekly, monthly, hyoka (rating), favnovelcnt (bookmarks)
        limit: int = 100,
    ) -> List[SyosetuNovel]:
        """
        Search for novels via API.
        
        Args:
            genre: Genre code (see SYOSETU_GENRES)
            biggenre: Big genre code (1-4, 98, 99)
            keyword: Search keyword
            novel_type: 1=serial, 2=short story
            end_flag: 0=ongoing, 1=complete
            min_length: Minimum character count
            max_length: Maximum character count
            order: Sort order (weekly, monthly, hyoka, favnovelcnt, lengthdesc)
            limit: Maximum results (API max is 500 per request)
            
        Returns:
            List of SyosetuNovel objects
        """
        params = {
            'lim': min(limit, 500),
            'order': order,
        }
        
        if genre:
            params['genre'] = genre
        if biggenre:
            params['biggenre'] = biggenre
        if keyword:
            params['word'] = keyword
        if novel_type:
            params['type'] = novel_type
        if end_flag is not None:
            params['end'] = end_flag
        if min_length:
            params['minlen'] = min_length
        if max_length:
            params['maxlen'] = max_length
        
        console.print(f"[yellow]Searching Syosetu API...[/yellow]")
        
        novels = []
        results = self._api_request(params)
        
        for item in results:
            novel = SyosetuNovel.from_api_response(item)
            novel.genre_name = SYOSETU_GENRES.get(novel.genre, f"Unknown ({novel.genre})")
            novels.append(novel)
        
        console.print(f"[green]✓ Found {len(novels)} novels[/green]")
        return novels
    
    def search_by_genres(
        self,
        genres: List[int],
        per_genre: int = 50,
        **kwargs
    ) -> List[SyosetuNovel]:
        """
        Search across multiple genres for stratified sampling.
        
        Args:
            genres: List of genre codes
            per_genre: Novels per genre
            **kwargs: Additional search parameters
            
        Returns:
            Combined list of novels
        """
        all_novels = []
        
        for genre in genres:
            console.print(f"[cyan]Searching genre {genre}: {SYOSETU_GENRES.get(genre, 'Unknown')}[/cyan]")
            novels = self.search(genre=genre, limit=per_genre, **kwargs)
            all_novels.extend(novels)
        
        console.print(f"[green]✓ Total: {len(all_novels)} novels across {len(genres)} genres[/green]")
        return all_novels
    
    def sample(
        self,
        novels: List[SyosetuNovel],
        n: int = 100,
        method: str = 'random',
        seed: int = 42,
    ) -> List[SyosetuNovel]:
        """
        Sample novels from search results.
        
        Args:
            novels: List of novels to sample from
            n: Number to sample
            method: 'random', 'popularity' (by global_point), or 'bookmarks'
            seed: Random seed
            
        Returns:
            Sampled list
        """
        random.seed(seed)
        
        if len(novels) <= n:
            return novels
        
        if method == 'random':
            return random.sample(novels, n)
        
        elif method == 'popularity':
            # Weight by global points
            sorted_novels = sorted(novels, key=lambda x: x.global_point, reverse=True)
            # Take top half by popularity, then random sample
            top_half = sorted_novels[:len(sorted_novels)//2]
            return random.sample(top_half, min(n, len(top_half)))
        
        elif method == 'bookmarks':
            # Weight by bookmark count
            sorted_novels = sorted(novels, key=lambda x: x.fav_novel_cnt, reverse=True)
            top_half = sorted_novels[:len(sorted_novels)//2]
            return random.sample(top_half, min(n, len(top_half)))
        
        return random.sample(novels, n)
    
    def _download_novel_text(self, novel: SyosetuNovel) -> Optional[str]:
        """
        Download full text of a novel.
        
        Note: This scrapes the web pages as there's no text API.
        """
        self._rate_limit(self.DOWNLOAD_DELAY)
        
        ncode = novel.ncode.lower()
        
        if novel.novel_type == 2:  # Short story (single page)
            url = self.NOVEL_URL.format(ncode=ncode)
        else:  # Serial (first chapter for now)
            url = self.CHAPTER_URL.format(ncode=ncode, chapter=1)
        
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'FunctorialNarrativeAnalysis/1.0 (Research)',
            })
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode('utf-8')
                
                # Extract main text (between <div id="novel_honbun"> tags)
                import re
                match = re.search(r'<div id="novel_honbun"[^>]*>(.*?)</div>', html, re.DOTALL)
                if match:
                    text = match.group(1)
                    # Clean HTML tags
                    text = re.sub(r'<[^>]+>', '\n', text)
                    text = re.sub(r'\n+', '\n', text).strip()
                    return text
                
                # Alternative: look for novel_view class
                match = re.search(r'<div class="novel_view"[^>]*>(.*?)</div>', html, re.DOTALL)
                if match:
                    text = match.group(1)
                    text = re.sub(r'<[^>]+>', '\n', text)
                    text = re.sub(r'\n+', '\n', text).strip()
                    return text
                
                return None
        except Exception as e:
            console.print(f"[red]Download error for {ncode}: {e}[/red]")
            return None
    
    def download_texts(
        self,
        novels: List[SyosetuNovel],
        max_chapters: int = 10,  # For serials, download first N chapters
    ) -> List[SyosetuNovel]:
        """
        Download texts for novels.
        
        Args:
            novels: Novels to download
            max_chapters: Maximum chapters to download for serials
            
        Returns:
            Novels with text downloaded
        """
        successful = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading novels...", total=len(novels))
            
            for novel in novels:
                progress.update(task, description=f"[cyan]{novel.title[:30]}...")
                
                text = self._download_novel_text(novel)
                
                if text and len(text) > 1000:
                    novel.text = text
                    successful.append(novel)
                
                progress.advance(task)
        
        console.print(f"[green]✓ Downloaded {len(successful)}/{len(novels)} novels[/green]")
        return successful
    
    def save_corpus(
        self,
        novels: List[SyosetuNovel],
        output_dir: Path,
    ) -> None:
        """
        Save corpus to disk.
        
        Args:
            novels: Novels to save
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_novels': len(novels),
            'source': 'syosetu',
            'novels': [
                {k: v for k, v in n.to_dict().items() if k != 'text'}
                for n in novels
            ]
        }
        
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save texts
        texts_dir = output_dir / 'texts'
        texts_dir.mkdir(exist_ok=True)
        
        for novel in novels:
            if novel.text:
                with open(texts_dir / f'{novel.ncode}.json', 'w', encoding='utf-8') as f:
                    json.dump(novel.to_dict(), f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Saved corpus to {output_dir}[/green]")
    
    def get_genre_distribution(self, novels: List[SyosetuNovel]) -> Dict[str, int]:
        """Get distribution of novels by genre."""
        dist = {}
        for novel in novels:
            genre = novel.genre_name or f"Unknown ({novel.genre})"
            dist[genre] = dist.get(genre, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))


def main():
    """CLI entry point."""
    import click
    
    @click.command()
    @click.option('--genre', '-g', multiple=True, type=int,
                  help='Genre codes to search (can specify multiple)')
    @click.option('--n-novels', '-n', default=50, help='Number of novels to download')
    @click.option('--output', '-o', default='data/raw/syosetu', help='Output directory')
    @click.option('--method', '-m', default='popularity',
                  type=click.Choice(['random', 'popularity', 'bookmarks']),
                  help='Sampling method')
    @click.option('--completed', is_flag=True, help='Only completed novels')
    @click.option('--seed', '-s', default=42, help='Random seed')
    def download_corpus(genre, n_novels, output, method, completed, seed):
        """Download Japanese web novel corpus from Syosetu."""
        pipeline = SyosetuPipeline()
        
        # Default genres: diverse selection
        if not genre:
            genre = [201, 202, 301, 302, 307]  # Fantasy, Literature, Comedy
        
        # Search
        novels = pipeline.search_by_genres(
            list(genre),
            per_genre=n_novels,
            end_flag=1 if completed else None,
        )
        
        # Sample
        sample = pipeline.sample(novels, n=n_novels, method=method, seed=seed)
        console.print(f"[cyan]Sampled {len(sample)} novels[/cyan]")
        
        # Show genre distribution
        dist = pipeline.get_genre_distribution(sample)
        console.print("\n[bold]Genre distribution:[/bold]")
        for genre_name, count in list(dist.items())[:5]:
            console.print(f"  {genre_name}: {count}")
        
        # Download
        successful = pipeline.download_texts(sample)
        
        # Save
        pipeline.save_corpus(successful, Path(output))
        
        console.print(f"\n[bold green]Done! Downloaded {len(successful)} novels to {output}[/bold green]")
    
    download_corpus()


if __name__ == "__main__":
    main()
