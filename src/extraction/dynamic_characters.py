"""
Dynamic Character Extraction Module

Automatically extracts character definitions from text without requiring
manual character dictionaries. Uses multiple heuristics:

1. Name frequency analysis - most mentioned names are likely main characters
2. Dialogue attribution - "said X" patterns identify speaking characters
3. Title patterns - "Mr./Mrs./Count/Prince" etc.
4. Co-occurrence analysis - characters mentioned together are related
5. Role inference - protagonists appear throughout; supporting cast is localized

Output format compatible with visualize_character_orbits.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import colorsys

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


# ============================================================================
# Configuration
# ============================================================================

ENGLISH_TITLES = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'doctor', 'prof', 'professor',
    'sir', 'lord', 'lady', 'captain', 'colonel', 'general', 'major',
    'king', 'queen', 'prince', 'princess', 'duke', 'duchess',
    'count', 'countess', 'baron', 'baroness', 'father', 'mother',
    'brother', 'sister', 'uncle', 'aunt', 'reverend', 'pastor',
    'monsieur', 'madame', 'mademoiselle', 'herr', 'frau', 'don', 'dona',
}

ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'june', 'july',
    'august', 'september', 'october', 'november', 'december',
    'chapter', 'part', 'book', 'volume', 'section', 'act', 'scene',
    'now', 'then', 'here', 'there', 'yes', 'no', 'not', 'only', 'just',
    'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very',
    'well', 'back', 'even', 'still', 'also', 'after', 'before', 'again',
    'once', 'upon', 'time', 'long', 'great', 'good', 'little', 'old', 'new',
    'first', 'last', 'next', 'same', 'own', 'never', 'ever', 'always',
    'away', 'right', 'left', 'hand', 'hands', 'eyes', 'face', 'head',
    'house', 'room', 'door', 'night', 'day', 'morning', 'evening',
    'man', 'woman', 'men', 'women', 'people', 'world', 'life', 'death',
    'god', 'heaven', 'earth', 'nature', 'love', 'heart', 'mind', 'soul',
    'thing', 'things', 'nothing', 'something', 'everything', 'anything',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'hundred', 'thousand', 'half', 'whole', 'much', 'many', 'few', 'several',
    'end', 'beginning', 'middle', 'way', 'place', 'side', 'kind', 'sort',
    'english', 'french', 'german', 'russian', 'american', 'european',
    'london', 'paris', 'rome', 'moscow', 'new york',  # common places
    'god', 'christ', 'lord',  # religious terms often capitalized
    'oh', 'ah', 'alas', 'indeed', 'perhaps', 'however', 'therefore', 'thus',
    'project', 'gutenberg', 'ebook', 'copyright', 'license',  # metadata
}

RUSSIAN_STOPWORDS = {
    'и', 'в', 'не', 'на', 'с', 'что', 'как', 'а', 'то', 'все', 'она', 'он',
    'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по',
    'только', 'её', 'мне', 'было', 'вот', 'от', 'меня', 'ещё', 'нет', 'о',
    'из', 'ему', 'теперь', 'когда', 'уже', 'вам', 'ни', 'быть', 'был', 'него',
    'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'сказал', 'себя', 'ей', 'под',
    'глава', 'часть', 'книга', 'том',  # structural
    # Common capitalized words that aren't names
    'это', 'этот', 'этого', 'этом', 'этой', 'эта', 'эти', 'этих',
    'если', 'когда', 'потом', 'тогда', 'здесь', 'там', 'сюда', 'туда',
    'они', 'оно', 'она', 'мы', 'вы', 'кто', 'что', 'который', 'которая',
    'какой', 'какая', 'такой', 'такая', 'весь', 'вся', 'все', 'всё',
    'другой', 'другая', 'самый', 'самая', 'первый', 'первая', 'последний',
    'может', 'могли', 'должен', 'должна', 'надо', 'нужно', 'можно', 'нельзя',
    'между', 'после', 'перед', 'через', 'против', 'около', 'кроме', 'вместо',
    'хотя', 'чтобы', 'потому', 'поэтому', 'однако', 'также', 'тоже', 'даже',
    'очень', 'совсем', 'слишком', 'почти', 'довольно', 'вполне', 'вообще',
    'наконец', 'вдруг', 'снова', 'опять', 'особенно', 'именно', 'просто',
    'несмотря', 'впрочем', 'поскольку', 'действительно', 'конечно', 'видимо',
    'москва', 'петербург', 'россия', 'европа', 'англия', 'франция', 'германия',
}

RUSSIAN_TITLES = {
    'князь', 'княгиня', 'граф', 'графиня', 'барон', 'баронесса',
    'господин', 'госпожа', 'мадам', 'месье', 'monsieur', 'madame',
}

# Color palette for automatic assignment (visually distinct colors)
CHARACTER_COLORS = [
    '#E91E63', '#3F51B5', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#FF5722', '#795548', '#607D8B', '#F44336',
    '#2196F3', '#8BC34A', '#FFC107', '#673AB7', '#009688',
    '#CDDC39', '#03A9F4', '#E040FB', '#00E676', '#FF6D00',
]


@dataclass
class ExtractedCharacter:
    """Represents an automatically extracted character."""
    name: str
    aliases: Set[str] = field(default_factory=set)
    mentions: int = 0
    first_appearance: float = 0.0  # normalized position [0, 1]
    last_appearance: float = 1.0
    dialogue_count: int = 0
    title: Optional[str] = None
    role: str = 'supporting'  # protagonist, major, supporting, minor
    color: str = '#888888'
    co_occurrences: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'aliases': list(self.aliases),
            'mentions': self.mentions,
            'first_appearance': self.first_appearance,
            'last_appearance': self.last_appearance,
            'dialogue_count': self.dialogue_count,
            'title': self.title,
            'role': self.role,
            'color': self.color,
            'co_occurrences': dict(self.co_occurrences.most_common(10)),
        }


class DynamicCharacterExtractor:
    """
    Extract character definitions dynamically from text.

    Usage:
        extractor = DynamicCharacterExtractor()
        characters = extractor.extract(text)
        char_dict = extractor.to_visualization_format(characters)
    """

    def __init__(
        self,
        language: str = 'en',
        min_mentions: int = 3,
        max_characters: int = 30,
        window_size: int = 200,  # words for co-occurrence
    ):
        self.language = language
        self.min_mentions = min_mentions
        self.max_characters = max_characters
        self.window_size = window_size

        if language == 'ru':
            self.stopwords = RUSSIAN_STOPWORDS
            self.titles = RUSSIAN_TITLES
        else:
            self.stopwords = ENGLISH_STOPWORDS
            self.titles = ENGLISH_TITLES

    def _detect_language(self, text: str) -> str:
        """Auto-detect language from text."""
        cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', text[:5000]))
        latin = len(re.findall(r'[a-zA-Z]', text[:5000]))
        return 'ru' if cyrillic > latin * 0.3 else 'en'

    def _extract_raw_names(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract all potential character names with their positions.

        Returns list of (name, character_position) tuples.
        """
        names_with_pos = []

        if self.language == 'ru':
            # Russian: Look for capitalized Cyrillic words
            # Pattern: Uppercase followed by lowercase Cyrillic
            pattern = r'\b([А-ЯЁ][а-яё]{2,})(?:\s+([А-ЯЁ][а-яё]+))?'
            for match in re.finditer(pattern, text):
                name = match.group(1)
                if match.group(2):
                    name = f"{match.group(1)} {match.group(2)}"
                if name.lower() not in self.stopwords and len(name) > 2:
                    names_with_pos.append((name, match.start()))

            # Also extract title + name patterns
            for title in self.titles:
                pattern = rf'{re.escape(title)}\s+([А-ЯЁ][а-яё]+)'
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group(1)
                    names_with_pos.append((name, match.start()))

        else:
            # English: Multiple extraction strategies

            # 1. Title + Name (most reliable)
            title_pattern = r'\b(' + '|'.join(self.titles) + r')\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            for match in re.finditer(title_pattern, text, re.IGNORECASE):
                name = match.group(2)
                names_with_pos.append((name, match.start()))

            # 2. Dialogue attribution (very reliable)
            dialogue_patterns = [
                r'(?:said|replied|asked|answered|exclaimed|shouted|whispered|muttered|cried|continued|added|remarked|observed|declared|announced|called|interrupted|began|responded|inquired)\s+([A-Z][a-z]+)',
                r'([A-Z][a-z]+)\s+(?:said|replied|asked|answered|exclaimed|shouted|whispered|muttered|cried|continued|added|remarked|observed|declared|announced|called|interrupted|began|responded|inquired)',
            ]
            for pattern in dialogue_patterns:
                for match in re.finditer(pattern, text):
                    name = match.group(1)
                    if name.lower() not in self.stopwords:
                        names_with_pos.append((name, match.start()))

            # 3. Capitalized words (less reliable, but catches names)
            cap_pattern = r'\b([A-Z][a-z]{2,})\b'
            for match in re.finditer(cap_pattern, text):
                name = match.group(1)
                if name.lower() not in self.stopwords:
                    names_with_pos.append((name, match.start()))

            # 4. Two-word names (First Last)
            two_word = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
            for match in re.finditer(two_word, text):
                first, last = match.group(1), match.group(2)
                if first.lower() not in self.stopwords and last.lower() not in self.stopwords:
                    # Check if it's not start of sentence
                    pos = match.start()
                    if pos > 0 and text[pos-1] not in '.!?\n':
                        names_with_pos.append((f"{first} {last}", pos))

        return names_with_pos

    def _resolve_aliases(self, name_counts: Counter) -> Dict[str, Set[str]]:
        """
        Group aliases together (e.g., "Elizabeth", "Miss Bennet", "Lizzy").

        Returns dict mapping canonical name -> set of aliases.
        """
        aliases = defaultdict(set)
        names = list(name_counts.keys())

        # Sort by frequency (most common = likely canonical)
        names.sort(key=lambda x: name_counts[x], reverse=True)

        processed = set()

        for name in names:
            if name in processed:
                continue

            canonical = name
            aliases[canonical].add(name)
            processed.add(name)

            # Check for related names
            name_parts = name.split()

            for other in names:
                if other in processed:
                    continue

                other_parts = other.split()

                # Match conditions:
                # 1. Same last name
                # 2. One is substring of other
                # 3. First name matches

                is_alias = False

                # Last name match
                if len(name_parts) >= 2 and len(other_parts) >= 2:
                    if name_parts[-1] == other_parts[-1]:
                        is_alias = True

                # First name match (if both are full names)
                if len(name_parts) >= 2 and len(other_parts) >= 1:
                    if name_parts[0] == other_parts[0]:
                        is_alias = True

                # One contains the other
                if name in other or other in name:
                    is_alias = True

                if is_alias:
                    aliases[canonical].add(other)
                    processed.add(other)

        return dict(aliases)

    def _calculate_co_occurrences(
        self,
        text: str,
        names_with_pos: List[Tuple[str, int]],
        alias_map: Dict[str, str],  # name -> canonical
    ) -> Dict[str, Counter]:
        """
        Calculate character co-occurrences within text windows.
        """
        text_len = len(text)

        # Group positions by canonical name
        canonical_positions = defaultdict(list)
        for name, pos in names_with_pos:
            canonical = alias_map.get(name, name)
            canonical_positions[canonical].append(pos)

        # Calculate window size in characters (approximate)
        char_window = self.window_size * 6  # ~6 chars per word

        co_occurrences = defaultdict(Counter)

        for name1, positions1 in canonical_positions.items():
            for pos1 in positions1:
                for name2, positions2 in canonical_positions.items():
                    if name1 == name2:
                        continue

                    for pos2 in positions2:
                        if abs(pos1 - pos2) < char_window:
                            co_occurrences[name1][name2] += 1

        return dict(co_occurrences)

    def _infer_roles(
        self,
        characters: Dict[str, ExtractedCharacter],
        total_mentions: int,
    ) -> None:
        """
        Infer character roles based on mention frequency and patterns.

        Modifies characters in place.
        """
        if not characters:
            return

        # Sort by mentions
        sorted_chars = sorted(
            characters.values(),
            key=lambda c: c.mentions,
            reverse=True
        )

        max_mentions = sorted_chars[0].mentions if sorted_chars else 1

        for i, char in enumerate(sorted_chars):
            mention_ratio = char.mentions / max_mentions
            coverage = char.last_appearance - char.first_appearance

            # Protagonist: high mentions, appears throughout
            if i == 0 and mention_ratio > 0.5 and coverage > 0.7:
                char.role = 'protagonist'
            # Major: top characters with significant coverage
            elif mention_ratio > 0.3 and coverage > 0.5:
                char.role = 'major'
            # Supporting: moderate mentions
            elif mention_ratio > 0.1:
                char.role = 'supporting'
            else:
                char.role = 'minor'

    def _assign_colors(self, characters: Dict[str, ExtractedCharacter]) -> None:
        """
        Assign distinct colors to characters based on role.

        Modifies characters in place.
        """
        # Sort by importance
        sorted_chars = sorted(
            characters.values(),
            key=lambda c: (
                {'protagonist': 0, 'major': 1, 'supporting': 2, 'minor': 3}[c.role],
                -c.mentions
            )
        )

        for i, char in enumerate(sorted_chars):
            char.color = CHARACTER_COLORS[i % len(CHARACTER_COLORS)]

    def extract(self, text: str, auto_language: bool = True) -> Dict[str, ExtractedCharacter]:
        """
        Extract character definitions from text.

        Args:
            text: Full text of the work
            auto_language: Whether to auto-detect language

        Returns:
            Dict mapping character name -> ExtractedCharacter
        """
        if auto_language:
            self.language = self._detect_language(text)
            self.stopwords = RUSSIAN_STOPWORDS if self.language == 'ru' else ENGLISH_STOPWORDS
            self.titles = RUSSIAN_TITLES if self.language == 'ru' else ENGLISH_TITLES

        text_len = len(text)

        # Step 1: Extract raw names with positions
        names_with_pos = self._extract_raw_names(text)

        # Step 2: Count frequencies
        name_counts = Counter(name for name, pos in names_with_pos)

        # Filter by minimum mentions
        name_counts = Counter({
            name: count
            for name, count in name_counts.items()
            if count >= self.min_mentions
        })

        if not name_counts:
            console.print("[yellow]Warning: No characters found with sufficient mentions[/yellow]")
            return {}

        # Step 3: Resolve aliases
        alias_groups = self._resolve_aliases(name_counts)

        # Create reverse mapping: alias -> canonical
        alias_map = {}
        for canonical, aliases in alias_groups.items():
            for alias in aliases:
                alias_map[alias] = canonical

        # Step 4: Build character objects
        characters = {}

        for canonical, aliases in alias_groups.items():
            # Sum mentions across all aliases
            total_mentions = sum(name_counts.get(alias, 0) for alias in aliases)

            if total_mentions < self.min_mentions:
                continue

            # Find first and last appearance
            positions = [
                pos for name, pos in names_with_pos
                if name in aliases
            ]

            if not positions:
                continue

            first_pos = min(positions) / text_len
            last_pos = max(positions) / text_len

            # Count dialogue attributions
            dialogue_count = 0
            for alias in aliases:
                dialogue_count += len(re.findall(
                    rf'(?:said|replied|asked)\s+{re.escape(alias)}|{re.escape(alias)}\s+(?:said|replied|asked)',
                    text,
                    re.IGNORECASE
                ))

            # Check for title
            title = None
            for t in self.titles:
                for alias in aliases:
                    if re.search(rf'\b{t}\.?\s+{re.escape(alias)}', text, re.IGNORECASE):
                        title = t.title()
                        break

            characters[canonical] = ExtractedCharacter(
                name=canonical,
                aliases=aliases,
                mentions=total_mentions,
                first_appearance=first_pos,
                last_appearance=last_pos,
                dialogue_count=dialogue_count,
                title=title,
            )

        # Step 5: Calculate co-occurrences
        co_occurrences = self._calculate_co_occurrences(text, names_with_pos, alias_map)
        for name, coocs in co_occurrences.items():
            if name in characters:
                characters[name].co_occurrences = coocs

        # Step 6: Infer roles
        total_mentions = sum(c.mentions for c in characters.values())
        self._infer_roles(characters, total_mentions)

        # Step 7: Assign colors
        self._assign_colors(characters)

        # Step 8: Limit to max characters
        if len(characters) > self.max_characters:
            sorted_chars = sorted(
                characters.items(),
                key=lambda x: x[1].mentions,
                reverse=True
            )
            characters = dict(sorted_chars[:self.max_characters])

        return characters

    def to_visualization_format(
        self,
        characters: Dict[str, ExtractedCharacter],
    ) -> Dict[str, Dict]:
        """
        Convert extracted characters to format expected by visualize_character_orbits.py

        Returns:
            Dict in format: {name: {'role': ..., 'color': ..., ...}}
        """
        result = {}

        for name, char in characters.items():
            result[name] = {
                'role': char.role,
                'color': char.color,
                'mentions': char.mentions,
                'aliases': list(char.aliases),
            }

            # Add title if present
            if char.title:
                result[name]['title'] = char.title

        return result

    def print_summary(self, characters: Dict[str, ExtractedCharacter]) -> None:
        """Print a rich summary table of extracted characters."""
        table = Table(title="Extracted Characters")

        table.add_column("Name", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Mentions", justify="right")
        table.add_column("Dialogue", justify="right")
        table.add_column("Appears", justify="center")
        table.add_column("Aliases", style="dim")
        table.add_column("Top Connections")

        # Sort by mentions
        sorted_chars = sorted(
            characters.values(),
            key=lambda c: c.mentions,
            reverse=True
        )

        for char in sorted_chars[:20]:
            # Format appearance range
            appear = f"{char.first_appearance:.0%}-{char.last_appearance:.0%}"

            # Format aliases (excluding canonical name)
            other_aliases = [a for a in char.aliases if a != char.name]
            alias_str = ", ".join(other_aliases[:3])
            if len(other_aliases) > 3:
                alias_str += f" +{len(other_aliases)-3}"

            # Format connections
            connections = [f"{n}({c})" for n, c in char.co_occurrences.most_common(3)]
            conn_str = ", ".join(connections)

            table.add_row(
                char.name,
                char.role,
                str(char.mentions),
                str(char.dialogue_count),
                appear,
                alias_str,
                conn_str,
            )

        console.print(table)


def extract_characters_from_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    min_mentions: int = 3,
    max_characters: int = 30,
) -> Dict[str, ExtractedCharacter]:
    """
    Extract characters from a text file.

    Args:
        input_path: Path to JSON file with 'text' field
        output_path: Optional path to save character definitions
        min_mentions: Minimum mentions to include character
        max_characters: Maximum number of characters to extract

    Returns:
        Dict of extracted characters
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '')
    if not text:
        raise ValueError(f"No text found in {input_path}")

    extractor = DynamicCharacterExtractor(
        min_mentions=min_mentions,
        max_characters=max_characters,
    )

    characters = extractor.extract(text)

    console.print(f"\n[bold]Character Extraction: {data.get('title', input_path.stem)}[/bold]")
    console.print(f"Language detected: {extractor.language}")
    console.print(f"Characters found: {len(characters)}")

    extractor.print_summary(characters)

    if output_path:
        output_data = {
            'source': str(input_path),
            'title': data.get('title', ''),
            'language': extractor.language,
            'characters': {
                name: char.to_dict()
                for name, char in characters.items()
            },
            'visualization_format': extractor.to_visualization_format(characters),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        console.print(f"\n[green]Saved to {output_path}[/green]")

    return characters


# CLI interface
if __name__ == '__main__':
    import click

    @click.command()
    @click.option('--input', '-i', 'input_path', required=True, type=click.Path(exists=True))
    @click.option('--output', '-o', 'output_path', type=click.Path())
    @click.option('--min-mentions', '-m', default=5, help='Minimum mentions to include')
    @click.option('--max-characters', '-n', default=30, help='Maximum characters to extract')
    def main(input_path: str, output_path: str, min_mentions: int, max_characters: int):
        """Extract character definitions from a text file."""
        extract_characters_from_file(
            Path(input_path),
            Path(output_path) if output_path else None,
            min_mentions,
            max_characters,
        )

    main()
