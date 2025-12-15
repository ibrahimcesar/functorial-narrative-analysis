#!/usr/bin/env python3
"""
Character Orbit Visualization

Creates orbital diagrams showing how events and other characters
cluster around major characters in complex novels like War and Peace.

Each major character becomes a "gravitational center" with:
- Events that involve them orbiting at different distances based on frequency
- Other characters orbiting based on interaction strength
- Visual clusters showing narrative "solar systems"
"""

import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, ConnectionPatch
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
from rich.console import Console

console = Console()


@dataclass
class CharacterOrbit:
    """Represents a character's orbital system."""
    name: str
    total_mentions: int
    events: Dict[str, int]  # event_type -> count
    satellites: Dict[str, float]  # other_character -> interaction_strength
    scenes: List[int]  # window indices where character appears


# Character definitions for War and Peace (Russian)
WAR_AND_PEACE_CHARACTERS_RU = {
    # Rostov family
    'Наташа': {'family': 'Ростовы', 'role': 'protagonist', 'color': '#FF6B6B'},
    'Николай': {'family': 'Ростовы', 'role': 'major', 'color': '#FF8E8E'},
    'Петя': {'family': 'Ростовы', 'role': 'supporting', 'color': '#FFB4B4'},
    'Соня': {'family': 'Ростовы', 'role': 'supporting', 'color': '#FFC4C4'},
    'Ростов': {'family': 'Ростовы', 'role': 'family_name', 'color': '#FF9999'},
    'Ростова': {'family': 'Ростовы', 'role': 'family_name', 'color': '#FF9999'},

    # Bolkonsky family
    'Андрей': {'family': 'Болконские', 'role': 'protagonist', 'color': '#4ECDC4'},
    'Болконский': {'family': 'Болконские', 'role': 'family_name', 'color': '#7EDCD6'},
    'Марья': {'family': 'Болконские', 'role': 'major', 'color': '#9EE6E1'},
    'Княжна': {'family': 'Болконские', 'role': 'title', 'color': '#BEF0EC'},

    # Bezukhov/Pierre
    'Пьер': {'family': 'Безуховы', 'role': 'protagonist', 'color': '#45B7D1'},
    'Безухов': {'family': 'Безуховы', 'role': 'family_name', 'color': '#6DC7DD'},

    # Kuragin family
    'Элен': {'family': 'Курагины', 'role': 'antagonist', 'color': '#9B59B6'},
    'Анатоль': {'family': 'Курагины', 'role': 'antagonist', 'color': '#AF7AC5'},
    'Курагин': {'family': 'Курагины', 'role': 'family_name', 'color': '#C39BD3'},

    # Historical figures
    'Наполеон': {'family': 'historical', 'role': 'historical', 'color': '#E74C3C'},
    'Кутузов': {'family': 'historical', 'role': 'historical', 'color': '#27AE60'},
    'Багратион': {'family': 'historical', 'role': 'historical', 'color': '#2ECC71'},

    # Other important characters
    'Долохов': {'family': 'other', 'role': 'antagonist', 'color': '#8E44AD'},
    'Денисов': {'family': 'other', 'role': 'supporting', 'color': '#F39C12'},
    'Каратаев': {'family': 'other', 'role': 'symbolic', 'color': '#1ABC9C'},
}

# Anna Karenina characters (Russian)
ANNA_KARENINA_CHARACTERS_RU = {
    'Анна': {'family': 'Каренины', 'role': 'protagonist', 'color': '#E74C3C'},
    'Каренин': {'family': 'Каренины', 'role': 'major', 'color': '#C0392B'},
    'Каренина': {'family': 'Каренины', 'role': 'family_name', 'color': '#E74C3C'},
    'Вронский': {'family': 'Вронские', 'role': 'protagonist', 'color': '#9B59B6'},
    'Левин': {'family': 'Левины', 'role': 'protagonist', 'color': '#27AE60'},
    'Кити': {'family': 'Щербацкие', 'role': 'protagonist', 'color': '#F1C40F'},
    'Китти': {'family': 'Щербацкие', 'role': 'protagonist', 'color': '#F1C40F'},
    'Стива': {'family': 'Облонские', 'role': 'major', 'color': '#E67E22'},
    'Облонский': {'family': 'Облонские', 'role': 'family_name', 'color': '#E67E22'},
    'Долли': {'family': 'Облонские', 'role': 'major', 'color': '#F39C12'},
    'Сережа': {'family': 'Каренины', 'role': 'supporting', 'color': '#E57373'},
    'Варенька': {'family': 'other', 'role': 'supporting', 'color': '#81C784'},
    'Николай': {'family': 'Левины', 'role': 'supporting', 'color': '#2ECC71'},
}

# ============================================================================
# ENGLISH CHARACTER DEFINITIONS
# ============================================================================

# War and Peace characters (English)
WAR_AND_PEACE_CHARACTERS_EN = {
    # Rostov family
    'Natasha': {'family': 'Rostov', 'role': 'protagonist', 'color': '#FF6B6B'},
    'Nicholas': {'family': 'Rostov', 'role': 'major', 'color': '#FF8E8E'},
    'Nikolai': {'family': 'Rostov', 'role': 'major', 'color': '#FF8E8E'},
    'Petya': {'family': 'Rostov', 'role': 'supporting', 'color': '#FFB4B4'},
    'Sonya': {'family': 'Rostov', 'role': 'supporting', 'color': '#FFC4C4'},
    'Rostov': {'family': 'Rostov', 'role': 'family_name', 'color': '#FF9999'},
    'Countess': {'family': 'Rostov', 'role': 'supporting', 'color': '#FF9999'},

    # Bolkonsky family
    'Andrew': {'family': 'Bolkonsky', 'role': 'protagonist', 'color': '#4ECDC4'},
    'Andrei': {'family': 'Bolkonsky', 'role': 'protagonist', 'color': '#4ECDC4'},
    'Bolkonsky': {'family': 'Bolkonsky', 'role': 'family_name', 'color': '#7EDCD6'},
    'Mary': {'family': 'Bolkonsky', 'role': 'major', 'color': '#9EE6E1'},
    'Princess': {'family': 'Bolkonsky', 'role': 'title', 'color': '#BEF0EC'},

    # Bezukhov/Pierre
    'Pierre': {'family': 'Bezukhov', 'role': 'protagonist', 'color': '#45B7D1'},
    'Bezukhov': {'family': 'Bezukhov', 'role': 'family_name', 'color': '#6DC7DD'},

    # Kuragin family
    'Helene': {'family': 'Kuragin', 'role': 'antagonist', 'color': '#9B59B6'},
    'Ellen': {'family': 'Kuragin', 'role': 'antagonist', 'color': '#9B59B6'},
    'Anatole': {'family': 'Kuragin', 'role': 'antagonist', 'color': '#AF7AC5'},
    'Kuragin': {'family': 'Kuragin', 'role': 'family_name', 'color': '#C39BD3'},

    # Historical figures
    'Napoleon': {'family': 'historical', 'role': 'historical', 'color': '#E74C3C'},
    'Kutuzov': {'family': 'historical', 'role': 'historical', 'color': '#27AE60'},
    'Bagration': {'family': 'historical', 'role': 'historical', 'color': '#2ECC71'},

    # Other important characters
    'Dolokhov': {'family': 'other', 'role': 'antagonist', 'color': '#8E44AD'},
    'Denisov': {'family': 'other', 'role': 'supporting', 'color': '#F39C12'},
    'Karataev': {'family': 'other', 'role': 'symbolic', 'color': '#1ABC9C'},
    'Platon': {'family': 'other', 'role': 'symbolic', 'color': '#1ABC9C'},
}

# Anna Karenina characters (English)
ANNA_KARENINA_CHARACTERS_EN = {
    'Anna': {'family': 'Karenin', 'role': 'protagonist', 'color': '#E74C3C'},
    'Karenin': {'family': 'Karenin', 'role': 'major', 'color': '#C0392B'},
    'Alexei': {'family': 'Karenin', 'role': 'major', 'color': '#C0392B'},
    'Vronsky': {'family': 'Vronsky', 'role': 'protagonist', 'color': '#9B59B6'},
    'Levin': {'family': 'Levin', 'role': 'protagonist', 'color': '#27AE60'},
    'Konstantin': {'family': 'Levin', 'role': 'protagonist', 'color': '#27AE60'},
    'Kitty': {'family': 'Shcherbatsky', 'role': 'protagonist', 'color': '#F1C40F'},
    'Stiva': {'family': 'Oblonsky', 'role': 'major', 'color': '#E67E22'},
    'Stepan': {'family': 'Oblonsky', 'role': 'major', 'color': '#E67E22'},
    'Oblonsky': {'family': 'Oblonsky', 'role': 'family_name', 'color': '#E67E22'},
    'Dolly': {'family': 'Oblonsky', 'role': 'major', 'color': '#F39C12'},
    'Darya': {'family': 'Oblonsky', 'role': 'major', 'color': '#F39C12'},
    'Seryozha': {'family': 'Karenin', 'role': 'supporting', 'color': '#E57373'},
    'Varenka': {'family': 'other', 'role': 'supporting', 'color': '#81C784'},
    'Nikolai': {'family': 'Levin', 'role': 'supporting', 'color': '#2ECC71'},
    'Nicholas': {'family': 'Levin', 'role': 'supporting', 'color': '#2ECC71'},
}

# ============================================================================
# ADDITIONAL WORKS - ENGLISH CLASSICS
# ============================================================================

# Pride and Prejudice - Jane Austen
PRIDE_AND_PREJUDICE_CHARACTERS = {
    'Elizabeth': {'family': 'Bennet', 'role': 'protagonist', 'color': '#E91E63'},
    'Lizzy': {'family': 'Bennet', 'role': 'protagonist', 'color': '#E91E63'},
    'Darcy': {'family': 'Darcy', 'role': 'protagonist', 'color': '#3F51B5'},
    'Jane': {'family': 'Bennet', 'role': 'major', 'color': '#F48FB1'},
    'Bingley': {'family': 'Bingley', 'role': 'major', 'color': '#64B5F6'},
    'Wickham': {'family': 'other', 'role': 'antagonist', 'color': '#F44336'},
    'Lydia': {'family': 'Bennet', 'role': 'supporting', 'color': '#FF80AB'},
    'Collins': {'family': 'other', 'role': 'comic', 'color': '#9E9E9E'},
    'Bennet': {'family': 'Bennet', 'role': 'family_name', 'color': '#EC407A'},
    'Lady': {'family': 'De Bourgh', 'role': 'antagonist', 'color': '#7B1FA2'},
    'Charlotte': {'family': 'Lucas', 'role': 'supporting', 'color': '#81C784'},
    'Georgiana': {'family': 'Darcy', 'role': 'supporting', 'color': '#7986CB'},
    'Mary': {'family': 'Bennet', 'role': 'minor', 'color': '#CE93D8'},
    'Kitty': {'family': 'Bennet', 'role': 'minor', 'color': '#F8BBD9'},
}

# Moby Dick - Herman Melville
MOBY_DICK_CHARACTERS = {
    'Ishmael': {'family': 'narrator', 'role': 'protagonist', 'color': '#4FC3F7'},
    'Ahab': {'family': 'Pequod', 'role': 'protagonist', 'color': '#F44336'},
    'Queequeg': {'family': 'harpooners', 'role': 'major', 'color': '#8D6E63'},
    'Starbuck': {'family': 'Pequod', 'role': 'major', 'color': '#66BB6A'},
    'Stubb': {'family': 'Pequod', 'role': 'supporting', 'color': '#FFA726'},
    'Flask': {'family': 'Pequod', 'role': 'supporting', 'color': '#FFCA28'},
    'Tashtego': {'family': 'harpooners', 'role': 'supporting', 'color': '#A1887F'},
    'Daggoo': {'family': 'harpooners', 'role': 'supporting', 'color': '#795548'},
    'Fedallah': {'family': 'Ahab', 'role': 'mysterious', 'color': '#5C6BC0'},
    'Pip': {'family': 'Pequod', 'role': 'symbolic', 'color': '#26A69A'},
    'Whale': {'family': 'nature', 'role': 'antagonist', 'color': '#ECEFF1'},
    'Moby': {'family': 'nature', 'role': 'antagonist', 'color': '#ECEFF1'},
    'Dick': {'family': 'nature', 'role': 'antagonist', 'color': '#ECEFF1'},
}

# A Tale of Two Cities - Charles Dickens
TALE_TWO_CITIES_CHARACTERS = {
    'Sydney': {'family': 'Carton', 'role': 'protagonist', 'color': '#5C6BC0'},
    'Carton': {'family': 'Carton', 'role': 'protagonist', 'color': '#5C6BC0'},
    'Charles': {'family': 'Darnay', 'role': 'protagonist', 'color': '#42A5F5'},
    'Darnay': {'family': 'Darnay', 'role': 'protagonist', 'color': '#42A5F5'},
    'Lucie': {'family': 'Manette', 'role': 'protagonist', 'color': '#EC407A'},
    'Manette': {'family': 'Manette', 'role': 'major', 'color': '#AB47BC'},
    'Defarge': {'family': 'revolutionaries', 'role': 'antagonist', 'color': '#EF5350'},
    'Madame': {'family': 'revolutionaries', 'role': 'antagonist', 'color': '#C62828'},
    'Lorry': {'family': 'Tellson', 'role': 'supporting', 'color': '#78909C'},
    'Jerry': {'family': 'Cruncher', 'role': 'comic', 'color': '#8D6E63'},
    'Pross': {'family': 'servants', 'role': 'supporting', 'color': '#66BB6A'},
    'Marquis': {'family': 'Evremonde', 'role': 'antagonist', 'color': '#7B1FA2'},
    'Stryver': {'family': 'lawyers', 'role': 'minor', 'color': '#9E9E9E'},
}

# Alice in Wonderland - Lewis Carroll
ALICE_WONDERLAND_CHARACTERS = {
    'Alice': {'family': 'real_world', 'role': 'protagonist', 'color': '#64B5F6'},
    'Queen': {'family': 'cards', 'role': 'antagonist', 'color': '#E53935'},
    'King': {'family': 'cards', 'role': 'authority', 'color': '#FDD835'},
    'Hatter': {'family': 'tea_party', 'role': 'eccentric', 'color': '#AB47BC'},
    'March': {'family': 'tea_party', 'role': 'eccentric', 'color': '#8D6E63'},
    'Hare': {'family': 'tea_party', 'role': 'eccentric', 'color': '#8D6E63'},
    'Cheshire': {'family': 'wonderland', 'role': 'mysterious', 'color': '#FF80AB'},
    'Cat': {'family': 'wonderland', 'role': 'mysterious', 'color': '#FF80AB'},
    'Caterpillar': {'family': 'wonderland', 'role': 'wise', 'color': '#4DB6AC'},
    'Duchess': {'family': 'wonderland', 'role': 'eccentric', 'color': '#7E57C2'},
    'Rabbit': {'family': 'wonderland', 'role': 'guide', 'color': '#BDBDBD'},
    'White': {'family': 'wonderland', 'role': 'guide', 'color': '#BDBDBD'},
    'Dormouse': {'family': 'tea_party', 'role': 'minor', 'color': '#A1887F'},
    'Mock': {'family': 'sea', 'role': 'eccentric', 'color': '#26A69A'},
    'Turtle': {'family': 'sea', 'role': 'eccentric', 'color': '#26A69A'},
    'Gryphon': {'family': 'sea', 'role': 'eccentric', 'color': '#FFB74D'},
}

# Little Women - Louisa May Alcott
LITTLE_WOMEN_CHARACTERS = {
    'Jo': {'family': 'March', 'role': 'protagonist', 'color': '#5C6BC0'},
    'Meg': {'family': 'March', 'role': 'major', 'color': '#EC407A'},
    'Beth': {'family': 'March', 'role': 'major', 'color': '#66BB6A'},
    'Amy': {'family': 'March', 'role': 'major', 'color': '#AB47BC'},
    'Marmee': {'family': 'March', 'role': 'supporting', 'color': '#8D6E63'},
    'March': {'family': 'March', 'role': 'family_name', 'color': '#78909C'},
    'Laurie': {'family': 'Laurence', 'role': 'major', 'color': '#42A5F5'},
    'Laurence': {'family': 'Laurence', 'role': 'supporting', 'color': '#64B5F6'},
    'Professor': {'family': 'Bhaer', 'role': 'major', 'color': '#795548'},
    'Bhaer': {'family': 'Bhaer', 'role': 'major', 'color': '#795548'},
    'John': {'family': 'Brooke', 'role': 'supporting', 'color': '#26A69A'},
    'Brooke': {'family': 'Brooke', 'role': 'supporting', 'color': '#26A69A'},
    'Hannah': {'family': 'servants', 'role': 'minor', 'color': '#BDBDBD'},
}

# The Phantom of the Opera - Gaston Leroux
PHANTOM_OPERA_CHARACTERS = {
    'Christine': {'family': 'Daae', 'role': 'protagonist', 'color': '#EC407A'},
    'Erik': {'family': 'Phantom', 'role': 'protagonist', 'color': '#212121'},
    'Phantom': {'family': 'Phantom', 'role': 'protagonist', 'color': '#212121'},
    'Raoul': {'family': 'Chagny', 'role': 'protagonist', 'color': '#42A5F5'},
    'Vicomte': {'family': 'Chagny', 'role': 'protagonist', 'color': '#42A5F5'},
    'Persian': {'family': 'mysterious', 'role': 'supporting', 'color': '#8D6E63'},
    'Carlotta': {'family': 'Opera', 'role': 'antagonist', 'color': '#EF5350'},
    'Managers': {'family': 'Opera', 'role': 'comic', 'color': '#9E9E9E'},
    'Madame': {'family': 'Giry', 'role': 'supporting', 'color': '#78909C'},
    'Meg': {'family': 'Giry', 'role': 'minor', 'color': '#90A4AE'},
}

# Sherlock Holmes (Adventures) - Arthur Conan Doyle
SHERLOCK_HOLMES_CHARACTERS = {
    'Holmes': {'family': 'detectives', 'role': 'protagonist', 'color': '#5C6BC0'},
    'Sherlock': {'family': 'detectives', 'role': 'protagonist', 'color': '#5C6BC0'},
    'Watson': {'family': 'detectives', 'role': 'protagonist', 'color': '#66BB6A'},
    'John': {'family': 'detectives', 'role': 'protagonist', 'color': '#66BB6A'},
    'Lestrade': {'family': 'police', 'role': 'supporting', 'color': '#78909C'},
    'Moriarty': {'family': 'villains', 'role': 'antagonist', 'color': '#E53935'},
    'Irene': {'family': 'clients', 'role': 'notable', 'color': '#AB47BC'},
    'Adler': {'family': 'clients', 'role': 'notable', 'color': '#AB47BC'},
    'Hudson': {'family': 'Baker_Street', 'role': 'supporting', 'color': '#8D6E63'},
    'Mycroft': {'family': 'Holmes', 'role': 'supporting', 'color': '#7986CB'},
}

# Generic/Auto-detect characters (fallback)
GENERIC_CHARACTERS = {
    # Common English titles that indicate character names follow
}

# Work detection mapping
WORK_CHARACTER_MAPS = {
    'pride': PRIDE_AND_PREJUDICE_CHARACTERS,
    'prejudice': PRIDE_AND_PREJUDICE_CHARACTERS,
    'bennet': PRIDE_AND_PREJUDICE_CHARACTERS,
    'moby': MOBY_DICK_CHARACTERS,
    'whale': MOBY_DICK_CHARACTERS,
    'ahab': MOBY_DICK_CHARACTERS,
    'two_cities': TALE_TWO_CITIES_CHARACTERS,
    'tale': TALE_TWO_CITIES_CHARACTERS,
    'alice': ALICE_WONDERLAND_CHARACTERS,
    'wonderland': ALICE_WONDERLAND_CHARACTERS,
    'little_women': LITTLE_WOMEN_CHARACTERS,
    'march': LITTLE_WOMEN_CHARACTERS,
    'phantom': PHANTOM_OPERA_CHARACTERS,
    'opera': PHANTOM_OPERA_CHARACTERS,
    'sherlock': SHERLOCK_HOLMES_CHARACTERS,
    'holmes': SHERLOCK_HOLMES_CHARACTERS,
    'watson': SHERLOCK_HOLMES_CHARACTERS,
}

# Event types with Russian keywords
EVENT_PATTERNS_RU = {
    'war': ['война', 'сражение', 'бой', 'битва', 'атака', 'армия', 'полк', 'солдат',
            'офицер', 'генерал', 'победа', 'поражение', 'враг', 'неприятель'],
    'love': ['любовь', 'любил', 'любит', 'любила', 'влюблен', 'страсть', 'поцелуй',
             'объятия', 'сердце', 'чувство', 'нежность', 'счастье'],
    'death': ['смерть', 'умер', 'умерла', 'погиб', 'убит', 'мертв', 'похороны',
              'могила', 'конец', 'гибель', 'застрелил'],
    'marriage': ['свадьба', 'венчание', 'жених', 'невеста', 'муж', 'жена',
                 'обручение', 'помолвка', 'брак'],
    'ball': ['бал', 'танец', 'танцевал', 'вальс', 'мазурка', 'приглашение'],
    'duel': ['дуэль', 'поединок', 'секундант', 'пистолет', 'выстрел'],
    'journey': ['поехал', 'уехал', 'приехал', 'дорога', 'путешествие', 'карета',
                'лошади', 'москва', 'петербург', 'деревня'],
    'society': ['общество', 'салон', 'гости', 'визит', 'прием', 'светский'],
    'philosophy': ['бог', 'душа', 'истина', 'смысл', 'жизнь', 'вера', 'думал',
                   'размышлял', 'понял', 'масон', 'франкмасон'],
    'conflict': ['ссора', 'спор', 'гнев', 'злость', 'обида', 'ревность', 'измена'],
}

# Event types with English keywords
EVENT_PATTERNS_EN = {
    'war': ['war', 'battle', 'fight', 'attack', 'army', 'regiment', 'soldier',
            'officer', 'general', 'victory', 'defeat', 'enemy', 'troops', 'cannon',
            'wounded', 'killed', 'cavalry', 'infantry', 'commander', 'campaign'],
    'love': ['love', 'loved', 'passion', 'kiss', 'embrace', 'heart', 'feeling',
             'tenderness', 'happiness', 'affection', 'desire', 'beloved'],
    'death': ['death', 'died', 'dead', 'killed', 'funeral', 'grave', 'dying',
              'corpse', 'body', 'mourning', 'perished'],
    'marriage': ['wedding', 'marriage', 'bride', 'groom', 'husband', 'wife',
                 'engagement', 'betrothal', 'married', 'marry'],
    'ball': ['ball', 'dance', 'danced', 'waltz', 'mazurka', 'cotillion', 'music'],
    'duel': ['duel', 'seconds', 'pistol', 'shot', 'challenge', 'honor'],
    'journey': ['traveled', 'journey', 'road', 'carriage', 'horses', 'Moscow',
                'Petersburg', 'village', 'estate', 'country', 'arrived', 'departed'],
    'society': ['society', 'salon', 'guests', 'visit', 'reception', 'drawing-room',
                'party', 'dinner', 'conversation', 'fashionable'],
    'philosophy': ['God', 'soul', 'truth', 'meaning', 'life', 'faith', 'thought',
                   'understood', 'realized', 'mason', 'freemason', 'belief', 'spirit'],
    'conflict': ['quarrel', 'argument', 'anger', 'rage', 'offense', 'jealousy',
                 'betrayal', 'dispute', 'angry', 'jealous'],
}


def create_windows_preserve_case(text: str, window_size: int = 1000, overlap: int = 500) -> List[str]:
    """Create overlapping windows preserving original case."""
    words = text.split()
    step = window_size - overlap
    windows = []
    for i in range(0, len(words), step):
        window = ' '.join(words[i:i + window_size])
        if len(window.split()) >= window_size // 2:
            windows.append(window)
    return windows if windows else [text]


def extract_character_mentions(text: str, characters: Dict[str, Any]) -> Dict[str, int]:
    """Count character mentions in text."""
    mentions = defaultdict(int)
    for char_name in characters:
        # Case-insensitive count
        pattern = rf'\b{re.escape(char_name)}\b'
        count = len(re.findall(pattern, text, re.IGNORECASE))
        if count > 0:
            mentions[char_name] = count
    return dict(mentions)


def extract_events(text: str, language: str = 'ru') -> Dict[str, int]:
    """Extract event counts from text."""
    text_lower = text.lower()
    events = defaultdict(int)
    patterns = EVENT_PATTERNS_RU if language == 'ru' else EVENT_PATTERNS_EN
    for event_type, keywords in patterns.items():
        for keyword in keywords:
            count = text_lower.count(keyword.lower())
            events[event_type] += count
    return dict(events)


def build_character_orbits(
    text: str,
    characters: Dict[str, Any],
    window_size: int = 1000,
    language: str = 'ru'
) -> Dict[str, CharacterOrbit]:
    """Build orbital data for each character."""
    windows = create_windows_preserve_case(text, window_size, window_size // 2)

    # Track per-character data
    char_events = defaultdict(lambda: defaultdict(int))
    char_interactions = defaultdict(lambda: defaultdict(float))
    char_scenes = defaultdict(list)
    char_total = defaultdict(int)

    console.print(f"[cyan]Analyzing {len(windows)} windows for character orbits...[/cyan]")

    for idx, window in enumerate(windows):
        # Find characters in this window
        mentions = extract_character_mentions(window, characters)
        events = extract_events(window, language)

        chars_in_window = [c for c, count in mentions.items() if count > 0]

        for char in chars_in_window:
            char_total[char] += mentions[char]
            char_scenes[char].append(idx)

            # Associate events with character
            for event_type, count in events.items():
                if count > 0:
                    char_events[char][event_type] += count

            # Track character co-occurrences (interactions)
            for other_char in chars_in_window:
                if char != other_char:
                    # Weight by geometric mean of mentions
                    weight = math.sqrt(mentions[char] * mentions[other_char])
                    char_interactions[char][other_char] += weight

    # Build orbit objects
    orbits = {}
    for char_name, char_info in characters.items():
        if char_total[char_name] > 0:
            orbits[char_name] = CharacterOrbit(
                name=char_name,
                total_mentions=char_total[char_name],
                events=dict(char_events[char_name]),
                satellites=dict(char_interactions[char_name]),
                scenes=char_scenes[char_name]
            )

    return orbits


def visualize_single_orbit(
    ax: plt.Axes,
    orbit: CharacterOrbit,
    characters: Dict[str, Any],
    center: Tuple[float, float],
    max_radius: float
):
    """Draw a single character's orbital system."""
    cx, cy = center
    char_info = characters.get(orbit.name, {'color': '#888888'})
    main_color = char_info.get('color', '#888888')

    # Draw central character (size based on mentions)
    size = min(2000, max(500, orbit.total_mentions * 2))
    ax.scatter([cx], [cy], s=size, c=[main_color], alpha=0.9, zorder=10,
               edgecolors='white', linewidths=2)
    ax.annotate(orbit.name, (cx, cy), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=11)

    # Sort satellites by interaction strength
    sorted_satellites = sorted(orbit.satellites.items(), key=lambda x: -x[1])[:8]

    if sorted_satellites:
        max_interaction = sorted_satellites[0][1] if sorted_satellites else 1

        # Draw satellite characters in orbit
        for i, (sat_name, strength) in enumerate(sorted_satellites):
            # Distance from center inversely related to strength
            normalized_strength = strength / max_interaction
            distance = max_radius * (0.3 + 0.7 * (1 - normalized_strength))

            # Angle around the character
            angle = (2 * math.pi * i) / len(sorted_satellites)
            sx = cx + distance * math.cos(angle)
            sy = cy + distance * math.sin(angle)

            # Satellite size based on interaction strength
            sat_size = max(100, min(800, strength * 3))
            sat_color = characters.get(sat_name, {}).get('color', '#AAAAAA')

            # Draw orbit path (ellipse)
            orbit_circle = Circle((cx, cy), distance, fill=False,
                                   color=main_color, alpha=0.2, linestyle='--')
            ax.add_patch(orbit_circle)

            # Draw connection line
            ax.plot([cx, sx], [cy, sy], color=main_color, alpha=0.3, linewidth=1)

            # Draw satellite
            ax.scatter([sx], [sy], s=sat_size, c=[sat_color], alpha=0.7, zorder=5,
                       edgecolors='white', linewidths=1)
            ax.annotate(sat_name, (sx, sy), ha='center', va='bottom',
                        fontsize=7, color='#333333', zorder=6)

    # Draw event indicators around the character
    if orbit.events:
        event_colors = {
            'war': '#E74C3C', 'love': '#E91E63', 'death': '#212121',
            'marriage': '#FFC107', 'ball': '#9C27B0', 'duel': '#FF5722',
            'journey': '#03A9F4', 'society': '#8BC34A', 'philosophy': '#607D8B',
            'conflict': '#F44336'
        }

        sorted_events = sorted(orbit.events.items(), key=lambda x: -x[1])[:5]
        event_radius = max_radius * 0.15

        for i, (event_type, count) in enumerate(sorted_events):
            angle = math.pi + (math.pi * i) / max(1, len(sorted_events) - 1)
            ex = cx + event_radius * math.cos(angle)
            ey = cy + event_radius * math.sin(angle)

            event_size = max(30, min(200, count))
            ax.scatter([ex], [ey], s=event_size,
                       c=[event_colors.get(event_type, '#888888')],
                       alpha=0.6, marker='s', zorder=4)


def visualize_all_orbits(
    orbits: Dict[str, CharacterOrbit],
    characters: Dict[str, Any],
    title: str,
    output_path: Path
):
    """Create the main orbital visualization with all major characters."""
    # Filter to top characters by mentions
    sorted_chars = sorted(orbits.items(), key=lambda x: -x[1].total_mentions)
    top_chars = sorted_chars[:12]  # Top 12 characters

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#1a1a2e')

    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')

    # Calculate grid layout
    n_chars = len(top_chars)
    cols = 4
    rows = (n_chars + cols - 1) // cols

    # Grid spacing
    x_spacing = 1.0 / (cols + 1)
    y_spacing = 1.0 / (rows + 1)
    max_radius = min(x_spacing, y_spacing) * 0.4

    # Draw each character's orbit
    for i, (char_name, orbit) in enumerate(top_chars):
        row = i // cols
        col = i % cols

        cx = (col + 1) * x_spacing
        cy = 1 - (row + 1) * y_spacing

        visualize_single_orbit(ax, orbit, characters, (cx, cy), max_radius)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.set_title(f'Character Orbital Systems: {title}\n'
                 f'Characters orbit around those they interact with most',
                 fontsize=16, color='white', pad=20)

    # Legend for events
    event_colors = {
        'war': '#E74C3C', 'love': '#E91E63', 'death': '#212121',
        'marriage': '#FFC107', 'ball': '#9C27B0', 'duel': '#FF5722',
        'journey': '#03A9F4', 'society': '#8BC34A', 'philosophy': '#607D8B',
        'conflict': '#F44336'
    }

    legend_y = 0.02
    legend_x = 0.05
    for i, (event, color) in enumerate(event_colors.items()):
        ax.scatter([legend_x + i * 0.09], [legend_y], s=50, c=[color],
                   marker='s', alpha=0.8)
        ax.annotate(event, (legend_x + i * 0.09, legend_y - 0.02),
                    ha='center', fontsize=7, color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()

    console.print(f"[green]✓ Saved orbital visualization: {output_path}[/green]")


def visualize_interaction_network(
    orbits: Dict[str, CharacterOrbit],
    characters: Dict[str, Any],
    title: str,
    output_path: Path
):
    """Create a network-style visualization showing character clusters."""
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Get top characters
    sorted_chars = sorted(orbits.items(), key=lambda x: -x[1].total_mentions)
    top_chars = dict(sorted_chars[:20])

    # Position characters using force-directed-like layout
    # Group by family
    families = defaultdict(list)
    for char_name, orbit in top_chars.items():
        family = characters.get(char_name, {}).get('family', 'other')
        families[family].append(char_name)

    # Assign positions - families in clusters
    positions = {}
    family_centers = {}
    n_families = len(families)

    for i, (family, members) in enumerate(families.items()):
        # Family center on a circle
        angle = (2 * math.pi * i) / n_families
        fcx = 0.5 + 0.3 * math.cos(angle)
        fcy = 0.5 + 0.3 * math.sin(angle)
        family_centers[family] = (fcx, fcy)

        # Members around family center
        for j, member in enumerate(members):
            member_angle = (2 * math.pi * j) / max(1, len(members))
            radius = 0.08 + 0.04 * (j % 2)
            mx = fcx + radius * math.cos(member_angle)
            my = fcy + radius * math.sin(member_angle)
            positions[member] = (mx, my)

    # Draw connections first (behind nodes)
    max_strength = 1
    for char_name, orbit in top_chars.items():
        for other, strength in orbit.satellites.items():
            if other in positions and strength > max_strength * 0.1:
                max_strength = max(max_strength, strength)

    for char_name, orbit in top_chars.items():
        if char_name not in positions:
            continue
        x1, y1 = positions[char_name]

        for other, strength in orbit.satellites.items():
            if other not in positions:
                continue
            if strength < max_strength * 0.05:  # Skip weak connections
                continue

            x2, y2 = positions[other]

            # Line thickness and alpha based on strength
            alpha = min(0.8, 0.1 + 0.7 * (strength / max_strength))
            linewidth = 0.5 + 4 * (strength / max_strength)

            ax.plot([x1, x2], [y1, y2],
                    color='#58a6ff', alpha=alpha, linewidth=linewidth, zorder=1)

    # Draw nodes
    for char_name, orbit in top_chars.items():
        if char_name not in positions:
            continue
        x, y = positions[char_name]

        char_info = characters.get(char_name, {'color': '#888888'})
        color = char_info.get('color', '#888888')

        # Size based on mentions
        size = min(3000, max(300, orbit.total_mentions * 3))

        ax.scatter([x], [y], s=size, c=[color], alpha=0.9, zorder=10,
                   edgecolors='white', linewidths=2)

        # Label
        ax.annotate(char_name, (x, y + 0.04), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='white', zorder=11)

        # Show mention count
        ax.annotate(f'({orbit.total_mentions})', (x, y - 0.03), ha='center', va='top',
                    fontsize=7, color='#888888', zorder=11)

    # Draw family labels
    for family, (fcx, fcy) in family_centers.items():
        ax.annotate(family, (fcx, fcy + 0.15), ha='center', va='bottom',
                    fontsize=11, color='#666666', style='italic', zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(f'Character Interaction Network: {title}\n'
                 f'Node size = mentions, Line thickness = interaction strength',
                 fontsize=14, color='white', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()

    console.print(f"[green]✓ Saved network visualization: {output_path}[/green]")


def visualize_event_clusters(
    orbits: Dict[str, CharacterOrbit],
    characters: Dict[str, Any],
    title: str,
    output_path: Path
):
    """Show which events cluster around which characters."""
    # Get top characters
    sorted_chars = sorted(orbits.items(), key=lambda x: -x[1].total_mentions)
    top_chars = sorted_chars[:10]

    # Build event matrix
    event_types = list(EVENT_PATTERNS_RU.keys())
    char_names = [c[0] for c in top_chars]

    matrix = np.zeros((len(char_names), len(event_types)))
    for i, (char_name, orbit) in enumerate(top_chars):
        for j, event_type in enumerate(event_types):
            matrix[i, j] = orbit.events.get(event_type, 0)

    # Normalize by row (character)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(matrix_norm, cmap='YlOrRd', aspect='auto')

    ax1.set_xticks(range(len(event_types)))
    ax1.set_xticklabels(event_types, rotation=45, ha='right')
    ax1.set_yticks(range(len(char_names)))
    ax1.set_yticklabels(char_names)

    # Add colorbar
    plt.colorbar(im, ax=ax1, label='Event proportion')

    ax1.set_title('Events by Character (normalized)', fontsize=12)
    ax1.set_xlabel('Event Type')
    ax1.set_ylabel('Character')

    # Stacked bar chart
    ax2 = axes[1]
    event_colors = plt.cm.tab10(np.linspace(0, 1, len(event_types)))

    bottom = np.zeros(len(char_names))
    for j, event_type in enumerate(event_types):
        values = matrix_norm[:, j]
        ax2.barh(char_names, values, left=bottom, label=event_type,
                 color=event_colors[j], alpha=0.8)
        bottom += values

    ax2.set_xlabel('Proportion of Events')
    ax2.set_title('Event Distribution per Character', fontsize=12)
    ax2.legend(loc='lower right', fontsize=8)
    ax2.set_xlim(0, 1)

    fig.suptitle(f'Character-Event Clusters: {title}', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved event cluster visualization: {output_path}[/green]")


def calculate_connectivity_scores(
    orbits: Dict[str, CharacterOrbit]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate connectivity scores for each character.

    Returns a dict with multiple connectivity metrics:
    - degree: Number of unique characters they interact with
    - strength: Total interaction weight (sum of all satellite weights)
    - weighted_degree: Strength normalized by degree (average connection strength)
    - centrality: How much they connect other characters (betweenness-like)
    - reach: Mentions × connections (narrative presence × social network)
    """
    scores = {}

    # First pass: calculate basic metrics
    all_interactions = defaultdict(float)
    for char_name, orbit in orbits.items():
        degree = len(orbit.satellites)
        strength = sum(orbit.satellites.values())
        weighted_degree = strength / max(1, degree)

        scores[char_name] = {
            'mentions': orbit.total_mentions,
            'degree': degree,
            'strength': strength,
            'weighted_degree': weighted_degree,
            'reach': orbit.total_mentions * degree,
        }

        # Track all interactions for centrality
        for other, weight in orbit.satellites.items():
            pair = tuple(sorted([char_name, other]))
            all_interactions[pair] = max(all_interactions[pair], weight)

    # Second pass: calculate centrality (simplified betweenness)
    # A character is central if they connect characters who don't otherwise connect
    for char_name, orbit in orbits.items():
        centrality = 0
        satellites = list(orbit.satellites.keys())

        # Check how many of this character's satellites don't connect to each other
        for i, sat1 in enumerate(satellites):
            for sat2 in satellites[i+1:]:
                # If sat1 and sat2 don't have a strong direct connection,
                # this character serves as a bridge
                if sat1 in orbits and sat2 in orbits.get(sat1, CharacterOrbit('', 0, {}, {}, [])).satellites:
                    direct_strength = orbits[sat1].satellites.get(sat2, 0)
                else:
                    direct_strength = 0

                # If their direct connection is weak, this char is a bridge
                bridge_strength = min(orbit.satellites.get(sat1, 0), orbit.satellites.get(sat2, 0))
                if bridge_strength > direct_strength:
                    centrality += (bridge_strength - direct_strength)

        scores[char_name]['centrality'] = centrality

    # Normalize all scores to 0-100 scale for easy comparison
    for metric in ['mentions', 'degree', 'strength', 'weighted_degree', 'reach', 'centrality']:
        max_val = max((s[metric] for s in scores.values()), default=1)
        if max_val > 0:
            for char_name in scores:
                scores[char_name][f'{metric}_normalized'] = (scores[char_name][metric] / max_val) * 100

    # Calculate composite "connectivity index" (weighted combination)
    for char_name in scores:
        s = scores[char_name]
        # Composite score: balance between mentions, connections, and centrality
        composite = (
            0.25 * s.get('mentions_normalized', 0) +
            0.20 * s.get('degree_normalized', 0) +
            0.25 * s.get('strength_normalized', 0) +
            0.15 * s.get('weighted_degree_normalized', 0) +
            0.15 * s.get('centrality_normalized', 0)
        )
        scores[char_name]['connectivity_index'] = composite

    return scores


def print_connectivity_report(
    scores: Dict[str, Dict[str, float]],
    title: str,
    top_n: int = 10
):
    """Print a formatted connectivity report."""
    from rich.table import Table

    # Sort by connectivity index
    sorted_chars = sorted(scores.items(), key=lambda x: -x[1]['connectivity_index'])[:top_n]

    console.print(f"\n[bold cyan]═══ CONNECTIVITY ANALYSIS: {title} ═══[/bold cyan]\n")

    table = Table(title="Character Connectivity Rankings")
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Character", style="cyan", width=15)
    table.add_column("Index", style="bold green", width=8)
    table.add_column("Mentions", width=10)
    table.add_column("Connections", width=12)
    table.add_column("Strength", width=10)
    table.add_column("Centrality", width=10)

    for i, (char_name, s) in enumerate(sorted_chars, 1):
        table.add_row(
            str(i),
            char_name,
            f"{s['connectivity_index']:.1f}",
            str(s['mentions']),
            str(s['degree']),
            f"{s['strength']:.0f}",
            f"{s['centrality']:.0f}"
        )

    console.print(table)

    # Print the winner
    if sorted_chars:
        winner = sorted_chars[0]
        console.print(f"\n[bold green]🏆 Most Connected Character: {winner[0]}[/bold green]")
        console.print(f"   Connectivity Index: {winner[1]['connectivity_index']:.1f}/100")
        console.print(f"   This character has {winner[1]['degree']} connections with total strength {winner[1]['strength']:.0f}")

    return sorted_chars


def visualize_connectivity_ranking(
    scores: Dict[str, Dict[str, float]],
    characters: Dict[str, Any],
    title: str,
    output_path: Path,
    top_n: int = 12
):
    """Create a visual ranking of character connectivity."""
    sorted_chars = sorted(scores.items(), key=lambda x: -x[1]['connectivity_index'])[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Character Connectivity Analysis: {title}', fontsize=14, fontweight='bold')

    # Left: Bar chart of connectivity index
    ax1 = axes[0]
    names = [c[0] for c in sorted_chars]
    indices = [c[1]['connectivity_index'] for c in sorted_chars]
    colors = [characters.get(name, {}).get('color', '#888888') for name in names]

    bars = ax1.barh(names[::-1], indices[::-1], color=colors[::-1], alpha=0.8)
    ax1.set_xlabel('Connectivity Index (0-100)')
    ax1.set_title('Overall Connectivity Ranking')
    ax1.set_xlim(0, 105)

    # Add value labels
    for bar, val in zip(bars, indices[::-1]):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                va='center', fontsize=9)

    # Right: Radar/spider chart of metrics for top 5
    ax2 = axes[1]

    # Metrics to show
    metrics = ['mentions_normalized', 'degree_normalized', 'strength_normalized',
               'weighted_degree_normalized', 'centrality_normalized']
    metric_labels = ['Mentions', 'Connections', 'Total\nStrength', 'Avg\nStrength', 'Centrality']

    # Number of variables
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    ax2 = plt.subplot(122, polar=True)

    for char_name, s in sorted_chars[:5]:
        values = [s.get(m, 0) for m in metrics]
        values += values[:1]  # Complete the loop
        color = characters.get(char_name, {}).get('color', '#888888')
        ax2.plot(angles, values, 'o-', linewidth=2, label=char_name, color=color)
        ax2.fill(angles, values, alpha=0.1, color=color)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_labels, size=9)
    ax2.set_ylim(0, 100)
    ax2.set_title('Top 5 Characters - Metric Breakdown', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ Saved connectivity ranking: {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description='Character Orbit Visualization')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--work', '-w', default='auto',
                        choices=['war_and_peace', 'anna_karenina', 'auto'],
                        help='Which work (for character definitions)')
    parser.add_argument('--language', '-l', default='auto',
                        choices=['ru', 'en', 'auto'],
                        help='Language of the text (ru, en, or auto-detect)')
    parser.add_argument('--window-size', type=int, default=1000, help='Window size')

    args = parser.parse_args()

    # Load text
    input_path = Path(args.input)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    text = data.get('text', '')
    title = data.get('title', input_path.stem)

    # Determine language
    language = args.language
    if language == 'auto':
        # Auto-detect: check for Cyrillic characters
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', text[:5000]))
        language = 'ru' if cyrillic_count > 100 else 'en'

    # Determine which character set to use
    work = args.work
    if work == 'auto':
        # Detect based on filename or title
        name_lower = (title + input_path.stem).lower()
        if 'война' in name_lower or 'war' in name_lower or 'peace' in name_lower:
            work = 'war_and_peace'
        else:
            work = 'anna_karenina'

    # Select character definitions based on work and language
    if work == 'war_and_peace':
        characters = WAR_AND_PEACE_CHARACTERS_RU if language == 'ru' else WAR_AND_PEACE_CHARACTERS_EN
    else:
        characters = ANNA_KARENINA_CHARACTERS_RU if language == 'ru' else ANNA_KARENINA_CHARACTERS_EN

    console.print(f"\n{'='*70}")
    console.print(f"[bold]CHARACTER ORBIT ANALYSIS[/bold]")
    console.print(f"Title: {title}")
    console.print(f"Work type: {work}")
    console.print(f"Language: {language.upper()}")
    console.print(f"{'='*70}")

    # Build orbits
    orbits = build_character_orbits(text, characters, args.window_size, language)

    console.print(f"[green]Found {len(orbits)} characters with mentions[/green]")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = input_path.stem

    # Generate visualizations
    visualize_all_orbits(
        orbits, characters, title,
        output_dir / f'{base_name}_character_orbits.png'
    )

    visualize_interaction_network(
        orbits, characters, title,
        output_dir / f'{base_name}_interaction_network.png'
    )

    visualize_event_clusters(
        orbits, characters, title,
        output_dir / f'{base_name}_event_clusters.png'
    )

    # Calculate and display connectivity scores
    scores = calculate_connectivity_scores(orbits)
    print_connectivity_report(scores, title)

    # Generate connectivity visualization
    visualize_connectivity_ranking(
        scores, characters, title,
        output_dir / f'{base_name}_connectivity_ranking.png'
    )

    console.print(f"\n{'='*70}")
    console.print(f"[green]✓ All visualizations saved to {output_dir}[/green]")


if __name__ == '__main__':
    main()
