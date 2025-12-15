"""
Semantic/Plot Tracking Functors for Functorial Narrative Analysis

These functors go beyond sentiment to track:
1. Character Interactions - Who interacts with whom, relationship dynamics
2. Narrative Events - Actions, state changes, plot points
3. Plot State - Key facts, revelations, conflicts, resolutions

This enables comparison of "what happens" across translations, not just
"how it feels" - addressing whether plot facts are preserved even when
emotional arcs diverge.

Theoretical Framework:
- F_interaction: Narr → Graph (character co-occurrence graph per window)
- F_event: Narr → EventSeq (sequence of narrative events)
- F_plot: Narr → StateSeq (sequence of plot states)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, NamedTuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .base import BaseFunctor, Trajectory


# =============================================================================
# Character Interaction Tracking
# =============================================================================

class InteractionType(Enum):
    """Types of character interactions."""
    CO_OCCURRENCE = "co_occurrence"      # Characters appear together
    DIALOGUE = "dialogue"                 # Speaking to each other
    ACTION = "action"                     # Physical interaction
    REFERENCE = "reference"               # One character mentions another
    EMOTIONAL = "emotional"               # Emotional connection


@dataclass
class CharacterInteraction:
    """Represents an interaction between characters."""
    char1: str
    char2: str
    interaction_type: InteractionType
    strength: float = 1.0
    context: str = ""

    def __hash__(self):
        return hash((frozenset([self.char1, self.char2]), self.interaction_type))

    def key(self) -> Tuple[str, str]:
        """Return sorted character pair for consistent keying."""
        return tuple(sorted([self.char1, self.char2]))


class CharacterInteractionExtractor:
    """
    Extract character interactions from narrative text.

    Detects:
    - Co-occurrence within sentences/paragraphs
    - Dialogue exchanges
    - Action verbs connecting characters
    - Pronouns resolved to nearby character names
    """

    # English dialogue verbs
    DIALOGUE_VERBS_EN = {
        'said', 'asked', 'replied', 'answered', 'whispered', 'shouted',
        'exclaimed', 'muttered', 'cried', 'called', 'told', 'spoke',
        'demanded', 'declared', 'explained', 'suggested', 'promised',
        'warned', 'pleaded', 'begged', 'questioned', 'remarked',
    }

    # English action verbs indicating interaction
    ACTION_VERBS_EN = {
        'kissed', 'embraced', 'held', 'touched', 'struck', 'hit',
        'pushed', 'pulled', 'grabbed', 'hugged', 'shook', 'slapped',
        'married', 'met', 'visited', 'left', 'followed', 'chased',
        'loved', 'hated', 'betrayed', 'saved', 'killed', 'helped',
    }

    # Russian dialogue verbs
    DIALOGUE_VERBS_RU = {
        'сказал', 'сказала', 'сказали', 'спросил', 'спросила',
        'ответил', 'ответила', 'прошептал', 'прошептала',
        'крикнул', 'крикнула', 'воскликнул', 'воскликнула',
        'проговорил', 'проговорила', 'произнес', 'произнесла',
        'молвил', 'молвила', 'заметил', 'заметила',
        'объявил', 'объявила', 'добавил', 'добавила',
    }

    # Russian action verbs
    ACTION_VERBS_RU = {
        'поцеловал', 'поцеловала', 'обнял', 'обняла', 'держал', 'держала',
        'ударил', 'ударила', 'толкнул', 'толкнула', 'схватил', 'схватила',
        'женился', 'вышла замуж', 'встретил', 'встретила', 'посетил',
        'любил', 'любила', 'ненавидел', 'ненавидела', 'предал', 'предала',
        'спас', 'спасла', 'убил', 'убила', 'помог', 'помогла',
    }

    def __init__(self, language: str = "en"):
        self.language = language
        if language == "ru":
            self.dialogue_verbs = self.DIALOGUE_VERBS_RU
            self.action_verbs = self.ACTION_VERBS_RU
        else:
            self.dialogue_verbs = self.DIALOGUE_VERBS_EN
            self.action_verbs = self.ACTION_VERBS_EN

    def extract_character_names(self, text: str) -> Set[str]:
        """Extract character names from text."""
        names = set()

        if self.language == "ru":
            # Russian: Look for capitalized Cyrillic words
            # Pattern: capital Cyrillic letter followed by lowercase (min 2 chars after capital)
            cyrillic_names = re.findall(r'(?<![А-ЯЁа-яё])([А-ЯЁ][а-яё]{2,})(?![а-яё])', text)

            # Extended Russian stopwords - common words that start with capital but aren't names
            ru_stopwords = {
                # Pronouns and demonstratives
                'Это', 'Она', 'Оно', 'Они', 'Его', 'Ему', 'Ней', 'Нем', 'Них',
                'Что', 'Кто', 'Как', 'Где', 'Там', 'Тут', 'Все', 'Всё', 'Вся',
                'Эти', 'Эта', 'Этот', 'Той', 'Тот', 'Того', 'Тем', 'Том',
                # Common sentence starters
                'Когда', 'Потом', 'После', 'Перед', 'Если', 'Хотя', 'Чтобы',
                'Потому', 'Поэтому', 'Однако', 'Впрочем', 'Между', 'Теперь',
                'Вдруг', 'Очень', 'Только', 'Уже', 'Еще', 'Ещё', 'Тогда',
                'Здесь', 'Сюда', 'Туда', 'Откуда', 'Куда', 'Никто', 'Ничто',
                'Каждый', 'Каждая', 'Каждое', 'Некто', 'Нечто', 'Кое',
                # Web/metadata (from lib.ru)
                'Классика', 'Регистрация', 'Найти', 'Рейтинги', 'Обсуждения',
                'Новинки', 'Обзоры', 'Помощь', 'Комментарии', 'Год', 'Обновлено',
                'Статистика', 'Роман', 'Проза', 'Романы', 'Скачать', 'Оценка',
                'Глава', 'Часть', 'Книга', 'Том', 'Эпилог', 'Пролог',
                # Misc common words
                'Может', 'Могут', 'Надо', 'Нужно', 'Хорошо', 'Плохо',
                'Правда', 'Неправда', 'Конечно', 'Наконец', 'Вообще',
                'Иногда', 'Всегда', 'Никогда', 'Нельзя', 'Можно',
                'Много', 'Мало', 'Больше', 'Меньше', 'Лучше', 'Хуже',
                # More sentence starters and common words
                'Одни', 'Одна', 'Одно', 'Один', 'Другой', 'Другая', 'Другие',
                'Либеральная', 'Оказалось', 'Окончив', 'Девочка', 'Мальчик',
                'Человек', 'Люди', 'Женщина', 'Мужчина', 'Ребенок', 'Дети',
                'Утром', 'Вечером', 'Ночью', 'Днем', 'Сегодня', 'Завтра', 'Вчера',
                'Первый', 'Второй', 'Третий', 'Последний', 'Следующий',
                'Почему', 'Зачем', 'Отчего', 'Почем', 'Разве', 'Неужели',
            }

            # Count occurrences - names appear multiple times
            name_counts = Counter(cyrillic_names)

            # Keep names that appear at least twice and aren't stopwords
            names = {n for n, count in name_counts.items()
                    if count >= 2 and n not in ru_stopwords and len(n) >= 3}

            # Also extract names from dialogue patterns
            # Pattern: -- Имя, or сказал(а) Имя, or Имя сказал(а)
            for verb in self.dialogue_verbs:
                # "сказал Иван" pattern
                pattern = rf'{verb}\s+([А-ЯЁ][а-яё]{{2,}})'
                for match in re.finditer(pattern, text):
                    name = match.group(1)
                    if name not in ru_stopwords:
                        names.add(name)
                # "Иван сказал" pattern
                pattern = rf'([А-ЯЁ][а-яё]{{2,}})\s+{verb}'
                for match in re.finditer(pattern, text):
                    name = match.group(1)
                    if name not in ru_stopwords:
                        names.add(name)

            # Russian dialogue marker pattern: "-- говорит Анна" or after em-dash
            dash_pattern = r'[—–-]\s*([А-ЯЁ][а-яё]{2,})\s+(?:сказал|говорит|ответил|спросил)'
            for match in re.finditer(dash_pattern, text):
                name = match.group(1)
                if name not in ru_stopwords:
                    names.add(name)

        else:
            # English: Look for capitalized words
            # Title patterns
            title_pattern = r'\b(?:Mr|Mrs|Miss|Dr|Sir|Lady|Count|Prince|Princess)\.?\s+([A-Z][a-z]+)'
            for match in re.finditer(title_pattern, text):
                names.add(match.group(1))

            # Dialogue attribution patterns
            for verb in self.dialogue_verbs:
                pattern = rf'\b([A-Z][a-z]+)\s+{verb}\b'
                for match in re.finditer(pattern, text):
                    names.add(match.group(1))
                pattern = rf'\b{verb}\s+([A-Z][a-z]+)\b'
                for match in re.finditer(pattern, text):
                    names.add(match.group(1))

            # General capitalized words (filter stopwords)
            en_stopwords = {
                'The', 'A', 'An', 'And', 'But', 'Or', 'In', 'On', 'At', 'To',
                'For', 'Of', 'With', 'By', 'From', 'Chapter', 'Part', 'Book',
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December',
            }
            cap_words = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
            word_counts = Counter(cap_words)
            for word, count in word_counts.items():
                if count >= 2 and word not in en_stopwords:
                    names.add(word)

        return names

    def extract_interactions(self, text: str, known_characters: Optional[Set[str]] = None) -> List[CharacterInteraction]:
        """
        Extract character interactions from text.

        Args:
            text: Narrative text
            known_characters: Optional set of known character names

        Returns:
            List of CharacterInteraction objects
        """
        interactions = []

        # Get character names
        if known_characters:
            characters = known_characters
        else:
            characters = self.extract_character_names(text)

        if len(characters) < 2:
            return interactions

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            # Find characters in this sentence
            chars_in_sentence = [c for c in characters if c in sentence]

            if len(chars_in_sentence) >= 2:
                # Co-occurrence interaction
                for i, char1 in enumerate(chars_in_sentence):
                    for char2 in chars_in_sentence[i+1:]:
                        interaction = CharacterInteraction(
                            char1=char1,
                            char2=char2,
                            interaction_type=InteractionType.CO_OCCURRENCE,
                            strength=1.0,
                            context=sentence[:100]
                        )
                        interactions.append(interaction)

                # Check for dialogue verbs
                sentence_lower = sentence.lower()
                for verb in self.dialogue_verbs:
                    if verb in sentence_lower:
                        for i, char1 in enumerate(chars_in_sentence):
                            for char2 in chars_in_sentence[i+1:]:
                                interaction = CharacterInteraction(
                                    char1=char1,
                                    char2=char2,
                                    interaction_type=InteractionType.DIALOGUE,
                                    strength=1.5,
                                    context=sentence[:100]
                                )
                                interactions.append(interaction)
                        break

                # Check for action verbs
                for verb in self.action_verbs:
                    if verb in sentence_lower:
                        for i, char1 in enumerate(chars_in_sentence):
                            for char2 in chars_in_sentence[i+1:]:
                                interaction = CharacterInteraction(
                                    char1=char1,
                                    char2=char2,
                                    interaction_type=InteractionType.ACTION,
                                    strength=2.0,
                                    context=sentence[:100]
                                )
                                interactions.append(interaction)
                        break

        return interactions


class CharacterInteractionFunctor(BaseFunctor):
    """
    Character interaction functor measuring relationship dynamics.

    Maps text windows to interaction density scores in [0, 1] where:
    - 0 = no character interactions
    - 1 = high character interaction density

    Also tracks:
    - Interaction graph (who interacts with whom)
    - Dominant relationships
    - New relationships forming
    """

    name = "character_interaction"

    def __init__(self, language: str = "en"):
        """
        Initialize character interaction functor.

        Args:
            language: "en", "ru", or "ja"
        """
        self.language = language
        self.extractor = CharacterInteractionExtractor(language)
        self._global_interactions: Dict[Tuple[str, str], float] = defaultdict(float)
        self._known_characters: Set[str] = set()

    def _score_window(self, text: str) -> Tuple[float, Dict]:
        """
        Compute interaction score for a text window.

        Returns:
            Tuple of (score, metadata)
        """
        # Extract interactions
        interactions = self.extractor.extract_interactions(text, self._known_characters)

        # Update known characters
        for interaction in interactions:
            self._known_characters.add(interaction.char1)
            self._known_characters.add(interaction.char2)

        # Count interaction types
        type_counts = Counter(i.interaction_type for i in interactions)

        # Calculate weighted interaction count
        weighted_count = sum(i.strength for i in interactions)

        # Update global interaction graph
        pair_strengths: Dict[Tuple[str, str], float] = defaultdict(float)
        for interaction in interactions:
            key = interaction.key()
            pair_strengths[key] += interaction.strength
            self._global_interactions[key] += interaction.strength

        # Normalize by text length
        word_count = len(text.split())
        if word_count == 0:
            return 0.0, {"interactions": 0}

        # Interaction density: interactions per 100 words
        density = (weighted_count / word_count) * 100

        # Normalize to [0, 1]
        # Typical range: 0-10 weighted interactions per 100 words
        interaction_score = min(1.0, density / 8)

        metadata = {
            "total_interactions": len(interactions),
            "weighted_interactions": weighted_count,
            "unique_pairs": len(pair_strengths),
            "top_pairs": dict(sorted(pair_strengths.items(), key=lambda x: -x[1])[:5]),
            "type_breakdown": {t.value: c for t, c in type_counts.items()},
        }

        return interaction_score, metadata

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply interaction functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with interaction scores
        """
        # Reset global tracking
        self._global_interactions = defaultdict(float)
        self._known_characters = set()

        # IMPORTANT: Pre-scan all windows to build character list
        # This ensures we find characters that appear across the full text
        full_text = ' '.join(windows)
        self._known_characters = self.extractor.extract_character_names(full_text)

        scores = []
        window_metadata = []

        for window in windows:
            score, meta = self._score_window(window)
            scores.append(score)
            window_metadata.append(meta)

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        # Get global relationship statistics
        top_relationships = dict(
            sorted(self._global_interactions.items(), key=lambda x: -x[1])[:10]
        )

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "language": self.language,
                "n_windows": len(windows),
                "total_characters": len(self._known_characters),
                "top_relationships": {
                    f"{k[0]}-{k[1]}": v for k, v in top_relationships.items()
                },
                "mean_interaction": float(np.mean(values)),
                "interaction_variance": float(np.var(values)),
            }
        )


class RussianCharacterInteractionFunctor(CharacterInteractionFunctor):
    """Character interaction functor for Russian text."""

    name = "character_interaction_ru"

    def __init__(self):
        super().__init__(language="ru")


# =============================================================================
# Narrative Event Detection
# =============================================================================

class EventType(Enum):
    """Types of narrative events."""
    MOVEMENT = "movement"           # Character moves/travels
    SPEECH_ACT = "speech_act"       # Dialogue, declaration
    PHYSICAL_ACTION = "physical"    # Physical action
    EMOTIONAL_CHANGE = "emotional"  # Emotional shift
    REVELATION = "revelation"       # Information revealed
    CONFLICT = "conflict"           # Conflict/confrontation
    RESOLUTION = "resolution"       # Resolution/reconciliation
    DEATH = "death"                 # Death event
    MARRIAGE = "marriage"           # Marriage/union
    ARRIVAL = "arrival"             # Arrival of character
    DEPARTURE = "departure"         # Departure/leaving


@dataclass
class NarrativeEvent:
    """Represents a narrative event."""
    event_type: EventType
    participants: List[str]
    action_verb: str
    intensity: float = 1.0
    context: str = ""


class NarrativeEventExtractor:
    """
    Extract narrative events from text.

    Detects plot-significant events through verb patterns and
    semantic markers.
    """

    # English event patterns by type
    EVENT_PATTERNS_EN = {
        EventType.MOVEMENT: [
            r'\b(went|came|arrived|left|traveled|journeyed|rode|walked|ran)\b',
            r'\b(returned|departed|entered|exited|approached)\b',
        ],
        EventType.DEATH: [
            r'\b(died|killed|murdered|dead|death|funeral|buried)\b',
            r'\b(perished|expired|passed away|breathed.+last)\b',
        ],
        EventType.MARRIAGE: [
            r'\b(married|wedding|bride|groom|engagement|engaged)\b',
            r'\b(husband|wife|matrimony|nuptials)\b',
        ],
        EventType.CONFLICT: [
            r'\b(fought|quarreled|argued|shouted|struck|attacked)\b',
            r'\b(angry|furious|rage|hatred|enemy|opposed)\b',
        ],
        EventType.REVELATION: [
            r'\b(discovered|revealed|learned|found out|realized)\b',
            r'\b(truth|secret|confession|admitted)\b',
        ],
        EventType.EMOTIONAL_CHANGE: [
            r'\b(fell in love|heartbroken|despair|joy|grief)\b',
            r'\b(wept|cried|laughed|sobbed|trembled)\b',
        ],
    }

    # Russian event patterns
    EVENT_PATTERNS_RU = {
        EventType.MOVEMENT: [
            r'\b(пошел|пошла|пришел|пришла|уехал|уехала|приехал|приехала)\b',
            r'\b(вернулся|вернулась|отправился|отправилась|вошел|вошла)\b',
        ],
        EventType.DEATH: [
            r'\b(умер|умерла|убил|убила|смерть|похороны|погиб|погибла)\b',
            r'\b(мертв|мертва|кончина|скончался|скончалась)\b',
        ],
        EventType.MARRIAGE: [
            r'\b(женился|вышла замуж|свадьба|невеста|жених|обручение)\b',
            r'\b(муж|жена|брак|венчание)\b',
        ],
        EventType.CONFLICT: [
            r'\b(ссорились|поссорились|кричал|кричала|ударил|напал)\b',
            r'\b(гнев|ярость|ненависть|враг|враждебно)\b',
        ],
        EventType.REVELATION: [
            r'\b(узнал|узнала|открылось|признание|признался|признала)\b',
            r'\b(правда|тайна|секрет|обнаружил|обнаружила)\b',
        ],
        EventType.EMOTIONAL_CHANGE: [
            r'\b(влюбился|влюбилась|разбитое сердце|отчаяние|радость|горе)\b',
            r'\b(плакал|плакала|рыдал|рыдала|смеялся|смеялась|дрожал)\b',
        ],
    }

    def __init__(self, language: str = "en"):
        self.language = language
        if language == "ru":
            self.patterns = self.EVENT_PATTERNS_RU
        else:
            self.patterns = self.EVENT_PATTERNS_EN

    def extract_events(self, text: str) -> List[NarrativeEvent]:
        """
        Extract narrative events from text.

        Args:
            text: Narrative text

        Returns:
            List of NarrativeEvent objects
        """
        events = []

        # Split into sentences for context
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()

            for event_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, sentence_lower)
                    if matches:
                        # Calculate intensity based on number of matches
                        intensity = min(2.0, 1.0 + 0.2 * len(matches))

                        # Weight certain event types higher
                        if event_type in [EventType.DEATH, EventType.MARRIAGE]:
                            intensity *= 1.5
                        elif event_type == EventType.CONFLICT:
                            intensity *= 1.2

                        event = NarrativeEvent(
                            event_type=event_type,
                            participants=[],  # Could extract from sentence
                            action_verb=matches[0] if matches else "",
                            intensity=intensity,
                            context=sentence[:100]
                        )
                        events.append(event)
                        break  # One event per type per sentence

        return events


class NarrativeEventFunctor(BaseFunctor):
    """
    Narrative event functor measuring plot activity.

    Maps text windows to event density scores in [0, 1] where:
    - 0 = no significant events
    - 1 = high event density (action-packed)

    Also tracks:
    - Event type distribution
    - Plot intensity curve
    - Major plot points
    """

    name = "narrative_event"

    def __init__(self, language: str = "en"):
        """
        Initialize event functor.

        Args:
            language: "en" or "ru"
        """
        self.language = language
        self.extractor = NarrativeEventExtractor(language)
        self._all_events: List[Tuple[float, NarrativeEvent]] = []

    def _score_window(self, text: str, position: float) -> Tuple[float, Dict]:
        """
        Compute event score for a text window.

        Args:
            text: Text window
            position: Narrative position (0-1)

        Returns:
            Tuple of (score, metadata)
        """
        events = self.extractor.extract_events(text)

        # Track events with position
        for event in events:
            self._all_events.append((position, event))

        # Calculate weighted event count
        weighted_count = sum(e.intensity for e in events)

        # Event type distribution
        type_counts = Counter(e.event_type for e in events)

        # Normalize by text length
        word_count = len(text.split())
        if word_count == 0:
            return 0.0, {"events": 0}

        # Event density per 100 words
        density = (weighted_count / word_count) * 100

        # Normalize to [0, 1]
        # Typical range: 0-5 weighted events per 100 words
        event_score = min(1.0, density / 4)

        metadata = {
            "total_events": len(events),
            "weighted_events": weighted_count,
            "type_breakdown": {t.value: c for t, c in type_counts.items()},
            "has_death": EventType.DEATH in type_counts,
            "has_marriage": EventType.MARRIAGE in type_counts,
            "has_conflict": EventType.CONFLICT in type_counts,
        }

        return event_score, metadata

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply event functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with event density scores
        """
        # Reset tracking
        self._all_events = []

        scores = []
        window_metadata = []

        for i, window in enumerate(windows):
            position = i / max(1, len(windows) - 1)
            score, meta = self._score_window(window, position)
            scores.append(score)
            window_metadata.append(meta)

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        # Analyze event distribution
        type_totals = Counter()
        for _, event in self._all_events:
            type_totals[event.event_type] += 1

        # Find major plot points (peaks in event density)
        major_points = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1] and values[i] > 0.5:
                major_points.append({
                    "position": float(time_points[i]),
                    "intensity": float(values[i]),
                    "events": window_metadata[i].get("type_breakdown", {}),
                })

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "language": self.language,
                "n_windows": len(windows),
                "total_events": len(self._all_events),
                "event_type_totals": {t.value: c for t, c in type_totals.items()},
                "major_plot_points": major_points[:10],  # Top 10
                "mean_event_density": float(np.mean(values)),
                "event_variance": float(np.var(values)),
            }
        )


class RussianNarrativeEventFunctor(NarrativeEventFunctor):
    """Narrative event functor for Russian text."""

    name = "narrative_event_ru"

    def __init__(self):
        super().__init__(language="ru")


# =============================================================================
# Plot State Tracking
# =============================================================================

class PlotStateType(Enum):
    """Types of plot states."""
    EQUILIBRIUM = "equilibrium"         # Stable state
    RISING_ACTION = "rising_action"     # Building tension
    CLIMAX = "climax"                   # Peak tension
    FALLING_ACTION = "falling_action"   # Resolution in progress
    RESOLUTION = "resolution"           # New equilibrium
    DISRUPTION = "disruption"           # Status quo broken


@dataclass
class PlotState:
    """Represents a plot state at a point in the narrative."""
    state_type: PlotStateType
    tension_level: float  # 0-1 scale
    active_conflicts: int
    unresolved_questions: int
    key_facts: List[str] = field(default_factory=list)


class PlotStateFunctor(BaseFunctor):
    """
    Plot state functor tracking narrative structure.

    Maps text windows to plot state scores representing:
    - Tension level (based on conflict and event markers)
    - Narrative phase (exposition, rising action, climax, etc.)
    - Information state (what the reader knows)

    This functor attempts to track "what is happening" structurally,
    complementing the sentiment functor's "how it feels."
    """

    name = "plot_state"

    def __init__(self, language: str = "en"):
        """
        Initialize plot state functor.

        Args:
            language: "en" or "ru"
        """
        self.language = language
        self.event_extractor = NarrativeEventExtractor(language)
        self._tension_history: List[float] = []
        self._conflict_markers = self._load_conflict_markers()
        self._resolution_markers = self._load_resolution_markers()

    def _load_conflict_markers(self) -> Set[str]:
        """Load conflict indicator words."""
        if self.language == "ru":
            return {
                'но', 'однако', 'хотя', 'несмотря', 'против', 'враг',
                'ссора', 'спор', 'конфликт', 'проблема', 'трудность',
                'невозможно', 'нельзя', 'отказ', 'борьба', 'война',
            }
        else:
            return {
                'but', 'however', 'although', 'despite', 'against', 'enemy',
                'quarrel', 'argument', 'conflict', 'problem', 'difficulty',
                'impossible', 'cannot', 'refused', 'struggle', 'war',
                'yet', 'still', 'nevertheless', 'obstacle', 'challenge',
            }

    def _load_resolution_markers(self) -> Set[str]:
        """Load resolution indicator words."""
        if self.language == "ru":
            return {
                'наконец', 'решено', 'согласие', 'мир', 'примирение',
                'понял', 'поняла', 'решение', 'ответ', 'конец',
                'счастливо', 'свадьба', 'вместе', 'прощение', 'любовь',
            }
        else:
            return {
                'finally', 'resolved', 'agreement', 'peace', 'reconciliation',
                'understood', 'solution', 'answer', 'ending', 'conclusion',
                'happily', 'wedding', 'together', 'forgiveness', 'love',
                'at last', 'eventually', 'settled', 'harmony',
            }

    def _calculate_tension(self, text: str) -> float:
        """
        Calculate tension level from text markers.

        Returns:
            Tension score in [0, 1]
        """
        text_lower = text.lower()
        words = text_lower.split()

        if not words:
            return 0.5

        # Count conflict and resolution markers
        conflict_count = sum(1 for w in words if w in self._conflict_markers)
        resolution_count = sum(1 for w in words if w in self._resolution_markers)

        # Extract events
        events = self.event_extractor.extract_events(text)

        # Event-based tension adjustment
        event_tension = 0.0
        for event in events:
            if event.event_type in [EventType.DEATH, EventType.CONFLICT]:
                event_tension += 0.15 * event.intensity
            elif event.event_type == EventType.REVELATION:
                event_tension += 0.10 * event.intensity
            elif event.event_type in [EventType.MARRIAGE, EventType.RESOLUTION]:
                event_tension -= 0.10 * event.intensity

        # Base tension from markers (normalized)
        word_count = len(words)
        marker_tension = (conflict_count - resolution_count) / max(1, word_count) * 10

        # Combine
        tension = 0.5 + marker_tension + event_tension

        # Smooth with history (momentum)
        if self._tension_history:
            last_tension = self._tension_history[-1]
            tension = 0.7 * tension + 0.3 * last_tension

        return float(np.clip(tension, 0, 1))

    def _classify_state(self, tension: float, position: float) -> PlotStateType:
        """
        Classify plot state based on tension and narrative position.

        Args:
            tension: Current tension level (0-1)
            position: Position in narrative (0-1)

        Returns:
            PlotStateType classification
        """
        # Use both tension and position
        if tension > 0.8:
            return PlotStateType.CLIMAX
        elif tension > 0.6:
            if position < 0.7:
                return PlotStateType.RISING_ACTION
            else:
                return PlotStateType.FALLING_ACTION
        elif tension < 0.3:
            if position < 0.2:
                return PlotStateType.EQUILIBRIUM
            else:
                return PlotStateType.RESOLUTION
        else:
            # Check tension trend
            if len(self._tension_history) >= 3:
                recent = self._tension_history[-3:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    return PlotStateType.RISING_ACTION
                elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    return PlotStateType.FALLING_ACTION
            return PlotStateType.EQUILIBRIUM

    def _score_window(self, text: str, position: float) -> Tuple[float, Dict]:
        """
        Compute plot state for a text window.

        Args:
            text: Text window
            position: Narrative position (0-1)

        Returns:
            Tuple of (tension_score, metadata)
        """
        tension = self._calculate_tension(text)
        self._tension_history.append(tension)

        state_type = self._classify_state(tension, position)

        metadata = {
            "tension": tension,
            "state_type": state_type.value,
            "position": position,
        }

        return tension, metadata

    def __call__(self, windows: List[str]) -> Trajectory:
        """
        Apply plot state functor to text windows.

        Args:
            windows: List of text windows

        Returns:
            Trajectory with tension scores
        """
        # Reset
        self._tension_history = []

        scores = []
        window_metadata = []
        state_sequence = []

        for i, window in enumerate(windows):
            position = i / max(1, len(windows) - 1)
            score, meta = self._score_window(window, position)
            scores.append(score)
            window_metadata.append(meta)
            state_sequence.append(meta["state_type"])

        values = np.array(scores)
        time_points = np.linspace(0, 1, len(values))

        # Analyze narrative structure
        state_counts = Counter(state_sequence)

        # Find climax position (highest tension)
        if len(values) > 0:
            climax_idx = int(np.argmax(values))
            climax_position = float(time_points[climax_idx])
            climax_tension = float(values[climax_idx])
        else:
            climax_position = 0.5
            climax_tension = 0.5

        return Trajectory(
            values=values,
            time_points=time_points,
            functor_name=self.name,
            metadata={
                "language": self.language,
                "n_windows": len(windows),
                "state_distribution": dict(state_counts),
                "climax_position": climax_position,
                "climax_tension": climax_tension,
                "mean_tension": float(np.mean(values)),
                "tension_variance": float(np.var(values)),
                "narrative_structure": self._analyze_structure(state_sequence),
            }
        )

    def _analyze_structure(self, state_sequence: List[str]) -> str:
        """
        Analyze overall narrative structure from state sequence.

        Returns:
            String describing narrative structure type
        """
        if not state_sequence:
            return "unknown"

        # Count transitions
        n = len(state_sequence)
        if n < 4:
            return "too_short"

        # Simple structure analysis
        first_quarter = Counter(state_sequence[:n//4])
        last_quarter = Counter(state_sequence[-n//4:])

        starts_equilibrium = first_quarter.get("equilibrium", 0) > first_quarter.get("climax", 0)
        ends_resolution = last_quarter.get("resolution", 0) > last_quarter.get("climax", 0)

        has_climax = "climax" in state_sequence

        if starts_equilibrium and has_climax and ends_resolution:
            return "classic_arc"  # Freytag's pyramid
        elif starts_equilibrium and has_climax and not ends_resolution:
            return "tragic_arc"  # No resolution
        elif not starts_equilibrium and has_climax:
            return "in_medias_res"  # Starts in action
        elif not has_climax:
            return "episodic"  # No clear climax
        else:
            return "complex"


class RussianPlotStateFunctor(PlotStateFunctor):
    """Plot state functor for Russian text."""

    name = "plot_state_ru"

    def __init__(self):
        super().__init__(language="ru")


# =============================================================================
# Utility Functions
# =============================================================================

def compare_semantic_trajectories(
    original_trajectory: Trajectory,
    translation_trajectory: Trajectory,
) -> Dict:
    """
    Compare semantic trajectories between original and translation.

    This complements sentiment comparison by looking at structural/plot
    divergence rather than just emotional divergence.

    Args:
        original_trajectory: Trajectory from original text
        translation_trajectory: Trajectory from translation

    Returns:
        Dict with comparison metrics
    """
    # Resample to same length
    n_points = 100
    orig = original_trajectory.resample(n_points).values
    trans = translation_trajectory.resample(n_points).values

    # Correlation
    pearson = float(np.corrcoef(orig, trans)[0, 1])

    # Mean absolute difference
    mad = float(np.mean(np.abs(orig - trans)))

    # Structural similarity (peak alignment)
    orig_peaks = np.where((orig[1:-1] > orig[:-2]) & (orig[1:-1] > orig[2:]))[0] + 1
    trans_peaks = np.where((trans[1:-1] > trans[:-2]) & (trans[1:-1] > trans[2:]))[0] + 1

    if len(orig_peaks) > 0 and len(trans_peaks) > 0:
        # Find closest peak alignment
        min_distances = []
        for op in orig_peaks:
            if len(trans_peaks) > 0:
                min_dist = min(abs(op - tp) for tp in trans_peaks)
                min_distances.append(min_dist)
        peak_alignment = 1.0 - (np.mean(min_distances) / n_points) if min_distances else 0.0
    else:
        peak_alignment = 0.0

    return {
        "pearson_correlation": pearson if not np.isnan(pearson) else 0.0,
        "mean_absolute_difference": mad,
        "peak_alignment": float(peak_alignment),
        "original_peaks": len(orig_peaks),
        "translation_peaks": len(trans_peaks),
        "functor": original_trajectory.functor_name,
    }
