"""
Extraction module for dynamic analysis of narrative texts.

Provides automatic extraction of:
- Character definitions from text
- Relationship networks
- Event patterns
"""

from .dynamic_characters import (
    DynamicCharacterExtractor,
    ExtractedCharacter,
    extract_characters_from_file,
)

__all__ = [
    'DynamicCharacterExtractor',
    'ExtractedCharacter',
    'extract_characters_from_file',
]
