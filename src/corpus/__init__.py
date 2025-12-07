"""
Corpus collection modules for Functorial Narrative Analysis.

This package provides collectors for various narrative corpora:
- gutenberg: Project Gutenberg public domain texts (English)
- aozora: Aozora Bunko public domain texts (Japanese)
- ao3: Archive of Our Own fan fiction
- syosetu: Japanese web novels
- ancient: Ancient epics (Gilgamesh, Homer, etc.)
- streaming: Streaming-era television content
"""

from .gutenberg import GutenbergCollector
from .aozora import AozoraCollector

__all__ = ["GutenbergCollector", "AozoraCollector"]
