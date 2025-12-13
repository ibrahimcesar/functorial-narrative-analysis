"""
Corpus ingestion pipelines.

Modules for downloading and processing text corpora from various sources.

Available Pipelines:
    - GutenbergPipeline: Project Gutenberg fiction corpus
    - SyosetuPipeline: Syosetu (小説家になろう) Japanese web novels
"""

from .gutenberg import GutenbergPipeline, GutenbergBook
from .syosetu import SyosetuPipeline, SyosetuNovel, SYOSETU_GENRES

__all__ = [
    "GutenbergPipeline",
    "GutenbergBook",
    "SyosetuPipeline",
    "SyosetuNovel",
    "SYOSETU_GENRES",
]
