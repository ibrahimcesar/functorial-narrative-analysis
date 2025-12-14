"""
Multi-Scale Narrative Analysis

Detects narrative structures at multiple scales:
- Whole work level
- Chapter/act level
- Scene/section level

Key insight: Longer works may contain nested structures - kishōtenketsu-within-kishōtenketsu,
or Western three-act structure containing multiple kishōtenketsu episodes.

This addresses the theoretical question: Are narrative shapes fractal/self-similar
across scales, or do different scales exhibit different structural principles?
"""

import json
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


@dataclass
class Segment:
    """A segment of text at a particular scale."""
    start_pos: float  # Normalized position [0, 1]
    end_pos: float
    start_char: int   # Character index
    end_char: int
    text: str
    level: str        # "work", "chapter", "section", "paragraph"
    index: int        # Index at this level
    title: Optional[str] = None


@dataclass
class ScaleAnalysis:
    """Analysis results at a single scale."""
    level: str
    n_segments: int
    segments: List[Segment]
    shape_label: str
    conformance_score: float
    ten_position: Optional[float] = None
    ten_strength: Optional[float] = None
    pattern_type: str = ""
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class MultiScaleResult:
    """Complete multi-scale analysis of a work."""
    work_id: str
    title: str
    author: str
    total_length: int

    # Analysis at each scale
    work_level: Optional[ScaleAnalysis] = None
    chapter_level: Optional[ScaleAnalysis] = None
    section_level: Optional[ScaleAnalysis] = None

    # Cross-scale patterns
    scale_consistency: float = 0.0  # Do shapes match across scales?
    nested_structures: List[Dict] = field(default_factory=list)
    fractal_dimension: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "work_id": self.work_id,
            "title": self.title,
            "author": self.author,
            "total_length": self.total_length,
            "work_level": asdict(self.work_level) if self.work_level else None,
            "chapter_level": asdict(self.chapter_level) if self.chapter_level else None,
            "section_level": asdict(self.section_level) if self.section_level else None,
            "scale_consistency": self.scale_consistency,
            "nested_structures": self.nested_structures,
            "fractal_dimension": self.fractal_dimension,
        }


class TextSegmenter:
    """
    Segments text into hierarchical units.

    Hierarchy:
    - Work (full text)
    - Chapters (explicit markers or length-based)
    - Sections (paragraph clusters or scene breaks)
    - Paragraphs (basic unit)
    """

    # Chapter markers for different languages
    CHAPTER_PATTERNS = [
        # English
        r'(?:^|\n)(?:Chapter|CHAPTER)\s+(?:\d+|[IVXLC]+|[A-Z][a-z]*)',
        r'(?:^|\n)(?:Part|PART)\s+(?:\d+|[IVXLC]+|[A-Z][a-z]*)',
        r'(?:^|\n)(?:Book|BOOK)\s+(?:\d+|[IVXLC]+|[A-Z][a-z]*)',
        # Japanese
        r'(?:^|\n)第[一二三四五六七八九十百千\d]+[章回話編節部]',
        r'(?:^|\n)[一二三四五六七八九十]+[\s　、]',
        r'(?:^|\n)【[^】]+】',
        # Numbered sections
        r'(?:^|\n)(?:\d+\.|\(\d+\)|\[\d+\])\s+',
    ]

    # Scene break patterns
    SCENE_BREAK_PATTERNS = [
        r'\n\s*\*\s*\*\s*\*\s*\n',
        r'\n\s*[*#-]{3,}\s*\n',
        r'\n\n\n+',
        r'(?:^|\n)　　[＊＃—]{3,}',
    ]

    def __init__(self, min_segment_chars: int = 1000):
        self.min_segment_chars = min_segment_chars

    def segment_into_chapters(self, text: str) -> List[Segment]:
        """
        Segment text into chapters using explicit markers or heuristics.
        """
        # Try explicit chapter markers
        for pattern in self.CHAPTER_PATTERNS:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if len(matches) >= 2:
                return self._create_segments_from_matches(text, matches, "chapter")

        # Fallback: divide by length into ~4-8 segments
        return self._segment_by_length(text, n_segments=6, level="chapter")

    def segment_into_sections(self, text: str) -> List[Segment]:
        """
        Segment text into sections (scenes, major paragraph groups).
        """
        # Try scene break markers
        for pattern in self.SCENE_BREAK_PATTERNS:
            matches = list(re.finditer(pattern, text))
            if len(matches) >= 3:
                return self._create_segments_from_matches(text, matches, "section")

        # Fallback: segment by paragraph clusters
        return self._segment_by_paragraphs(text, target_sections=12)

    def segment_into_paragraphs(self, text: str) -> List[Segment]:
        """
        Segment text into paragraphs.
        """
        # Split on double newlines or Japanese paragraph markers
        para_pattern = r'\n\n+|(?<=。)\n(?=　)'
        parts = re.split(para_pattern, text)

        segments = []
        char_pos = 0

        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) < 50:  # Skip very short paragraphs
                char_pos += len(part) + 2
                continue

            start_char = text.find(part, char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(part)

            segments.append(Segment(
                start_pos=start_char / len(text),
                end_pos=end_char / len(text),
                start_char=start_char,
                end_char=end_char,
                text=part,
                level="paragraph",
                index=len(segments),
            ))

            char_pos = end_char

        return segments

    def _create_segments_from_matches(
        self,
        text: str,
        matches: List[re.Match],
        level: str
    ) -> List[Segment]:
        """Create segments from regex matches."""
        segments = []
        n = len(text)

        for i, match in enumerate(matches):
            start_char = match.start()
            end_char = matches[i + 1].start() if i + 1 < len(matches) else n

            # Extract title from match
            title = match.group().strip()

            segment_text = text[start_char:end_char]

            if len(segment_text) >= self.min_segment_chars:
                segments.append(Segment(
                    start_pos=start_char / n,
                    end_pos=end_char / n,
                    start_char=start_char,
                    end_char=end_char,
                    text=segment_text,
                    level=level,
                    index=len(segments),
                    title=title,
                ))

        return segments

    def _segment_by_length(
        self,
        text: str,
        n_segments: int,
        level: str
    ) -> List[Segment]:
        """Divide text into roughly equal segments."""
        n = len(text)
        segment_size = n // n_segments

        segments = []
        for i in range(n_segments):
            start_char = i * segment_size
            end_char = (i + 1) * segment_size if i < n_segments - 1 else n

            segments.append(Segment(
                start_pos=start_char / n,
                end_pos=end_char / n,
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
                level=level,
                index=i,
                title=f"{level.title()} {i + 1}",
            ))

        return segments

    def _segment_by_paragraphs(
        self,
        text: str,
        target_sections: int
    ) -> List[Segment]:
        """Group paragraphs into sections."""
        paragraphs = self.segment_into_paragraphs(text)

        if len(paragraphs) <= target_sections:
            # Relabel as sections
            for i, seg in enumerate(paragraphs):
                seg.level = "section"
                seg.index = i
            return paragraphs

        # Group paragraphs
        paras_per_section = len(paragraphs) // target_sections

        segments = []
        for i in range(target_sections):
            start_idx = i * paras_per_section
            end_idx = (i + 1) * paras_per_section if i < target_sections - 1 else len(paragraphs)

            group = paragraphs[start_idx:end_idx]
            if not group:
                continue

            combined_text = "\n\n".join(p.text for p in group)

            segments.append(Segment(
                start_pos=group[0].start_pos,
                end_pos=group[-1].end_pos,
                start_char=group[0].start_char,
                end_char=group[-1].end_char,
                text=combined_text,
                level="section",
                index=len(segments),
            ))

        return segments


class MultiScaleAnalyzer:
    """
    Analyzes narrative structure at multiple scales.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.segmenter = TextSegmenter()

    def analyze(
        self,
        text: str,
        work_id: str = "",
        title: str = "",
        author: str = "",
    ) -> MultiScaleResult:
        """
        Perform multi-scale analysis on a text.
        """
        from src.geometry.surprisal import SurprisalExtractor
        from src.detectors.kishotenketsu import KishotenketsuDetector

        extractor = SurprisalExtractor(method='entropy', window_size=self.window_size)
        detector = KishotenketsuDetector()

        result = MultiScaleResult(
            work_id=work_id,
            title=title,
            author=author,
            total_length=len(text),
        )

        # === Work-level analysis ===
        try:
            trajectory = extractor.extract(text)
            match = detector.detect(trajectory.values, trajectory_id=work_id, title=title)

            result.work_level = ScaleAnalysis(
                level="work",
                n_segments=1,
                segments=[],
                shape_label=self._trajectory_to_shape(trajectory.values),
                conformance_score=match.conformance_score,
                ten_position=match.ten_position,
                ten_strength=match.ten_strength,
                pattern_type=match.pattern_type,
                features=self._extract_features(trajectory.values),
            )
        except Exception as e:
            print(f"Work-level error: {e}")

        # === Chapter-level analysis ===
        chapters = self.segmenter.segment_into_chapters(text)
        if len(chapters) >= 3:
            chapter_shapes = []
            chapter_features = []

            for chapter in chapters:
                try:
                    traj = extractor.extract(chapter.text)
                    shape = self._trajectory_to_shape(traj.values)
                    chapter_shapes.append(shape)
                    chapter_features.append(self._extract_features(traj.values))
                except:
                    chapter_shapes.append("unknown")
                    chapter_features.append({})

            # Aggregate chapter patterns
            shape_counts = Counter(chapter_shapes)
            dominant_shape = shape_counts.most_common(1)[0][0] if shape_counts else "unknown"

            # Compute chapter-level trajectory (mean surprisal per chapter)
            chapter_means = []
            for f in chapter_features:
                if 'mean_surprisal' in f:
                    chapter_means.append(f['mean_surprisal'])

            if chapter_means:
                chapter_trajectory = np.array(chapter_means)
                chapter_match = detector.detect(
                    chapter_trajectory,
                    trajectory_id=f"{work_id}_chapters",
                    title=f"{title} (chapters)"
                )

                result.chapter_level = ScaleAnalysis(
                    level="chapter",
                    n_segments=len(chapters),
                    segments=chapters,
                    shape_label=dominant_shape,
                    conformance_score=chapter_match.conformance_score,
                    ten_position=chapter_match.ten_position,
                    ten_strength=chapter_match.ten_strength,
                    pattern_type=chapter_match.pattern_type,
                    features={
                        "shape_distribution": dict(shape_counts),
                        "chapter_shapes": chapter_shapes,
                    },
                )

        # === Section-level analysis ===
        sections = self.segmenter.segment_into_sections(text)
        if len(sections) >= 4:
            section_shapes = []
            section_features = []

            for section in sections:
                try:
                    if len(section.text) < 500:
                        continue
                    traj = extractor.extract(section.text)
                    shape = self._trajectory_to_shape(traj.values)
                    section_shapes.append(shape)
                    section_features.append(self._extract_features(traj.values))
                except:
                    pass

            if section_shapes:
                shape_counts = Counter(section_shapes)
                dominant_shape = shape_counts.most_common(1)[0][0]

                result.section_level = ScaleAnalysis(
                    level="section",
                    n_segments=len(sections),
                    segments=sections,
                    shape_label=dominant_shape,
                    conformance_score=0.0,
                    features={
                        "shape_distribution": dict(shape_counts),
                    },
                )

        # === Cross-scale analysis ===
        result.scale_consistency = self._compute_scale_consistency(result)
        result.nested_structures = self._detect_nested_structures(result, text, extractor, detector)

        return result

    def _trajectory_to_shape(self, values: np.ndarray) -> str:
        """
        Classify trajectory into Reagan-style shape.
        """
        if len(values) < 4:
            return "unknown"

        # Compute quartile means
        q_size = len(values) // 4
        if q_size == 0:
            return "unknown"

        q1 = np.mean(values[:q_size])
        q2 = np.mean(values[q_size:2*q_size])
        q3 = np.mean(values[2*q_size:3*q_size])
        q4 = np.mean(values[3*q_size:])

        # Overall trend
        start = np.mean(values[:len(values)//4])
        end = np.mean(values[-len(values)//4:])
        mid = np.mean(values[len(values)//4:3*len(values)//4])

        # Shape classification
        rising = end > start * 1.1
        falling = end < start * 0.9

        mid_low = mid < (start + end) / 2 * 0.9
        mid_high = mid > (start + end) / 2 * 1.1

        if rising and not mid_low and not mid_high:
            return "rags_to_riches"
        elif falling and not mid_low and not mid_high:
            return "tragedy"
        elif mid_low:
            return "man_in_hole"
        elif mid_high:
            return "icarus"
        elif q4 > q2 and (q1 < q2 or q3 < q4):
            return "cinderella"
        elif q2 > q4 and (q1 > q2 or q3 > q4):
            return "oedipus"
        else:
            return "other"

    def _extract_features(self, values: np.ndarray) -> Dict[str, float]:
        """Extract basic features from trajectory."""
        return {
            "mean_surprisal": float(np.mean(values)),
            "std_surprisal": float(np.std(values)),
            "min_surprisal": float(np.min(values)),
            "max_surprisal": float(np.max(values)),
            "entropy_change": float(np.mean(values[-len(values)//4:]) - np.mean(values[:len(values)//4])),
        }

    def _compute_scale_consistency(self, result: MultiScaleResult) -> float:
        """
        Compute how consistent shapes are across scales.

        High consistency = fractal/self-similar structure
        Low consistency = different organizing principles at different scales
        """
        shapes = []

        if result.work_level:
            shapes.append(result.work_level.shape_label)
        if result.chapter_level:
            shapes.append(result.chapter_level.shape_label)
        if result.section_level:
            shapes.append(result.section_level.shape_label)

        if len(shapes) < 2:
            return 0.0

        # Simple consistency: proportion of matching shapes
        matches = sum(1 for i in range(len(shapes) - 1) if shapes[i] == shapes[i + 1])
        return matches / (len(shapes) - 1)

    def _detect_nested_structures(
        self,
        result: MultiScaleResult,
        text: str,
        extractor,
        detector,
    ) -> List[Dict]:
        """
        Detect nested kishōtenketsu or other structures.

        A nested structure is when a chapter/section itself contains
        a complete four-act (Ki-Shō-Ten-Ketsu) pattern.
        """
        nested = []

        if not result.chapter_level:
            return nested

        for chapter in result.chapter_level.segments:
            if len(chapter.text) < 3000:
                continue

            try:
                traj = extractor.extract(chapter.text)
                match = detector.detect(
                    traj.values,
                    trajectory_id=f"chapter_{chapter.index}",
                    title=chapter.title or f"Chapter {chapter.index + 1}"
                )

                if "kishotenketsu" in match.pattern_type:
                    nested.append({
                        "level": "chapter",
                        "index": chapter.index,
                        "title": chapter.title,
                        "pattern": match.pattern_type,
                        "conformance": match.conformance_score,
                        "ten_position": match.ten_position,
                        "position_in_work": (chapter.start_pos + chapter.end_pos) / 2,
                    })
            except:
                pass

        return nested


def analyze_corpus_multiscale(
    corpus_dir: Path,
    output_path: Optional[Path] = None,
    max_works: int = 100,
) -> Dict:
    """
    Run multi-scale analysis on a corpus.
    """
    analyzer = MultiScaleAnalyzer()
    results = []

    json_files = list(corpus_dir.glob("*.json"))[:max_works]

    for i, f in enumerate(json_files, 1):
        with open(f) as file:
            data = json.load(file)

        text = data.get('text', '')
        if len(text) < 5000:
            continue

        title = data.get('title', f.stem)
        print(f"[{i}/{len(json_files)}] Analyzing: {title}")

        try:
            result = analyzer.analyze(
                text,
                work_id=str(data.get('book_id', f.stem)),
                title=title,
                author=data.get('author', 'Unknown'),
            )
            results.append(result.to_dict())
        except Exception as e:
            print(f"  Error: {e}")

    # Aggregate statistics
    output = {
        "n_works": len(results),
        "results": results,
        "summary": _compute_summary(results),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    return output


def _compute_summary(results: List[Dict]) -> Dict:
    """Compute aggregate summary statistics."""
    if not results:
        return {}

    # Scale consistency
    consistencies = [r.get('scale_consistency', 0) for r in results]

    # Nested structures
    total_nested = sum(len(r.get('nested_structures', [])) for r in results)
    works_with_nested = sum(1 for r in results if r.get('nested_structures'))

    # Shape distributions at each level
    work_shapes = Counter()
    chapter_shapes = Counter()

    for r in results:
        if r.get('work_level'):
            work_shapes[r['work_level'].get('shape_label', 'unknown')] += 1
        if r.get('chapter_level'):
            chapter_shapes[r['chapter_level'].get('shape_label', 'unknown')] += 1

    return {
        "mean_scale_consistency": float(np.mean(consistencies)) if consistencies else 0,
        "std_scale_consistency": float(np.std(consistencies)) if consistencies else 0,
        "works_with_nested_structures": works_with_nested,
        "total_nested_structures": total_nested,
        "work_level_shapes": dict(work_shapes),
        "chapter_level_shapes": dict(chapter_shapes),
    }


if __name__ == "__main__":
    # Run on Japanese corpus
    print("=== Multi-Scale Analysis: Japanese Corpus ===")
    jp_results = analyze_corpus_multiscale(
        Path("data/raw/corpus_expanded/japanese"),
        Path("data/results/multiscale_japanese.json"),
    )

    print("\n=== Multi-Scale Analysis: Western Corpus ===")
    west_results = analyze_corpus_multiscale(
        Path("data/raw/corpus_expanded/western"),
        Path("data/results/multiscale_western.json"),
    )

    # Print summary comparison
    print("\n" + "=" * 60)
    print("MULTI-SCALE ANALYSIS SUMMARY")
    print("=" * 60)

    jp_sum = jp_results.get('summary', {})
    west_sum = west_results.get('summary', {})

    print(f"\n{'Metric':<35} {'Japanese':<15} {'Western':<15}")
    print("-" * 65)
    print(f"{'Works analyzed':<35} {jp_results.get('n_works', 0):<15} {west_results.get('n_works', 0):<15}")
    print(f"{'Mean scale consistency':<35} {jp_sum.get('mean_scale_consistency', 0):.3f}          {west_sum.get('mean_scale_consistency', 0):.3f}")
    print(f"{'Works with nested structures':<35} {jp_sum.get('works_with_nested_structures', 0):<15} {west_sum.get('works_with_nested_structures', 0):<15}")
    print(f"{'Total nested structures':<35} {jp_sum.get('total_nested_structures', 0):<15} {west_sum.get('total_nested_structures', 0):<15}")
