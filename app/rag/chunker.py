import re
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings


@dataclass
class DocumentChunk:
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0


class MarkdownChunker:
    """
    Markdown-aware document chunker.
    Ported from Java DocumentChunkService:
    1. Split by markdown headings
    2. Split long sections by paragraphs
    3. Apply max_size and overlap at section boundaries
    """

    def __init__(self, max_size: int = 800, overlap: int = 100):
        self.max_size = max_size
        self.overlap = overlap

    def chunk(self, content: str, source: str) -> list[DocumentChunk]:
        if not content or not content.strip():
            return []

        sections = self._split_by_headings(content)
        chunks: list[DocumentChunk] = []
        global_chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, global_chunk_index)
            chunks.extend(section_chunks)
            global_chunk_index += len(section_chunks)

        # Add source metadata to all chunks
        for chunk in chunks:
            chunk.metadata["_source"] = source.replace("\\", "/")

        return chunks

    def _split_by_headings(self, content: str) -> list["Section"]:
        """
        Split content by markdown headings (# ## ### etc).
        Ported from DocumentChunkService.splitByHeadings()
        """
        sections = []
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        last_end = 0
        current_title: Optional[str] = None

        for match in heading_pattern.finditer(content):
            if last_end < match.start():
                section_content = content[last_end : match.start()].strip()
                if section_content:
                    sections.append(Section(title=current_title, content=section_content, start_index=last_end))

            current_title = match.group(2).strip()
            last_end = match.start()

        # Last section
        if last_end < len(content):
            section_content = content[last_end:].strip()
            if section_content:
                sections.append(Section(title=current_title, content=section_content, start_index=last_end))

        # If no headings found, treat whole document as one section
        if not sections:
            sections.append(Section(title=None, content=content, start_index=0))

        return sections

    def _chunk_section(self, section: "Section", start_chunk_index: int) -> list[DocumentChunk]:
        """Chunk a single section, ported from DocumentChunkService.chunkSection()"""
        content = section.content
        title = section.title
        chunks = []

        # If section is small enough, keep as one chunk
        if len(content) <= self.max_size:
            chunk = DocumentChunk(
                content=content,
                metadata={"title": title} if title else {},
                chunk_index=start_chunk_index,
            )
            chunks.append(chunk)
            return chunks

        # Split long sections by paragraphs
        paragraphs = self._split_by_paragraphs(content)

        current_chunk = ""
        current_start_index = section.start_index
        chunk_index = start_chunk_index

        for paragraph in paragraphs:
            if current_chunk and len(current_chunk) + len(paragraph) > self.max_size:
                # Save current chunk
                chunk_content = current_chunk.strip()
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata={"title": title} if title else {},
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(chunk_content)
                current_chunk = overlap_text
                current_start_index = current_start_index + len(chunk_content) - len(overlap_text)

            current_chunk += paragraph + "\n\n"

        # Save last chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata={"title": title} if title else {},
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

        return chunks

    def _split_by_paragraphs(self, content: str) -> list[str]:
        """Split content by double newlines"""
        paragraphs = []
        for part in re.split(r"\n\n+", content):
            trimmed = part.strip()
            if trimmed:
                paragraphs.append(trimmed)
        return paragraphs

    def _get_overlap_text(self, text: str) -> str:
        """Get overlapping text from the end, ported from DocumentChunkService.getOverlapText()"""
        overlap_size = min(self.overlap, len(text))
        if overlap_size <= 0:
            return ""

        overlap = text[-overlap_size:]

        # Try to break at sentence boundary
        for punct in ("。", "？", "！"):
            idx = overlap.rfind(punct)
            if idx > overlap_size // 2:
                return overlap[idx + 1 :].strip()

        return overlap.strip()


class Section:
    def __init__(self, title: Optional[str], content: str, start_index: int):
        self.title = title
        self.content = content
        self.start_index = start_index
