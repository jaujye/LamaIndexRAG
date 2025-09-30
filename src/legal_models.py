"""
Shared data models for legal document fetchers.
This module provides common data structures used across different legal document fetchers.
"""

from dataclasses import dataclass


@dataclass
class LegalArticle:
    """
    Represents a single legal article from any Taiwan legal code.

    Attributes:
        article_number: The article number (e.g., "1", "2", "100-1")
        title: The title or brief description of the article
        content: The full text content of the article
        section: The section/chapter/book name (章/編)
        section_number: The section/chapter/book number
        url: The full URL to the article on law.moj.gov.tw
    """
    article_number: str
    title: str
    content: str
    section: str  # Generic term: can be "章" (chapter) or "編" (book)
    section_number: str
    url: str

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'article_number': self.article_number,
            'title': self.title,
            'content': self.content,
            'section': self.section,
            'section_number': self.section_number,
            'url': self.url
        }