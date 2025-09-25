"""
Document processor for legal texts
Handles chunking while preserving legal structure and metadata
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
from llama_index.core import Document
from llama_index.core.schema import TextNode


@dataclass
class LegalChunk:
    """Represents a processed chunk of legal text"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    article_number: str
    chapter: str


class LegalDocumentProcessor:
    """Processes legal documents for RAG indexing"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize document processor

        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def load_legal_data(self, filepath: str) -> Dict[str, Any]:
        """Load legal data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def clean_legal_text(self, text: str) -> str:
        """Clean and normalize legal text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Normalize punctuation
        text = re.sub(r'：', ':', text)
        text = re.sub(r'；', ';', text)

        # Remove page references and footnotes
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'第\s*\d+\s*頁', '', text)

        return text

    def extract_article_structure(self, content: str) -> Dict[str, str]:
        """Extract structured information from article content"""
        structure = {
            'main_provision': '',
            'items': [],
            'exceptions': '',
            'penalties': ''
        }

        # Split by common legal structures
        lines = content.split('\n')
        current_section = 'main_provision'
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect numbered items
            if re.match(r'[一二三四五六七八九十]\s*、', line) or re.match(r'\d+\s*、', line):
                if current_text:
                    if current_section == 'main_provision':
                        structure['main_provision'] = '\n'.join(current_text)
                    current_text = []
                structure['items'].append(line)
                current_section = 'items'

            # Detect penalty clauses
            elif '處' in line and ('罰' in line or '元' in line):
                structure['penalties'] = line
                current_section = 'penalties'

            # Detect exception clauses
            elif '但' in line[:5] or '除' in line[:5]:
                structure['exceptions'] += line + '\n'
                current_section = 'exceptions'

            else:
                current_text.append(line)

        # Handle remaining text
        if current_text and current_section == 'main_provision':
            structure['main_provision'] = '\n'.join(current_text)

        return structure

    def create_article_chunks(self, article: Dict[str, Any]) -> List[LegalChunk]:
        """Create chunks from a single article"""
        chunks = []
        article_number = article['article_number']
        content = self.clean_legal_text(article['content'])

        # Base metadata for all chunks from this article
        base_metadata = {
            'article_number': article_number,
            'article_title': article['title'],
            'chapter': article['chapter'],
            'chapter_number': article['chapter_number'],
            'source_url': article['url'],
            'law_name': '食品安全衛生管理法',
            'law_code': 'L0040001'
        }

        # Extract article structure
        structure = self.extract_article_structure(content)

        # Create chunk for main provision
        if structure['main_provision']:
            main_text = f"第{article_number}條 {structure['main_provision']}"
            chunks.append(LegalChunk(
                text=main_text,
                metadata={
                    **base_metadata,
                    'section_type': 'main_provision',
                    'chunk_type': 'article_main'
                },
                chunk_id=f"art_{article_number}_main",
                article_number=article_number,
                chapter=article['chapter']
            ))

        # Create chunks for items
        if structure['items']:
            items_text = f"第{article_number}條規定項目:\n" + '\n'.join(structure['items'])
            if self.count_tokens(items_text) > self.chunk_size:
                # Split items if too large
                item_chunks = self._split_large_text(items_text, f"art_{article_number}_items")
                for i, chunk_text in enumerate(item_chunks):
                    chunks.append(LegalChunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            'section_type': 'items',
                            'chunk_type': 'article_items',
                            'item_chunk_index': i
                        },
                        chunk_id=f"art_{article_number}_items_{i}",
                        article_number=article_number,
                        chapter=article['chapter']
                    ))
            else:
                chunks.append(LegalChunk(
                    text=items_text,
                    metadata={
                        **base_metadata,
                        'section_type': 'items',
                        'chunk_type': 'article_items'
                    },
                    chunk_id=f"art_{article_number}_items",
                    article_number=article_number,
                    chapter=article['chapter']
                ))

        # Create chunk for exceptions
        if structure['exceptions']:
            exception_text = f"第{article_number}條例外規定:\n{structure['exceptions']}"
            chunks.append(LegalChunk(
                text=exception_text,
                metadata={
                    **base_metadata,
                    'section_type': 'exceptions',
                    'chunk_type': 'article_exceptions'
                },
                chunk_id=f"art_{article_number}_exceptions",
                article_number=article_number,
                chapter=article['chapter']
            ))

        # Create chunk for penalties
        if structure['penalties']:
            penalty_text = f"第{article_number}條罰則:\n{structure['penalties']}"
            chunks.append(LegalChunk(
                text=penalty_text,
                metadata={
                    **base_metadata,
                    'section_type': 'penalties',
                    'chunk_type': 'article_penalties'
                },
                chunk_id=f"art_{article_number}_penalties",
                article_number=article_number,
                chapter=article['chapter']
            ))

        # Fallback: create single chunk if no structure detected
        if not chunks:
            full_text = f"第{article_number}條 {content}"
            chunks.append(LegalChunk(
                text=full_text,
                metadata={
                    **base_metadata,
                    'section_type': 'full_article',
                    'chunk_type': 'article_complete'
                },
                chunk_id=f"art_{article_number}_full",
                article_number=article_number,
                chapter=article['chapter']
            ))

        return chunks

    def _split_large_text(self, text: str, base_id: str) -> List[str]:
        """Split large text into smaller chunks with overlap"""
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(tokens):
                break

            chunk_index += 1

        return chunks

    def process_all_articles(self, data: Dict[str, Any]) -> List[LegalChunk]:
        """Process all articles into chunks"""
        all_chunks = []

        for article in data['articles']:
            article_chunks = self.create_article_chunks(article)
            all_chunks.extend(article_chunks)

        return all_chunks

    def convert_to_llama_documents(self, chunks: List[LegalChunk]) -> List[Document]:
        """Convert legal chunks to LlamaIndex Document format"""
        documents = []

        for chunk in chunks:
            # Create enhanced metadata for LlamaIndex
            metadata = {
                **chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'text_length': len(chunk.text),
                'token_count': self.count_tokens(chunk.text)
            }

            doc = Document(
                text=chunk.text,
                metadata=metadata,
                id_=chunk.chunk_id
            )
            documents.append(doc)

        return documents

    def create_text_nodes(self, chunks: List[LegalChunk]) -> List[TextNode]:
        """Convert legal chunks to LlamaIndex TextNode format"""
        nodes = []

        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'text_length': len(chunk.text),
                'token_count': self.count_tokens(chunk.text)
            }

            node = TextNode(
                text=chunk.text,
                metadata=metadata,
                id_=chunk.chunk_id
            )
            nodes.append(node)

        return nodes

    def get_processing_stats(self, chunks: List[LegalChunk]) -> Dict[str, Any]:
        """Get statistics about processed chunks"""
        stats = {
            'total_chunks': len(chunks),
            'total_tokens': sum(self.count_tokens(chunk.text) for chunk in chunks),
            'avg_chunk_size': 0,
            'chunks_by_type': {},
            'chunks_by_chapter': {},
            'articles_processed': len(set(chunk.article_number for chunk in chunks))
        }

        if chunks:
            stats['avg_chunk_size'] = stats['total_tokens'] / len(chunks)

        # Count by type
        for chunk in chunks:
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            stats['chunks_by_type'][chunk_type] = stats['chunks_by_type'].get(chunk_type, 0) + 1

        # Count by chapter
        for chunk in chunks:
            chapter = chunk.chapter
            stats['chunks_by_chapter'][chapter] = stats['chunks_by_chapter'].get(chapter, 0) + 1

        return stats


def main():
    """Example usage"""
    processor = LegalDocumentProcessor(chunk_size=512, chunk_overlap=50)

    # Load and process the food safety act
    try:
        data = processor.load_legal_data('data/food_safety_act.json')
        print(f"Loaded data for {data['law_name']} with {data['total_articles']} articles")

        # Process into chunks
        chunks = processor.process_all_articles(data)
        print(f"Created {len(chunks)} chunks")

        # Get statistics
        stats = processor.get_processing_stats(chunks)
        print(f"\nProcessing Statistics:")
        print(f"- Total articles processed: {stats['articles_processed']}")
        print(f"- Total chunks created: {stats['total_chunks']}")
        print(f"- Average chunk size: {stats['avg_chunk_size']:.1f} tokens")
        print(f"- Chunks by type: {stats['chunks_by_type']}")

        # Show sample chunks
        print(f"\nSample chunks:")
        for chunk in chunks[:3]:
            print(f"- {chunk.chunk_id}")
            print(f"  Type: {chunk.metadata['chunk_type']}")
            print(f"  Preview: {chunk.text[:100]}...")
            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()