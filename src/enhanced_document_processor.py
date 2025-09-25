"""
Enhanced Document Processor for Legal Documents
Implements advanced strategies: cross-reference graph, semantic chunking, hierarchical indexing
"""

import json
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import tiktoken
from collections import defaultdict, Counter
# Simple graph implementation for cross-references
class SimpleGraph:
    """Simple directed graph implementation"""
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, from_node, to_node, **attrs):
        if from_node not in self.edges:
            self.edges[from_node] = {}
        self.edges[from_node][to_node] = attrs

    def neighbors(self, node_id):
        return list(self.edges.get(node_id, {}).keys())

    def predecessors(self, node_id):
        predecessors = []
        for source, targets in self.edges.items():
            if node_id in targets:
                predecessors.append(source)
        return predecessors

    def number_of_nodes(self):
        return len(self.nodes)

    def number_of_edges(self):
        return sum(len(targets) for targets in self.edges.values())

    def clear(self):
        self.nodes.clear()
        self.edges.clear()

    def __contains__(self, node_id):
        return node_id in self.nodes

    def __getitem__(self, from_node):
        class EdgeProxy:
            def __init__(self, edges, from_node):
                self.edges = edges
                self.from_node = from_node

            def __getitem__(self, to_node):
                return self.edges.get(self.from_node, {}).get(to_node, {})

        return EdgeProxy(self.edges, from_node)

from llama_index.core import Document
from llama_index.core.schema import TextNode

from .document_processor import LegalDocumentProcessor, LegalChunk


@dataclass
class CrossReference:
    """Represents a cross-reference between articles"""
    from_article: str
    to_article: str
    reference_type: str  # "direct", "indirect", "penalty", "definition"
    context: str
    confidence: float


@dataclass
class SemanticChunk(LegalChunk):
    """Enhanced chunk with semantic information"""
    semantic_keywords: List[str] = field(default_factory=list)
    cross_references: List[CrossReference] = field(default_factory=list)
    importance_score: float = 0.0
    hierarchical_level: int = 0  # 0=main law, 1=chapter, 2=article, 3=item
    related_concepts: List[str] = field(default_factory=list)


class EnhancedLegalProcessor(LegalDocumentProcessor):
    """Enhanced processor with advanced NLP strategies"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)

        # Cross-reference patterns for Taiwan legal documents
        self.reference_patterns = {
            'direct': [
                r'第\s*(\d+)\s*條',  # 第X條
                r'前\s*條',          # 前條
                r'本\s*條',          # 本條
                r'次\s*條',          # 次條
            ],
            'penalty': [
                r'違反.*第\s*(\d+)\s*條',  # 違反第X條
                r'依.*第\s*(\d+)\s*條.*處',  # 依第X條處罰
            ],
            'definition': [
                r'本法所稱',         # 本法所稱
                r'前項所稱',         # 前項所稱
                r'本條所稱',         # 本條所稱
            ]
        }

        # Legal concept keywords for semantic analysis
        self.legal_concepts = {
            '勞動契約': ['勞動契約', '契約', '僱傭', '聘僱', '試用期', '定期契約', '不定期契約'],
            '工資': ['工資', '薪資', '薪水', '報酬', '基本工資', '最低工資'],
            '工時': ['工時', '工作時間', '正常工時', '延長工時', '加班', '超時工作'],
            '休假': ['休假', '特別休假', '年假', '病假', '事假', '產假', '陪產假'],
            '退休': ['退休', '退休金', '勞工退休', '退休準備金'],
            '職災': ['職業災害', '職災', '工傷', '職業病', '工安', '安全衛生'],
            '解僱': ['解僱', '終止契約', '資遣', '不當解僱', '預告期間'],
            '工會': ['工會', '團體協約', '勞資爭議', '集體談判'],
            '檢查': ['勞動檢查', '檢查', '違反', '處罰', '罰鍰'],
        }

        # Relationship graph for cross-references
        self.reference_graph = SimpleGraph()

    def extract_cross_references(self, article_number: str, content: str) -> List[CrossReference]:
        """Extract cross-references from article content"""
        references = []

        for ref_type, patterns in self.reference_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    if ref_type == 'direct' and match.group(1):
                        # Direct article reference
                        to_article = match.group(1)
                        if to_article != article_number:  # Avoid self-reference
                            context = self._extract_context(content, match.start(), match.end())
                            ref = CrossReference(
                                from_article=article_number,
                                to_article=to_article,
                                reference_type=ref_type,
                                context=context,
                                confidence=0.9
                            )
                            references.append(ref)

                    elif ref_type == 'penalty':
                        # Penalty clause reference
                        to_article = match.group(1) if match.lastindex else 'unknown'
                        context = self._extract_context(content, match.start(), match.end())
                        ref = CrossReference(
                            from_article=article_number,
                            to_article=to_article,
                            reference_type=ref_type,
                            context=context,
                            confidence=0.8
                        )
                        references.append(ref)

                    elif ref_type == 'definition':
                        # Definition reference
                        context = self._extract_context(content, match.start(), match.end())
                        ref = CrossReference(
                            from_article=article_number,
                            to_article='definition',
                            reference_type=ref_type,
                            context=context,
                            confidence=0.7
                        )
                        references.append(ref)

        return references

    def _extract_context(self, content: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a match"""
        context_start = max(0, start - window)
        context_end = min(len(content), end + window)
        return content[context_start:context_end].strip()

    def extract_semantic_keywords(self, content: str) -> Tuple[List[str], List[str]]:
        """Extract semantic keywords and related concepts"""
        content_lower = content.lower()
        found_keywords = []
        related_concepts = []

        for concept, keywords in self.legal_concepts.items():
            for keyword in keywords:
                if keyword in content_lower:
                    if concept not in related_concepts:
                        related_concepts.append(concept)
                    if keyword not in found_keywords:
                        found_keywords.append(keyword)

        # Extract additional keywords using frequency analysis
        words = re.findall(r'[\u4e00-\u9fff]+', content)  # Chinese characters only
        word_freq = Counter(words)

        # Add frequent meaningful words
        for word, freq in word_freq.most_common(10):
            if len(word) >= 2 and freq >= 2 and word not in found_keywords:
                found_keywords.append(word)

        return found_keywords, related_concepts

    def calculate_importance_score(self, article: Dict[str, Any], all_articles: List[Dict[str, Any]]) -> float:
        """Calculate importance score for an article"""
        score = 0.0
        content = article.get('content', '')

        # Base score from article type
        if '總則' in article.get('chapter', ''):
            score += 0.3  # General provisions are important

        # Score from cross-references (how often referenced by others)
        article_num = article.get('article_number', '')
        reference_count = 0
        for other_article in all_articles:
            if article_num != other_article.get('article_number', ''):
                other_content = other_article.get('content', '')
                if f'第{article_num}條' in other_content:
                    reference_count += 1

        score += min(reference_count * 0.1, 0.5)  # Cap at 0.5

        # Score from penalty clauses (enforcement articles are important)
        if any(keyword in content for keyword in ['處', '罰', '元', '刑']):
            score += 0.2

        # Score from key legal concepts
        for concept_keywords in self.legal_concepts.values():
            concept_found = any(keyword in content for keyword in concept_keywords)
            if concept_found:
                score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def determine_hierarchical_level(self, chunk_type: str, chapter: str) -> int:
        """Determine hierarchical level for chunk"""
        level_mapping = {
            'law_title': 0,
            'chapter_title': 1,
            'article_main': 2,
            'article_items': 3,
            'article_exceptions': 3,
            'article_penalties': 2,  # Penalties are important
            'article_complete': 2
        }
        return level_mapping.get(chunk_type, 2)

    def create_semantic_chunks(self, articles: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """Create enhanced semantic chunks with cross-references and metadata"""
        semantic_chunks = []

        # First pass: create basic chunks and extract cross-references
        for article in articles:
            article_number = article['article_number']
            content = self.clean_legal_text(article['content'])

            # Extract semantic information
            keywords, concepts = self.extract_semantic_keywords(content)
            cross_refs = self.extract_cross_references(article_number, content)
            importance = self.calculate_importance_score(article, articles)

            # Create base chunk metadata
            base_metadata = {
                'article_number': article_number,
                'article_title': article['title'],
                'chapter': article['chapter'],
                'chapter_number': article['chapter_number'],
                'source_url': article['url'],
                'law_name': article.get('law_name', '勞動基準法'),
                'law_code': article.get('law_code', 'N0030001')
            }

            # Extract article structure for detailed chunking
            structure = self.extract_article_structure(content)

            # Create main provision chunk
            if structure['main_provision']:
                main_text = f"第{article_number}條 {structure['main_provision']}"
                chunk = SemanticChunk(
                    text=main_text,
                    metadata={
                        **base_metadata,
                        'section_type': 'main_provision',
                        'chunk_type': 'article_main'
                    },
                    chunk_id=f"art_{article_number}_main",
                    article_number=article_number,
                    chapter=article['chapter'],
                    semantic_keywords=keywords,
                    cross_references=cross_refs,
                    importance_score=importance,
                    hierarchical_level=self.determine_hierarchical_level('article_main', article['chapter']),
                    related_concepts=concepts
                )
                semantic_chunks.append(chunk)

            # Create items chunks
            if structure['items']:
                items_text = f"第{article_number}條規定項目:\n" + '\n'.join(structure['items'])
                if self.count_tokens(items_text) > self.chunk_size:
                    # Split large items
                    item_chunks = self._split_large_text(items_text, f"art_{article_number}_items")
                    for i, chunk_text in enumerate(item_chunks):
                        chunk = SemanticChunk(
                            text=chunk_text,
                            metadata={
                                **base_metadata,
                                'section_type': 'items',
                                'chunk_type': 'article_items',
                                'item_chunk_index': i
                            },
                            chunk_id=f"art_{article_number}_items_{i}",
                            article_number=article_number,
                            chapter=article['chapter'],
                            semantic_keywords=keywords,
                            cross_references=cross_refs,
                            importance_score=importance * 0.8,  # Items are slightly less important
                            hierarchical_level=self.determine_hierarchical_level('article_items', article['chapter']),
                            related_concepts=concepts
                        )
                        semantic_chunks.append(chunk)
                else:
                    chunk = SemanticChunk(
                        text=items_text,
                        metadata={
                            **base_metadata,
                            'section_type': 'items',
                            'chunk_type': 'article_items'
                        },
                        chunk_id=f"art_{article_number}_items",
                        article_number=article_number,
                        chapter=article['chapter'],
                        semantic_keywords=keywords,
                        cross_references=cross_refs,
                        importance_score=importance * 0.8,
                        hierarchical_level=self.determine_hierarchical_level('article_items', article['chapter']),
                        related_concepts=concepts
                    )
                    semantic_chunks.append(chunk)

            # Create penalties chunk (high importance)
            if structure['penalties']:
                penalty_text = f"第{article_number}條罰則:\n{structure['penalties']}"
                chunk = SemanticChunk(
                    text=penalty_text,
                    metadata={
                        **base_metadata,
                        'section_type': 'penalties',
                        'chunk_type': 'article_penalties'
                    },
                    chunk_id=f"art_{article_number}_penalties",
                    article_number=article_number,
                    chapter=article['chapter'],
                    semantic_keywords=keywords + ['處罰', '罰鍰', '刑責'],
                    cross_references=cross_refs,
                    importance_score=min(importance + 0.3, 1.0),  # Penalties are very important
                    hierarchical_level=self.determine_hierarchical_level('article_penalties', article['chapter']),
                    related_concepts=concepts + ['處罰']
                )
                semantic_chunks.append(chunk)

            # Create exceptions chunk
            if structure['exceptions']:
                exception_text = f"第{article_number}條例外規定:\n{structure['exceptions']}"
                chunk = SemanticChunk(
                    text=exception_text,
                    metadata={
                        **base_metadata,
                        'section_type': 'exceptions',
                        'chunk_type': 'article_exceptions'
                    },
                    chunk_id=f"art_{article_number}_exceptions",
                    article_number=article_number,
                    chapter=article['chapter'],
                    semantic_keywords=keywords + ['例外', '但是'],
                    cross_references=cross_refs,
                    importance_score=importance * 0.9,
                    hierarchical_level=self.determine_hierarchical_level('article_exceptions', article['chapter']),
                    related_concepts=concepts
                )
                semantic_chunks.append(chunk)

            # Fallback: create complete chunk if no structure detected
            if not any([structure['main_provision'], structure['items'], structure['penalties'], structure['exceptions']]):
                full_text = f"第{article_number}條 {content}"
                chunk = SemanticChunk(
                    text=full_text,
                    metadata={
                        **base_metadata,
                        'section_type': 'full_article',
                        'chunk_type': 'article_complete'
                    },
                    chunk_id=f"art_{article_number}_full",
                    article_number=article_number,
                    chapter=article['chapter'],
                    semantic_keywords=keywords,
                    cross_references=cross_refs,
                    importance_score=importance,
                    hierarchical_level=self.determine_hierarchical_level('article_complete', article['chapter']),
                    related_concepts=concepts
                )
                semantic_chunks.append(chunk)

        # Build cross-reference graph
        self._build_reference_graph(semantic_chunks)

        return semantic_chunks

    def _build_reference_graph(self, chunks: List[SemanticChunk]):
        """Build cross-reference graph from chunks"""
        self.reference_graph.clear()

        # Add nodes
        for chunk in chunks:
            self.reference_graph.add_node(
                chunk.article_number,
                importance=chunk.importance_score,
                concepts=chunk.related_concepts,
                chunk_id=chunk.chunk_id
            )

        # Add edges from cross-references
        for chunk in chunks:
            for cross_ref in chunk.cross_references:
                if cross_ref.to_article != 'unknown' and cross_ref.to_article != 'definition':
                    self.reference_graph.add_edge(
                        cross_ref.from_article,
                        cross_ref.to_article,
                        type=cross_ref.reference_type,
                        confidence=cross_ref.confidence,
                        context=cross_ref.context
                    )

    def get_related_articles(self, article_number: str, max_related: int = 5) -> List[Dict[str, Any]]:
        """Get related articles using graph analysis"""
        if article_number not in self.reference_graph:
            return []

        related = []

        # Direct references (outgoing edges)
        for target in self.reference_graph.neighbors(article_number):
            edge_data = self.reference_graph[article_number][target]
            related.append({
                'article_number': target,
                'relationship': 'references',
                'type': edge_data.get('type', 'unknown'),
                'confidence': edge_data.get('confidence', 0.5),
                'context': edge_data.get('context', '')
            })

        # Reverse references (incoming edges)
        for source in self.reference_graph.predecessors(article_number):
            edge_data = self.reference_graph[source][article_number]
            related.append({
                'article_number': source,
                'relationship': 'referenced_by',
                'type': edge_data.get('type', 'unknown'),
                'confidence': edge_data.get('confidence', 0.5),
                'context': edge_data.get('context', '')
            })

        # Sort by confidence and limit
        related.sort(key=lambda x: x['confidence'], reverse=True)
        return related[:max_related]

    def get_processing_stats(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Get enhanced processing statistics"""
        base_stats = super().get_processing_stats(chunks)

        # Add semantic statistics
        total_keywords = sum(len(chunk.semantic_keywords) for chunk in chunks)
        total_concepts = sum(len(chunk.related_concepts) for chunk in chunks)
        total_cross_refs = sum(len(chunk.cross_references) for chunk in chunks)

        concept_distribution = defaultdict(int)
        for chunk in chunks:
            for concept in chunk.related_concepts:
                concept_distribution[concept] += 1

        importance_distribution = [chunk.importance_score for chunk in chunks]
        avg_importance = sum(importance_distribution) / len(importance_distribution) if importance_distribution else 0

        enhanced_stats = {
            **base_stats,
            'total_keywords': total_keywords,
            'avg_keywords_per_chunk': total_keywords / len(chunks) if chunks else 0,
            'total_concepts': total_concepts,
            'avg_concepts_per_chunk': total_concepts / len(chunks) if chunks else 0,
            'total_cross_references': total_cross_refs,
            'avg_cross_refs_per_chunk': total_cross_refs / len(chunks) if chunks else 0,
            'concept_distribution': dict(concept_distribution),
            'avg_importance_score': avg_importance,
            'high_importance_chunks': len([c for c in chunks if c.importance_score > 0.7]),
            'graph_nodes': self.reference_graph.number_of_nodes(),
            'graph_edges': self.reference_graph.number_of_edges()
        }

        return enhanced_stats

    def convert_to_llama_documents(self, chunks: List[SemanticChunk]) -> List[Document]:
        """Convert semantic chunks to LlamaIndex Document format with enhanced metadata"""
        import json
        documents = []

        for chunk in chunks:
            # Create comprehensive metadata (serialize complex types for ChromaDB compatibility)
            metadata = {
                **chunk.metadata,
                'chunk_id': chunk.chunk_id,
                'text_length': len(chunk.text),
                'token_count': self.count_tokens(chunk.text),
                'semantic_keywords': json.dumps(chunk.semantic_keywords, ensure_ascii=False),
                'related_concepts': json.dumps(chunk.related_concepts, ensure_ascii=False),
                'importance_score': chunk.importance_score,
                'hierarchical_level': chunk.hierarchical_level,
                'cross_reference_count': len(chunk.cross_references),
                'cross_reference_types': json.dumps(list(set(ref.reference_type for ref in chunk.cross_references)), ensure_ascii=False)
            }

            doc = Document(
                text=chunk.text,
                metadata=metadata,
                id_=chunk.chunk_id
            )
            documents.append(doc)

        return documents


def main():
    """Example usage"""
    processor = EnhancedLegalProcessor(chunk_size=512, chunk_overlap=50)

    # Test with labor law data
    try:
        if hasattr(processor, 'load_legal_data'):
            # Try loading existing data
            data_files = ['data/labor_standards_act.json', 'data/food_safety_act.json']

            for data_file in data_files:
                try:
                    print(f"\nTesting with {data_file}")
                    data = processor.load_legal_data(data_file)
                    print(f"Loaded {data['law_name']} with {data['total_articles']} articles")

                    # Process first 3 articles for testing
                    test_articles = data['articles'][:3]
                    chunks = processor.create_semantic_chunks(test_articles)

                    print(f"Created {len(chunks)} semantic chunks")

                    # Show statistics
                    stats = processor.get_processing_stats(chunks)
                    print(f"\nEnhanced Processing Statistics:")
                    print(f"- Total keywords: {stats['total_keywords']}")
                    print(f"- Total concepts: {stats['total_concepts']}")
                    print(f"- Total cross-references: {stats['total_cross_references']}")
                    print(f"- Average importance score: {stats['avg_importance_score']:.3f}")
                    print(f"- High importance chunks: {stats['high_importance_chunks']}")
                    print(f"- Reference graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")

                    # Show concept distribution
                    print(f"\nConcept Distribution:")
                    for concept, count in stats['concept_distribution'].items():
                        print(f"  - {concept}: {count}")

                    # Test cross-reference graph
                    if chunks:
                        sample_article = chunks[0].article_number
                        related = processor.get_related_articles(sample_article)
                        print(f"\nRelated articles for Article {sample_article}:")
                        for rel in related:
                            print(f"  - Article {rel['article_number']} ({rel['relationship']}, confidence: {rel['confidence']:.2f})")

                    break  # Exit after successful test

                except FileNotFoundError:
                    print(f"Data file not found: {data_file}")
                    continue

        else:
            print("Base processor methods not available")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()