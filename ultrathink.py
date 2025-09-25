"""
UltraThink - Advanced Legal RAG System
Main entry point integrating all components:
- Query Router with intelligent domain classification
- Advanced RAG optimization (Hybrid Search, Query Expansion, Reranking)
- Multi-domain knowledge bases (Labor Law, Food Safety)
- Enhanced document processing with semantic analysis
"""

import sys
import json
import argparse
from typing import Dict, Any, List
import time
from datetime import datetime

# Import our core systems
from src.query_router import QueryRouter, KnowledgeBase
from src.advanced_rag_system import AdvancedRAGSystem
from src.enhanced_document_processor import EnhancedLegalProcessor


class UltraThinkRAG:
    """
    UltraThink Advanced Legal RAG System
    Unified interface for intelligent legal document querying
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.version = "1.0.0"
        self.system_name = "UltraThink Legal RAG"

        # System components
        self.query_router = None
        self.query_history = []
        self.session_stats = {
            'queries_processed': 0,
            'total_response_time': 0.0,
            'kb_usage': {'labor_law': 0, 'food_safety_act': 0, 'multi_domain': 0},
            'start_time': datetime.now()
        }

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize all system components"""
        print(f"Initializing {self.system_name} v{self.version}...")
        print("="*60)

        # Initialize Query Router (this initializes everything else)
        try:
            chroma_host = self.config.get('chroma_host', '192.168.0.114')
            chroma_port = self.config.get('chroma_port', 7000)

            print("Starting Query Router with Advanced RAG System...")
            self.query_router = QueryRouter(
                chroma_host=chroma_host,
                chroma_port=chroma_port
            )

            print("[OK] Query Router initialized successfully")
            print("[OK] Advanced RAG System ready")
            print("[OK] Multi-domain knowledge bases loaded")

            # Display available knowledge bases
            self._display_system_info()

        except Exception as e:
            print(f"[ERROR] System initialization failed: {e}")
            sys.exit(1)

    def _display_system_info(self):
        """Display system capabilities and information"""
        print("\n" + "="*60)
        print("SYSTEM CAPABILITIES")
        print("="*60)

        print("\nKnowledge Bases:")
        print("  - Labor Law (勞動基準法): 11 documents")
        print("  - Food Safety (食品安全法): 104 documents")

        print("\nAdvanced RAG Features:")
        print("  - Intelligent Query Routing")
        print("  - Hybrid Search (Semantic + Keyword)")
        print("  - Query Expansion with Legal Synonyms")
        print("  - Cross-Reference Analysis")
        print("  - Importance-Based Reranking")
        print("  - Multi-Domain Response Fusion")

        print("\nSupported Query Types:")
        print("  - Definition queries (什麼是...)")
        print("  - Procedure queries (如何...)")
        print("  - Penalty queries (違反...處罰)")
        print("  - Cross-domain queries")
        print("  - Comparative legal analysis")

        print("\n" + "="*60)

    def query(
        self,
        question: str,
        top_k: int = 5,
        show_routing: bool = True,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a legal query through the advanced RAG system

        Args:
            question: The legal question to answer
            top_k: Number of top results to retrieve
            show_routing: Whether to show routing decision details
            show_sources: Whether to show source documents

        Returns:
            Comprehensive response with answer and metadata
        """
        start_time = time.time()

        try:
            # Route and process query
            response = self.query_router.route_query(
                query=question,
                top_k=top_k,
                fusion_strategy='domain_weighted'
            )

            processing_time = time.time() - start_time

            # Update session statistics
            self._update_session_stats(response, processing_time)

            # Format response
            formatted_response = self._format_response(
                response,
                processing_time,
                show_routing=show_routing,
                show_sources=show_sources
            )

            # Store in query history
            self.query_history.append({
                'question': question,
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'primary_kb': response.route_decision.primary_kb.value,
                'confidence': response.route_decision.confidence_score
            })

            return formatted_response

        except Exception as e:
            return {
                'error': True,
                'message': f'Query processing failed: {e}',
                'question': question,
                'timestamp': datetime.now().isoformat()
            }

    def _update_session_stats(self, response, processing_time: float):
        """Update session statistics"""
        self.session_stats['queries_processed'] += 1
        self.session_stats['total_response_time'] += processing_time

        # Update KB usage stats
        if response.route_decision.primary_kb == KnowledgeBase.ALL:
            self.session_stats['kb_usage']['multi_domain'] += 1
        else:
            kb_name = response.route_decision.primary_kb.value
            if kb_name in self.session_stats['kb_usage']:
                self.session_stats['kb_usage'][kb_name] += 1

    def _format_response(
        self,
        response,
        processing_time: float,
        show_routing: bool = True,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """Format the response for display"""

        formatted = {
            'question': response.query,
            'answer': response.fused_response or "No answer generated",
            'processing_time': f"{processing_time:.2f}s",
            'timestamp': datetime.now().isoformat()
        }

        # Add routing information
        if show_routing:
            formatted['routing'] = {
                'primary_knowledge_base': response.route_decision.primary_kb.value,
                'secondary_knowledge_bases': [kb.value for kb in response.route_decision.secondary_kbs],
                'confidence_score': response.route_decision.confidence_score,
                'reasoning': response.route_decision.reasoning,
                'query_intent': response.route_decision.query_context.intent_type,
                'expanded_terms': response.route_decision.query_context.expanded_terms[:5]
            }

        # Add source information
        if show_sources and response.responses:
            formatted['sources'] = {}
            for kb_name, kb_response in response.responses.items():
                if 'retrieved_documents' in kb_response:
                    formatted['sources'][kb_name] = [
                        {
                            'rank': doc['rank'],
                            'score': doc['score'],
                            'article': doc['metadata'].get('article_number', 'N/A'),
                            'title': doc['metadata'].get('article_title', 'N/A'),
                            'preview': doc['text_preview'][:150] + "..."
                        }
                        for doc in kb_response['retrieved_documents'][:3]  # Top 3 sources
                    ]

        # Add system metadata
        formatted['system_metadata'] = {
            'total_kbs_queried': response.metadata.get('total_kbs_queried', 0),
            'fusion_strategy': response.metadata.get('fusion_strategy'),
            'system_version': self.version
        }

        return formatted

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        session_duration = datetime.now() - self.session_stats['start_time']
        avg_response_time = (
            self.session_stats['total_response_time'] /
            max(1, self.session_stats['queries_processed'])
        )

        return {
            'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
            'queries_processed': self.session_stats['queries_processed'],
            'average_response_time': f"{avg_response_time:.2f}s",
            'kb_usage_distribution': self.session_stats['kb_usage'],
            'recent_queries': [
                {
                    'question': q['question'][:50] + "..." if len(q['question']) > 50 else q['question'],
                    'primary_kb': q['primary_kb'],
                    'confidence': f"{q['confidence']:.2f}"
                }
                for q in self.query_history[-5:]  # Last 5 queries
            ]
        }

    def interactive_mode(self):
        """Run in interactive CLI mode"""
        print(f"\nWelcome to {self.system_name}!")
        print("Ask any legal questions about Labor Law or Food Safety.")
        print("Type 'help' for commands, 'quit' to exit.\n")

        while True:
            try:
                user_input = input("UltraThink> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                elif not user_input:
                    continue

                # Process the query
                print("\nProcessing your query...")
                response = self.query(user_input)

                if response.get('error'):
                    print(f"[ERROR] {response['message']}")
                else:
                    self._display_response(response)

                print("\n" + "-"*60 + "\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")

    def _show_help(self):
        """Show help information"""
        print("""
Available Commands:
  help     - Show this help message
  stats    - Show session statistics
  history  - Show recent query history
  quit/q   - Exit the system

Example Queries:
  勞動契約的規定是什麼
  食品添加物的標示要求
  餐廳員工的工作時間規定 (cross-domain)
  什麼是職業安全衛生 (definition)
  違反勞基法的處罰 (penalty)
        """)

    def _show_stats(self):
        """Show session statistics"""
        stats = self.get_session_stats()
        print(f"""
Session Statistics:
  Duration: {stats['session_duration']}
  Queries Processed: {stats['queries_processed']}
  Average Response Time: {stats['average_response_time']}

Knowledge Base Usage:
  Labor Law: {stats['kb_usage_distribution']['labor_law']} queries
  Food Safety: {stats['kb_usage_distribution']['food_safety_act']} queries
  Multi-Domain: {stats['kb_usage_distribution']['multi_domain']} queries
        """)

    def _show_history(self):
        """Show recent query history"""
        if not self.query_history:
            print("No queries in history yet.")
            return

        print("\nRecent Queries:")
        for i, query in enumerate(self.query_history[-5:], 1):
            print(f"  {i}. {query['question'][:60]}...")
            print(f"     KB: {query['primary_kb']}, Confidence: {query['confidence']:.2f}")

    def _display_response(self, response: Dict[str, Any]):
        """Display formatted response"""
        print(f"\n[ANSWER]")
        print(f"{response['answer']}\n")

        if 'routing' in response:
            routing = response['routing']
            print(f"[ROUTING INFO]")
            print(f"  Primary KB: {routing['primary_knowledge_base']}")
            if routing['secondary_knowledge_bases']:
                print(f"  Secondary KBs: {', '.join(routing['secondary_knowledge_bases'])}")
            print(f"  Confidence: {routing['confidence_score']:.2f}")
            print(f"  Intent: {routing['query_intent']}")

        if 'sources' in response:
            print(f"\n[TOP SOURCES]")
            for kb_name, sources in response['sources'].items():
                domain_name = "Labor Law" if kb_name == "labor_law" else "Food Safety"
                print(f"  From {domain_name}:")
                for source in sources:
                    print(f"    - Article {source['article']}: {source['preview']}")

        print(f"\n[PROCESSING TIME] {response['processing_time']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='UltraThink - Advanced Legal RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ultrathink.py                           # Interactive mode
  python ultrathink.py -q "勞動契約的規定"       # Single query
  python ultrathink.py --stats                   # Show system info only
        """
    )

    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query to process'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics only'
    )

    parser.add_argument(
        '--no-routing',
        action='store_true',
        help='Hide routing information in output'
    )

    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source documents in output'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to retrieve (default: 5)'
    )

    args = parser.parse_args()

    # Initialize system
    config = {}
    ultrathink = UltraThinkRAG(config)

    if args.stats:
        # Just show system info
        return

    if args.query:
        # Process single query
        response = ultrathink.query(
            args.query,
            top_k=args.top_k,
            show_routing=not args.no_routing,
            show_sources=not args.no_sources
        )

        if response.get('error'):
            print(f"Error: {response['message']}")
            sys.exit(1)
        else:
            ultrathink._display_response(response)
    else:
        # Interactive mode
        ultrathink.interactive_mode()


if __name__ == "__main__":
    main()