"""
Test script for Advanced RAG System
Tests Hybrid Search, Query Expansion, and Reranking
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.advanced_rag_system import AdvancedRAGSystem, LegalQueryExpander
import json

def test_query_expansion():
    """Test query expansion functionality"""
    print("Testing Query Expansion...")

    expander = LegalQueryExpander()

    test_queries = [
        "勞動契約的規定",
        "食品添加物標示",
        "什麼是工作時間",
        "違反食品安全法的處罰"
    ]

    for query in test_queries:
        context = expander.expand_query(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded: {context.expanded_terms[:5]}")
        print(f"Intent: {context.intent_type}")
        print(f"Target Collections: {context.target_collections}")
        print(f"Legal Concepts: {context.legal_concepts}")


def test_advanced_rag():
    """Test the complete Advanced RAG System"""
    print("\n" + "="*60)
    print("Testing Advanced RAG System")
    print("="*60)

    # Initialize the system
    rag_system = AdvancedRAGSystem()

    # Test queries covering different legal domains and intent types
    test_queries = [
        {
            "query": "勞動契約的規定",
            "description": "Labor contract provisions (should target labor_law)"
        },
        {
            "query": "食品添加物的標示要求",
            "description": "Food additive labeling (should target food_safety)"
        },
        {
            "query": "違反勞基法的罰則是什麼",
            "description": "Labor law penalties (intent: penalty)"
        },
        {
            "query": "什麼是工作時間",
            "description": "Definition of working hours (intent: definition)"
        },
        {
            "query": "如何申請特別休假",
            "description": "Special leave procedure (intent: procedure)"
        },
        {
            "query": "食品安全衛生管理",
            "description": "General food safety management"
        }
    ]

    for test_case in test_queries:
        query = test_case["query"]
        description = test_case["description"]

        print(f"\n{'-'*50}")
        print(f"Test: {description}")
        print(f"Query: {query}")
        print(f"{'-'*50}")

        # Test with reranking enabled
        result = rag_system.query(
            query=query,
            top_k=3,
            enable_reranking=True,
            response_mode="compact"
        )

        if "error" in result:
            print(f"[ERROR] {result['error']}")
        else:
            # Display query analysis
            context = result["query_context"]
            print(f"Intent Type: {context['intent_type']}")
            print(f"Target Collections: {context['target_collections']}")
            print(f"Expanded Terms: {', '.join(context['expanded_terms'][:5])}")

            # Display response
            print(f"\nGenerated Response:")
            print(f"{result['response'][:300]}...")

            # Display top retrieved documents
            print(f"\nTop {len(result['retrieved_documents'])} Retrieved Documents:")
            for i, doc in enumerate(result['retrieved_documents'], 1):
                print(f"  {i}. Score: {doc['score']:.3f}")
                print(f"     Preview: {doc['text_preview'][:150]}...")

                # Show relevant metadata
                metadata = doc['metadata']
                article_num = metadata.get('article_number', 'N/A')
                section_type = metadata.get('section_type', 'N/A')
                print(f"     Article: {article_num}, Type: {section_type}")
                print()

            # Show system performance metadata
            metadata = result["metadata"]
            print(f"System Stats:")
            print(f"  - Total Retrieved: {metadata['total_retrieved']}")
            print(f"  - Hybrid Search: {metadata['hybrid_search']}")
            print(f"  - Reranking: {metadata['reranking_enabled']}")


def test_hybrid_vs_semantic():
    """Compare hybrid search vs semantic-only search"""
    print("\n" + "="*60)
    print("Comparing Hybrid vs Semantic-Only Search")
    print("="*60)

    rag_system = AdvancedRAGSystem()

    test_query = "勞動契約終止的規定"

    print(f"Query: {test_query}")

    # Test with hybrid search (default)
    print(f"\n{'-'*30}")
    print("Hybrid Search Results:")
    print(f"{'-'*30}")

    hybrid_result = rag_system.query(test_query, top_k=3)

    if "error" not in hybrid_result:
        for i, doc in enumerate(hybrid_result['retrieved_documents'], 1):
            print(f"{i}. Score: {doc['score']:.3f} - {doc['text_preview'][:100]}...")

    # For comparison, we'd need to implement semantic-only mode
    # This is just a demonstration of how to analyze different approaches


def main():
    """Run all tests"""
    print("Advanced RAG System Test Suite")
    print("="*60)

    try:
        # Test individual components
        test_query_expansion()

        # Test complete system
        test_advanced_rag()

        # Test comparisons
        test_hybrid_vs_semantic()

        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()