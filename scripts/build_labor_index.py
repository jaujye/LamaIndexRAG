"""
Build Labor Law ChromaDB index using enhanced processing
"""
from src.enhanced_document_processor import EnhancedLegalProcessor
from src.index_builder import LegalIndexBuilder
import json
import os

def build_labor_law_index():
    """Build ChromaDB index for labor law data"""
    print("Building Labor Law ChromaDB Index...")

    # Check if sample data exists
    data_file = 'data/labor_law_sample.json'
    if not os.path.exists(data_file):
        print(f"[ERROR] Data file not found: {data_file}")
        print("Please run test_labor_data.py first to fetch sample data")
        return False

    try:
        # Initialize enhanced processor
        print("Initializing enhanced document processor...")
        processor = EnhancedLegalProcessor(chunk_size=512, chunk_overlap=50)

        # Load labor law data
        print("Loading labor law data...")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Loaded {data['law_name']} with {data['total_articles']} articles")

        # Process with enhanced features
        print("Processing articles with enhanced semantic analysis...")
        chunks = processor.create_semantic_chunks(data['articles'])
        print(f"Created {len(chunks)} semantic chunks")

        # Show enhanced statistics
        stats = processor.get_processing_stats(chunks)
        print("\nEnhanced Processing Statistics:")
        print(f"- Total keywords: {stats['total_keywords']}")
        print(f"- Total concepts: {stats['total_concepts']}")
        print(f"- Total cross-references: {stats['total_cross_references']}")
        print(f"- Average importance score: {stats['avg_importance_score']:.3f}")
        print(f"- High importance chunks: {stats['high_importance_chunks']}")
        print(f"- Reference graph: {stats['graph_nodes']} nodes, {stats['graph_edges']} edges")

        # Initialize index builder for labor law collection
        print("\nInitializing ChromaDB index builder...")
        index_builder = LegalIndexBuilder(
            collection_name="labor_law",  # New collection for labor law
            enable_monitoring=False  # Disable monitoring to avoid encoding issues
        )

        # Convert to documents
        print("Converting chunks to LlamaIndex documents...")
        documents = processor.convert_to_llama_documents(chunks)
        print(f"Converted {len(documents)} documents")

        # Create ChromaDB collection and build index
        print("\nBuilding vector index and storing in ChromaDB...")
        collection = index_builder.create_collection(reset=True)
        print(f"Created ChromaDB collection: {collection.name}")

        # Setup vector store and build index
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import VectorStoreIndex, StorageContext

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Building vector index...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )

        # Save metadata
        metadata = {
            'law_name': data['law_name'],
            'law_code': data['law_code'],
            'source_url': data['source_url'],
            'total_articles': data['total_articles'],
            'collection_name': 'labor_law',
            'processing_stats': stats,
            'enhanced_features': {
                'semantic_keywords': True,
                'cross_references': True,
                'importance_scoring': True,
                'hierarchical_indexing': True
            }
        }

        with open('data/labor_law_index_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print("\n[SUCCESS] Labor Law ChromaDB index created successfully!")
        print(f"Collection: labor_law")
        print(f"Documents: {len(documents)}")
        print(f"Metadata saved: data/labor_law_index_metadata.json")

        # Test basic query
        print("\nTesting basic query...")
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("勞動契約的規定")
        print(f"Query: 勞動契約的規定")
        print(f"Response preview: {str(response)[:200]}...")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = build_labor_law_index()
    if success:
        print("\n[OK] Labor Law ChromaDB index build completed!")
    else:
        print("\n[FAIL] Index build failed!")