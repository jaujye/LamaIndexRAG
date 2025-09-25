"""
UltraThink System Performance Test & Optimization
Comprehensive testing suite for the Advanced Legal RAG System
"""

import time
import json
from typing import Dict, List, Any
from datetime import datetime
import statistics

from src.query_router import QueryRouter, LegalDomainClassifier, KnowledgeBase
from src.legal_multi_domain_rag import LegalQueryExpander, AdvancedRAGSystem
from src.legal_enhanced_processor import EnhancedLegalProcessor


class SystemPerformanceTester:
    """Comprehensive performance testing for UltraThink system"""

    def __init__(self):
        self.test_results = {
            'component_tests': {},
            'integration_tests': {},
            'performance_metrics': {},
            'edge_case_tests': {},
            'optimization_recommendations': []
        }
        self.query_router = None

    def run_comprehensive_tests(self):
        """Run all system tests"""
        print("UltraThink System Performance Test Suite")
        print("="*60)
        print("Testing all system components and integration...\n")

        # Component-level tests
        self.test_query_expansion()
        self.test_domain_classification()
        self.test_document_processing()

        # Integration tests
        self.test_query_routing_integration()
        self.test_cross_domain_scenarios()

        # Performance tests
        self.test_response_times()
        self.test_concurrent_queries()

        # Edge cases
        self.test_edge_cases()

        # Generate optimization recommendations
        self.generate_optimization_report()

        return self.test_results

    def test_query_expansion(self):
        """Test query expansion functionality"""
        print("Testing Query Expansion...")
        print("-" * 40)

        expander = LegalQueryExpander()
        test_queries = [
            ("勞動契約", ["勞動契約", "工作契約", "僱傭契約"]),
            ("工資", ["工資", "薪資", "薪水"]),
            ("食品安全", ["食品安全", "食安", "食品衛生"]),
            ("什麼是工作時間", ["定義"]),  # Intent test
            ("違反規定的處罰", ["處罰"])   # Penalty intent
        ]

        results = []
        for query, expected_elements in test_queries:
            start_time = time.time()
            context = expander.expand_query(query)
            processing_time = time.time() - start_time

            # Check if expected elements are found
            found_elements = 0
            for element in expected_elements:
                if element in context.expanded_terms or element in context.legal_concepts:
                    found_elements += 1

            accuracy = found_elements / len(expected_elements) if expected_elements else 1.0

            result = {
                'query': query,
                'processing_time': processing_time * 1000,  # Convert to ms
                'accuracy': accuracy,
                'expanded_terms_count': len(context.expanded_terms),
                'legal_concepts': context.legal_concepts,
                'intent_detected': context.intent_type
            }
            results.append(result)

            print(f"  Query: {query}")
            print(f"    Processing: {processing_time*1000:.2f}ms")
            print(f"    Accuracy: {accuracy:.2f}")
            print(f"    Expanded Terms: {len(context.expanded_terms)}")
            print(f"    Intent: {context.intent_type}")

        avg_time = statistics.mean([r['processing_time'] for r in results])
        avg_accuracy = statistics.mean([r['accuracy'] for r in results])

        self.test_results['component_tests']['query_expansion'] = {
            'status': 'PASS' if avg_accuracy > 0.7 else 'FAIL',
            'average_processing_time_ms': avg_time,
            'average_accuracy': avg_accuracy,
            'total_tests': len(results),
            'individual_results': results
        }

        print(f"    Overall: {avg_time:.2f}ms avg, {avg_accuracy:.2f} accuracy\n")

    def test_domain_classification(self):
        """Test domain classification accuracy"""
        print("Testing Domain Classification...")
        print("-" * 40)

        classifier = LegalDomainClassifier()
        expander = LegalQueryExpander()

        # Test cases with expected domains
        test_cases = [
            ("勞動契約的規定", KnowledgeBase.LABOR_LAW),
            ("食品添加物標示", KnowledgeBase.FOOD_SAFETY),
            ("餐廳員工工作時間", KnowledgeBase.ALL),  # Cross-domain
            ("什麼是職業安全", KnowledgeBase.ALL),    # Definition - multi-domain
            ("勞工食品安全訓練", KnowledgeBase.ALL),  # Mixed keywords
        ]

        results = []
        for query, expected_kb in test_cases:
            start_time = time.time()
            context = expander.expand_query(query)
            route_decision = classifier.classify_query(query, context)
            processing_time = time.time() - start_time

            # Check classification accuracy
            correct = (
                route_decision.primary_kb == expected_kb or
                (expected_kb == KnowledgeBase.ALL and
                 (route_decision.primary_kb == KnowledgeBase.ALL or
                  len(route_decision.secondary_kbs) > 0))
            )

            result = {
                'query': query,
                'expected_kb': expected_kb.value,
                'actual_kb': route_decision.primary_kb.value,
                'secondary_kbs': [kb.value for kb in route_decision.secondary_kbs],
                'confidence': route_decision.confidence_score,
                'processing_time': processing_time * 1000,
                'correct': correct
            }
            results.append(result)

            print(f"  Query: {query}")
            print(f"    Expected: {expected_kb.value}")
            print(f"    Actual: {route_decision.primary_kb.value}")
            print(f"    Confidence: {route_decision.confidence_score:.2f}")
            print(f"    Correct: {'YES' if correct else 'NO'}")

        accuracy = sum(1 for r in results if r['correct']) / len(results)
        avg_time = statistics.mean([r['processing_time'] for r in results])
        avg_confidence = statistics.mean([r['confidence'] for r in results])

        self.test_results['component_tests']['domain_classification'] = {
            'status': 'PASS' if accuracy >= 0.8 else 'FAIL',
            'accuracy': accuracy,
            'average_processing_time_ms': avg_time,
            'average_confidence': avg_confidence,
            'total_tests': len(results),
            'individual_results': results
        }

        print(f"    Overall: {accuracy:.2f} accuracy, {avg_confidence:.2f} avg confidence\n")

    def test_document_processing(self):
        """Test enhanced document processing"""
        print("Testing Enhanced Document Processing...")
        print("-" * 40)

        # Sample legal article for testing
        sample_article = {
            'article_number': '9',
            'title': '第九條 勞動契約之規定',
            'content': '勞動契約，分為定期契約及不定期契約。臨時性、短期性、季節性及特定性工作得為定期契約；有繼續性工作應為不定期契約。定期契約屆滿後，有下列情形之一者，視為不定期契約：一、勞工繼續工作而雇主不即表示反對意思者。',
            'chapter': '第二章 勞動契約',
            'chapter_number': '二',
            'url': 'https://law.moj.gov.tw/test'
        }

        try:
            processor = EnhancedLegalProcessor()

            start_time = time.time()
            chunks = processor.create_semantic_chunks([sample_article])
            processing_time = time.time() - start_time

            if chunks:
                chunk = chunks[0]

                # Check processing quality
                has_keywords = len(chunk.semantic_keywords) > 0
                has_concepts = len(chunk.related_concepts) > 0
                has_score = chunk.importance_score > 0

                result = {
                    'processing_time': processing_time * 1000,
                    'chunks_created': len(chunks),
                    'has_keywords': has_keywords,
                    'keyword_count': len(chunk.semantic_keywords),
                    'has_concepts': has_concepts,
                    'concept_count': len(chunk.related_concepts),
                    'importance_score': chunk.importance_score,
                    'has_cross_references': len(chunk.cross_references) > 0
                }

                print(f"  Processing Time: {processing_time*1000:.2f}ms")
                print(f"  Chunks Created: {len(chunks)}")
                print(f"  Keywords Extracted: {len(chunk.semantic_keywords)}")
                print(f"  Concepts Identified: {len(chunk.related_concepts)}")
                print(f"  Importance Score: {chunk.importance_score:.3f}")
                print(f"  Cross References: {len(chunk.cross_references)}")

                quality_score = sum([has_keywords, has_concepts, has_score]) / 3

                self.test_results['component_tests']['document_processing'] = {
                    'status': 'PASS' if quality_score > 0.5 else 'FAIL',
                    'quality_score': quality_score,
                    'processing_time_ms': processing_time * 1000,
                    'details': result
                }

            else:
                self.test_results['component_tests']['document_processing'] = {
                    'status': 'FAIL',
                    'error': 'No chunks created'
                }

        except Exception as e:
            self.test_results['component_tests']['document_processing'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"  ERROR: {e}")

        print()

    def test_query_routing_integration(self):
        """Test integrated query routing"""
        print("Testing Query Routing Integration...")
        print("-" * 40)

        try:
            self.query_router = QueryRouter()

            test_queries = [
                "勞動契約的定義",
                "食品添加物標示規定",
                "餐廳員工的安全規定",
                "違反法規的處罰"
            ]

            results = []
            for query in test_queries:
                start_time = time.time()
                response = self.query_router.route_query(query, top_k=3)
                processing_time = time.time() - start_time

                result = {
                    'query': query,
                    'processing_time': processing_time * 1000,
                    'primary_kb': response.route_decision.primary_kb.value,
                    'confidence': response.route_decision.confidence_score,
                    'has_response': bool(response.fused_response),
                    'kbs_queried': response.metadata.get('total_kbs_queried', 0)
                }
                results.append(result)

                print(f"  Query: {query}")
                print(f"    Time: {processing_time*1000:.2f}ms")
                print(f"    Route: {response.route_decision.primary_kb.value}")
                print(f"    Confidence: {response.route_decision.confidence_score:.2f}")
                print(f"    KBs Queried: {response.metadata.get('total_kbs_queried', 0)}")

            avg_time = statistics.mean([r['processing_time'] for r in results])
            responses_generated = sum(1 for r in results if r['has_response'])

            self.test_results['integration_tests']['query_routing'] = {
                'status': 'PASS' if responses_generated >= len(results) * 0.5 else 'PARTIAL',
                'average_processing_time_ms': avg_time,
                'response_success_rate': responses_generated / len(results),
                'total_tests': len(results),
                'results': results
            }

            print(f"    Overall: {avg_time:.2f}ms avg, {responses_generated}/{len(results)} responses\n")

        except Exception as e:
            self.test_results['integration_tests']['query_routing'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"  ERROR: {e}\n")

    def test_cross_domain_scenarios(self):
        """Test cross-domain query scenarios"""
        print("Testing Cross-Domain Scenarios...")
        print("-" * 40)

        if not self.query_router:
            print("  Skipped: Query Router not available\n")
            return

        cross_domain_queries = [
            "食品工廠員工的工作安全規定",
            "餐廳業者與員工的勞動關係",
            "食品製造業的職業安全衛生"
        ]

        results = []
        for query in cross_domain_queries:
            try:
                start_time = time.time()
                response = self.query_router.route_query(query, top_k=3)
                processing_time = time.time() - start_time

                is_cross_domain = (
                    response.route_decision.primary_kb == KnowledgeBase.ALL or
                    len(response.route_decision.secondary_kbs) > 0
                )

                result = {
                    'query': query,
                    'processing_time': processing_time * 1000,
                    'detected_cross_domain': is_cross_domain,
                    'kbs_queried': response.metadata.get('total_kbs_queried', 0),
                    'fusion_applied': response.metadata.get('fusion_strategy') is not None
                }
                results.append(result)

                print(f"  Query: {query}")
                print(f"    Cross-domain: {'YES' if is_cross_domain else 'NO'}")
                print(f"    KBs Queried: {response.metadata.get('total_kbs_queried', 0)}")
                print(f"    Fusion: {'YES' if result['fusion_applied'] else 'NO'}")

            except Exception as e:
                print(f"  ERROR in query '{query}': {e}")

        cross_domain_detection_rate = sum(1 for r in results if r['detected_cross_domain']) / len(results)

        self.test_results['integration_tests']['cross_domain'] = {
            'status': 'PASS' if cross_domain_detection_rate >= 0.8 else 'FAIL',
            'detection_rate': cross_domain_detection_rate,
            'total_tests': len(results),
            'results': results
        }

        print(f"    Cross-domain detection: {cross_domain_detection_rate:.2f}\n")

    def test_response_times(self):
        """Test system response times"""
        print("Testing Response Time Performance...")
        print("-" * 40)

        if not self.query_router:
            print("  Skipped: Query Router not available\n")
            return

        # Test with various query complexities
        simple_queries = ["工資", "食品安全"]
        complex_queries = ["餐廳員工的工作時間與食品安全管理規定", "勞動契約終止後的食品業者責任"]

        all_times = []

        for query_set, description in [(simple_queries, "Simple"), (complex_queries, "Complex")]:
            times = []
            for query in query_set:
                try:
                    start_time = time.time()
                    self.query_router.route_query(query, top_k=3)
                    processing_time = time.time() - start_time
                    times.append(processing_time * 1000)
                    all_times.append(processing_time * 1000)
                except Exception as e:
                    print(f"    ERROR in '{query}': {e}")

            if times:
                avg_time = statistics.mean(times)
                print(f"  {description} Queries: {avg_time:.2f}ms average")

        if all_times:
            overall_avg = statistics.mean(all_times)
            overall_std = statistics.stdev(all_times) if len(all_times) > 1 else 0

            self.test_results['performance_metrics']['response_times'] = {
                'average_ms': overall_avg,
                'std_deviation_ms': overall_std,
                'min_ms': min(all_times),
                'max_ms': max(all_times),
                'total_queries': len(all_times)
            }

            print(f"  Overall: {overall_avg:.2f}ms ± {overall_std:.2f}ms")
            print(f"  Range: {min(all_times):.2f}ms - {max(all_times):.2f}ms\n")

    def test_concurrent_queries(self):
        """Test concurrent query handling (simulated)"""
        print("Testing Concurrent Query Simulation...")
        print("-" * 40)

        if not self.query_router:
            print("  Skipped: Query Router not available\n")
            return

        # Simulate rapid sequential queries
        queries = [
            "勞動契約",
            "食品標示",
            "工作時間",
            "添加物規定",
            "職業安全"
        ]

        start_time = time.time()
        successful_queries = 0

        for query in queries:
            try:
                self.query_router.route_query(query, top_k=2)
                successful_queries += 1
            except Exception as e:
                print(f"    Failed query '{query}': {e}")

        total_time = time.time() - start_time
        throughput = successful_queries / total_time if total_time > 0 else 0

        self.test_results['performance_metrics']['concurrent_queries'] = {
            'total_queries': len(queries),
            'successful_queries': successful_queries,
            'total_time_seconds': total_time,
            'throughput_qps': throughput
        }

        print(f"  Queries: {successful_queries}/{len(queries)} successful")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} queries/sec\n")

    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("Testing Edge Cases...")
        print("-" * 40)

        edge_cases = [
            ("", "Empty query"),
            ("   ", "Whitespace only"),
            ("a", "Single character"),
            ("123", "Numbers only"),
            ("!@#$%", "Special characters only"),
            ("這是一個非常長的查詢句子包含了很多不相關的詞語和內容用來測試系統的處理能力", "Very long query")
        ]

        results = []
        for query, description in edge_cases:
            try:
                if self.query_router:
                    response = self.query_router.route_query(query)
                    has_response = bool(response.fused_response)
                    has_error = "error" in response.responses
                else:
                    # Test components individually
                    expander = LegalQueryExpander()
                    context = expander.expand_query(query)
                    has_response = bool(context.expanded_terms)
                    has_error = False

                result = {
                    'query': query,
                    'description': description,
                    'handled_gracefully': True,
                    'has_response': has_response,
                    'has_error': has_error
                }

                print(f"  {description}: {'OK' if not has_error else 'ERROR'}")

            except Exception as e:
                result = {
                    'query': query,
                    'description': description,
                    'handled_gracefully': False,
                    'error': str(e)
                }
                print(f"  {description}: EXCEPTION - {e}")

            results.append(result)

        graceful_handling_rate = sum(1 for r in results if r['handled_gracefully']) / len(results)

        self.test_results['edge_case_tests'] = {
            'graceful_handling_rate': graceful_handling_rate,
            'total_tests': len(results),
            'results': results
        }

        print(f"    Graceful handling: {graceful_handling_rate:.2f}\n")

    def generate_optimization_report(self):
        """Generate optimization recommendations"""
        print("Generating Optimization Report...")
        print("="*60)

        recommendations = []

        # Performance optimizations
        if 'response_times' in self.test_results['performance_metrics']:
            avg_time = self.test_results['performance_metrics']['response_times']['average_ms']
            if avg_time > 1000:  # > 1 second
                recommendations.append({
                    'type': 'Performance',
                    'priority': 'High',
                    'issue': f'Average response time is {avg_time:.2f}ms',
                    'recommendation': 'Consider implementing result caching and optimizing embedding model'
                })
            elif avg_time > 500:  # > 0.5 second
                recommendations.append({
                    'type': 'Performance',
                    'priority': 'Medium',
                    'issue': f'Response time could be improved ({avg_time:.2f}ms)',
                    'recommendation': 'Consider query preprocessing and result caching'
                })

        # Component optimizations
        for component, results in self.test_results['component_tests'].items():
            if results.get('status') == 'FAIL':
                recommendations.append({
                    'type': 'Functionality',
                    'priority': 'High',
                    'issue': f'{component} component failed tests',
                    'recommendation': f'Debug and fix {component} component issues'
                })
            elif component == 'domain_classification' and results.get('accuracy', 0) < 0.9:
                recommendations.append({
                    'type': 'Accuracy',
                    'priority': 'Medium',
                    'issue': f'Domain classification accuracy is {results["accuracy"]:.2f}',
                    'recommendation': 'Expand keyword dictionary and improve classification patterns'
                })

        # System architecture recommendations
        recommendations.extend([
            {
                'type': 'Architecture',
                'priority': 'Medium',
                'issue': 'OpenAI embedding dependency',
                'recommendation': 'Implement local embedding model for better reliability and cost control'
            },
            {
                'type': 'Monitoring',
                'priority': 'Low',
                'issue': 'Limited performance monitoring',
                'recommendation': 'Add comprehensive logging and monitoring dashboard'
            },
            {
                'type': 'Scalability',
                'priority': 'Low',
                'issue': 'Single-threaded query processing',
                'recommendation': 'Implement async query processing for better concurrency'
            }
        ])

        self.test_results['optimization_recommendations'] = recommendations

        # Display recommendations
        for rec in recommendations:
            print(f"[{rec['priority'].upper()}] {rec['type']}")
            print(f"  Issue: {rec['issue']}")
            print(f"  Recommendation: {rec['recommendation']}\n")

        # Overall system health
        component_pass_rate = sum(1 for r in self.test_results['component_tests'].values()
                                 if r.get('status') == 'PASS') / len(self.test_results['component_tests'])

        print(f"Overall System Health: {component_pass_rate:.1%}")
        if component_pass_rate >= 0.8:
            print("Status: EXCELLENT - System is production ready")
        elif component_pass_rate >= 0.6:
            print("Status: GOOD - Minor optimizations needed")
        elif component_pass_rate >= 0.4:
            print("Status: FAIR - Significant improvements needed")
        else:
            print("Status: POOR - Major fixes required")


def main():
    """Run comprehensive system tests"""
    tester = SystemPerformanceTester()

    try:
        results = tester.run_comprehensive_tests()

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"system_test_results_{timestamp}.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nTest results saved to: {results_file}")
        print("="*60)

    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()