"""
Intelligent Query Router for Multi-Domain Legal RAG System
Routes queries to appropriate knowledge bases (Labor Law, Food Safety, etc.)
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict
import logging

# Import our advanced RAG system
from .advanced_rag_system import AdvancedRAGSystem, QueryContext, LegalQueryExpander


class KnowledgeBase(Enum):
    """Available knowledge bases"""
    LABOR_LAW = "labor_law"
    FOOD_SAFETY = "food_safety_act"
    ALL = "all"


@dataclass
class RouteDecision:
    """Query routing decision with confidence scores"""
    primary_kb: KnowledgeBase
    secondary_kbs: List[KnowledgeBase]
    confidence_score: float
    reasoning: str
    query_context: QueryContext
    requires_fusion: bool = False  # Whether to combine results from multiple KBs


@dataclass
class QueryResponse:
    """Unified response from query router"""
    query: str
    route_decision: RouteDecision
    responses: Dict[str, Any]  # KB name -> response
    fused_response: Optional[str] = None
    metadata: Dict[str, Any] = None


class LegalDomainClassifier:
    """Classifies queries into legal domains"""

    def __init__(self):
        # Domain-specific keywords and patterns
        self.domain_keywords = {
            KnowledgeBase.LABOR_LAW: {
                'primary': ['勞動', '勞工', '僱傭', '工資', '薪資', '工時', '工作時間',
                           '勞基法', '勞動契約', '解僱', '資遣', '特別休假', '加班',
                           '職業安全', '工傷', '勞保', '勞退', '工會'],
                'secondary': ['契約', '時間', '休假', '安全', '保險', '退休', '工作'],
                'legal_concepts': ['第四條', '第九條', '第三十條', '勞動基準法']
            },
            KnowledgeBase.FOOD_SAFETY: {
                'primary': ['食品', '食安', '食品安全', '添加物', '食品添加劑',
                           '標示', '標籤', '衛生', '食品衛生', '食品安全法',
                           '製造', '加工', '販售', '進口', '輸入'],
                'secondary': ['安全', '品質', '檢驗', '認證', '管理', '監督'],
                'legal_concepts': ['第三條', '第十五條', '食品安全衛生管理法']
            }
        }

        # Cross-domain concepts that might require multiple knowledge bases
        self.cross_domain_patterns = [
            r'.*(?:勞工|員工).*(?:食品|餐廳).*',  # Workers in food industry
            r'.*(?:食品|餐廳).*(?:勞動|工作).*',  # Food industry labor
            r'.*(?:職業安全|工安).*(?:食品|廚房).*',  # Food industry safety
            r'.*(?:法律|法規|規定).*(?:比較|對比).*'  # Legal comparisons
        ]

        # Intent-based routing preferences
        self.intent_preferences = {
            'definition': KnowledgeBase.ALL,  # Definitions might be in either domain
            'penalty': None,  # Route based on content
            'procedure': None,  # Route based on content
            'rights': KnowledgeBase.LABOR_LAW,  # Worker rights usually labor law
            'obligations': None  # Route based on content
        }

    def classify_query(self, query: str, query_context: QueryContext) -> RouteDecision:
        """Classify query and determine routing strategy"""

        # Check for cross-domain patterns first
        for pattern in self.cross_domain_patterns:
            if re.match(pattern, query):
                return RouteDecision(
                    primary_kb=KnowledgeBase.ALL,
                    secondary_kbs=[KnowledgeBase.LABOR_LAW, KnowledgeBase.FOOD_SAFETY],
                    confidence_score=0.9,
                    reasoning="Cross-domain query detected",
                    query_context=query_context,
                    requires_fusion=True
                )

        # Score each knowledge base
        scores = {}

        for kb, keywords in self.domain_keywords.items():
            score = 0.0

            # Primary keyword matches (high weight)
            for keyword in keywords['primary']:
                if keyword in query:
                    score += 2.0

            # Secondary keyword matches (medium weight)
            for keyword in keywords['secondary']:
                if keyword in query:
                    score += 0.5

            # Legal concept matches (high weight)
            for concept in keywords['legal_concepts']:
                if concept in query:
                    score += 1.5

            # Context-based scoring
            if hasattr(query_context, 'target_collections'):
                for collection in query_context.target_collections:
                    if collection == kb.value:
                        score += 1.0

            scores[kb] = score

        # Determine primary and secondary knowledge bases
        sorted_kbs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_kbs or sorted_kbs[0][1] == 0:
            # No clear domain match - use all knowledge bases
            return RouteDecision(
                primary_kb=KnowledgeBase.ALL,
                secondary_kbs=[KnowledgeBase.LABOR_LAW, KnowledgeBase.FOOD_SAFETY],
                confidence_score=0.3,
                reasoning="No clear domain detected, searching all knowledge bases",
                query_context=query_context,
                requires_fusion=True
            )

        primary_kb = sorted_kbs[0][0]
        primary_score = sorted_kbs[0][1]

        # Determine if secondary knowledge bases should be included
        secondary_kbs = []
        confidence = min(1.0, primary_score / 5.0)  # Normalize confidence

        # Include secondary if scores are close
        if len(sorted_kbs) > 1 and sorted_kbs[1][1] > primary_score * 0.3:
            secondary_kbs.append(sorted_kbs[1][0])
            confidence *= 0.8  # Reduce confidence for multi-KB queries

        # Intent-based adjustments
        if query_context.intent_type in self.intent_preferences:
            preferred_kb = self.intent_preferences[query_context.intent_type]
            if preferred_kb == KnowledgeBase.ALL:
                return RouteDecision(
                    primary_kb=KnowledgeBase.ALL,
                    secondary_kbs=[KnowledgeBase.LABOR_LAW, KnowledgeBase.FOOD_SAFETY],
                    confidence_score=0.8,
                    reasoning=f"Intent '{query_context.intent_type}' suggests multi-domain search",
                    query_context=query_context,
                    requires_fusion=True
                )

        return RouteDecision(
            primary_kb=primary_kb,
            secondary_kbs=secondary_kbs,
            confidence_score=confidence,
            reasoning=f"Domain classified as {primary_kb.value} (score: {primary_score:.2f})",
            query_context=query_context,
            requires_fusion=len(secondary_kbs) > 0
        )


class ResponseFusion:
    """Fuses responses from multiple knowledge bases"""

    def __init__(self):
        self.fusion_strategies = {
            'concatenate': self._concatenate_responses,
            'rank_merge': self._rank_merge_responses,
            'domain_weighted': self._domain_weighted_responses
        }

    def fuse_responses(
        self,
        responses: Dict[str, Any],
        route_decision: RouteDecision,
        strategy: str = 'domain_weighted'
    ) -> str:
        """Fuse multiple responses into a coherent answer"""

        if len(responses) == 1:
            return list(responses.values())[0].get('response', '')

        fusion_func = self.fusion_strategies.get(strategy, self._domain_weighted_responses)
        return fusion_func(responses, route_decision)

    def _concatenate_responses(self, responses: Dict[str, Any], route_decision: RouteDecision) -> str:
        """Simple concatenation of responses"""
        fused_parts = []

        for kb_name, response_data in responses.items():
            response_text = response_data.get('response', '')
            if response_text.strip():
                fused_parts.append(f"【{kb_name.upper()}】\n{response_text}")

        return "\n\n".join(fused_parts)

    def _rank_merge_responses(self, responses: Dict[str, Any], route_decision: RouteDecision) -> str:
        """Merge responses based on relevance ranking"""
        ranked_responses = []

        # Sort by confidence/relevance
        for kb_name, response_data in responses.items():
            response_text = response_data.get('response', '')
            if response_text.strip():
                # Use metadata to determine relevance
                metadata = response_data.get('metadata', {})
                total_retrieved = metadata.get('total_retrieved', 0)
                relevance_score = total_retrieved  # Simple relevance metric

                ranked_responses.append((relevance_score, kb_name, response_text))

        # Sort by relevance (descending)
        ranked_responses.sort(reverse=True)

        # Merge top responses
        if not ranked_responses:
            return "No relevant information found."

        if len(ranked_responses) == 1:
            return ranked_responses[0][2]

        # Combine top responses with clear attribution
        fused = f"根據相關法規，主要資訊如下：\n\n"

        for i, (score, kb_name, response_text) in enumerate(ranked_responses[:2]):
            domain_name = "勞動基準法" if kb_name == "labor_law" else "食品安全法"
            fused += f"{i+1}. 依據{domain_name}：\n{response_text}\n\n"

        return fused.strip()

    def _domain_weighted_responses(self, responses: Dict[str, Any], route_decision: RouteDecision) -> str:
        """Weight responses based on domain relevance"""

        if route_decision.primary_kb == KnowledgeBase.ALL:
            # Equal weighting for cross-domain queries
            return self._rank_merge_responses(responses, route_decision)

        primary_kb_name = route_decision.primary_kb.value

        # Prioritize primary knowledge base response
        if primary_kb_name in responses:
            primary_response = responses[primary_kb_name].get('response', '')

            # Check if we have significant secondary responses
            secondary_responses = []
            for kb_name, response_data in responses.items():
                if kb_name != primary_kb_name:
                    response_text = response_data.get('response', '')
                    if response_text.strip() and len(response_text) > 50:  # Significant response
                        secondary_responses.append((kb_name, response_text))

            if not secondary_responses:
                return primary_response

            # Combine primary with relevant secondary information
            domain_name = "勞動基準法" if primary_kb_name == "labor_law" else "食品安全法"
            fused = f"依據{domain_name}：\n{primary_response}"

            if secondary_responses:
                fused += "\n\n相關資訊："
                for kb_name, response_text in secondary_responses:
                    other_domain = "食品安全法" if kb_name == "food_safety_act" else "勞動基準法"
                    fused += f"\n- {other_domain}：{response_text[:200]}..."

            return fused

        # Fallback to rank merge if primary response not available
        return self._rank_merge_responses(responses, route_decision)


class QueryRouter:
    """Main query router orchestrating the multi-domain legal RAG system"""

    def __init__(self, chroma_host: str = "192.168.0.114", chroma_port: int = 7000):
        self.classifier = LegalDomainClassifier()
        self.query_expander = LegalQueryExpander()
        self.response_fusion = ResponseFusion()

        # Initialize RAG system
        self.rag_system = AdvancedRAGSystem(
            chroma_host=chroma_host,
            chroma_port=chroma_port
        )

        # Query statistics for optimization
        self.query_stats = defaultdict(int)
        self.routing_history = []

    def route_query(
        self,
        query: str,
        top_k: int = 5,
        fusion_strategy: str = 'domain_weighted'
    ) -> QueryResponse:
        """Route query to appropriate knowledge bases and return fused response"""

        try:
            # Expand query and get context
            query_context = self.query_expander.expand_query(query)

            # Classify and route
            route_decision = self.classifier.classify_query(query, query_context)

            # Track routing statistics
            self.query_stats[route_decision.primary_kb.value] += 1
            self.routing_history.append({
                'query': query,
                'primary_kb': route_decision.primary_kb.value,
                'confidence': route_decision.confidence_score
            })

            # Query appropriate knowledge bases
            responses = {}

            if route_decision.primary_kb == KnowledgeBase.ALL:
                # Query all available knowledge bases
                target_kbs = [KnowledgeBase.LABOR_LAW, KnowledgeBase.FOOD_SAFETY]
            else:
                # Query primary and any secondary knowledge bases
                target_kbs = [route_decision.primary_kb] + route_decision.secondary_kbs

            for kb in target_kbs:
                try:
                    # Modify query context for specific KB
                    kb_query_context = self._adapt_query_for_kb(query_context, kb)

                    # Query the RAG system
                    response = self.rag_system.query(
                        query=query,
                        top_k=top_k,
                        enable_reranking=True
                    )

                    if "error" not in response:
                        responses[kb.value] = response

                except Exception as e:
                    logging.warning(f"Failed to query {kb.value}: {e}")

            # Fuse responses if multiple knowledge bases were queried
            fused_response = None
            if len(responses) > 1 and route_decision.requires_fusion:
                fused_response = self.response_fusion.fuse_responses(
                    responses, route_decision, fusion_strategy
                )
            elif len(responses) == 1:
                # Single response - use as is
                single_response = list(responses.values())[0]
                fused_response = single_response.get('response', '')

            return QueryResponse(
                query=query,
                route_decision=route_decision,
                responses=responses,
                fused_response=fused_response,
                metadata={
                    'total_kbs_queried': len(responses),
                    'fusion_strategy': fusion_strategy if len(responses) > 1 else None,
                    'query_context': query_context.__dict__
                }
            )

        except Exception as e:
            logging.error(f"Query routing failed: {e}")
            return QueryResponse(
                query=query,
                route_decision=RouteDecision(
                    primary_kb=KnowledgeBase.ALL,
                    secondary_kbs=[],
                    confidence_score=0.0,
                    reasoning=f"Error: {e}",
                    query_context=QueryContext("", [], [], "error", [])
                ),
                responses={"error": {"error": str(e)}},
                fused_response=f"查詢處理時發生錯誤: {e}"
            )

    def _adapt_query_for_kb(self, query_context: QueryContext, kb: KnowledgeBase) -> QueryContext:
        """Adapt query context for specific knowledge base"""
        # For now, return the original context
        # Could be enhanced to modify expanded terms or targeting based on KB
        return query_context

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get query routing statistics"""
        return {
            'total_queries': sum(self.query_stats.values()),
            'kb_distribution': dict(self.query_stats),
            'recent_queries': self.routing_history[-10:],  # Last 10 queries
            'average_confidence': sum(h['confidence'] for h in self.routing_history[-20:]) / min(20, len(self.routing_history)) if self.routing_history else 0
        }


def main():
    """Example usage of the Query Router"""
    print("Initializing Query Router System...")
    router = QueryRouter()

    # Test queries for different scenarios
    test_queries = [
        # Single domain queries
        "勞動契約的規定是什麼",
        "食品添加物的標示要求",
        "工作時間的計算方式",

        # Cross-domain queries
        "餐廳員工的工作安全規定",
        "食品業勞工的工作時間規定",

        # General queries that might need both
        "什麼是職業安全衛生",
        "違法的處罰規定"
    ]

    print("\nTesting Query Router with different scenarios:")
    print("="*60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)

        response = router.route_query(query, top_k=3)

        print(f"Primary KB: {response.route_decision.primary_kb.value}")
        print(f"Secondary KBs: {[kb.value for kb in response.route_decision.secondary_kbs]}")
        print(f"Confidence: {response.route_decision.confidence_score:.2f}")
        print(f"Reasoning: {response.route_decision.reasoning}")
        print(f"KBs Queried: {len(response.responses)}")

        if response.fused_response:
            print(f"Response Preview: {response.fused_response[:200]}...")
        else:
            print("No response generated")

    # Show routing statistics
    print(f"\n{'='*60}")
    print("Query Router Statistics:")
    stats = router.get_routing_stats()
    print(f"Total Queries: {stats['total_queries']}")
    print(f"KB Distribution: {stats['kb_distribution']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")


if __name__ == "__main__":
    main()