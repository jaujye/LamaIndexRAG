"""
CLI Renderer Module
Handles all Rich console UI display logic for the Legal RAG CLI.
"""

import sys
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ..cli.environment_validator import ValidationResult


class CLIRenderer:
    """
    Responsible for all Rich console UI display operations.

    Handles:
    - Welcome messages and menus
    - Index statistics display
    - Query result rendering
    - Multi-domain result display
    - Environment validation results
    - Interactive prompts
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize CLI renderer with Rich console.

        Args:
            console: Optional Rich Console instance. If not provided, creates a new one
                    with Windows UTF-8 compatibility handling.
        """
        if console is None:
            # Initialize console with UTF-8 for Windows compatibility
            try:
                console = Console(force_terminal=True, width=None)
                # Try to configure Rich console for UTF-8 on Windows
                if sys.platform == "win32":
                    console._file = sys.stdout
            except Exception:
                console = Console()

        self.console = console

    def display_welcome(self, enable_monitoring: bool = False):
        """
        Display welcome banner for the Legal RAG system.

        Args:
            enable_monitoring: Whether W&B monitoring is enabled
        """
        self.console.print("\n" + "="*60)
        panel_text = "[bold blue]台灣法規 RAG 知識檢索系統[/bold blue]\n" \
                    "輸入您的問題，系統將基於法規內容為您解答\n" \
                    "[dim]輸入 'quit' 或 'exit' 結束程式[/dim]"

        if enable_monitoring:
            panel_text += "\n[dim]W&B 監控已啟用[/dim]"

        self.console.print(Panel.fit(panel_text, border_style="blue"))

    def display_multi_domain_welcome(self, enable_monitoring: bool = False):
        """
        Display welcome banner for multi-domain Legal RAG system.

        Args:
            enable_monitoring: Whether W&B monitoring is enabled
        """
        self.console.print("\n" + "="*60)
        panel_text = "[bold blue]台灣多法規整合 RAG 知識檢索系統[/bold blue]\n" \
                    "支持食品安全法、勞基法、民法及跨法規查詢\n" \
                    "系統將智能路由您的問題到最適合的法規知識庫\n" \
                    "[dim]輸入 'quit' 或 'exit' 結束程式[/dim]"

        if enable_monitoring:
            panel_text += "\n[dim]W&B 監控已啟用[/dim]"

        self.console.print(Panel.fit(panel_text, border_style="blue"))

    def display_index_stats(self, stats: dict, law_name: str = ""):
        """
        Display index statistics for a legal document collection.

        Args:
            stats: Dictionary containing index statistics with keys like:
                   - document_count: Number of documents in index
                   - embedding_model: Model used for embeddings
                   - law_name: Name of the law
                   - processing_stats: Processing statistics dict
            law_name: Name of the law for display title
        """
        title = f"{law_name}索引統計資訊" if law_name else "索引統計資訊"
        table = Table(title=title)
        table.add_column("項目", style="cyan")
        table.add_column("數值", style="green")

        # Basic statistics
        table.add_row("文件總數", str(stats.get("document_count", "未知")))
        table.add_row("嵌入模型", stats.get("embedding_model", "未知"))
        table.add_row("法規名稱", stats.get("law_name", "未知"))

        # Processing statistics
        if "processing_stats" in stats:
            ps = stats["processing_stats"]
            table.add_row("處理的條文數", str(ps.get("articles_processed", "未知")))
            table.add_row("總chunk數", str(ps.get("total_chunks", "未知")))
            table.add_row("平均chunk大小", f"{ps.get('avg_chunk_size', 0):.1f} tokens")

        self.console.print(table)

    def display_enhanced_stats(self, stats: dict, law_name: str = ""):
        """
        Display enhanced processing statistics for legal documents.

        Args:
            stats: Dictionary containing enhanced processing statistics with keys like:
                   - total_keywords: Total keywords extracted
                   - total_concepts: Total concepts identified
                   - total_cross_references: Total cross-references found
                   - avg_importance_score: Average importance score
                   - high_importance_chunks: Number of high importance chunks
                   - graph_nodes: Number of nodes in reference graph
                   - graph_edges: Number of edges in reference graph
            law_name: Name of the law for display title
        """
        title = f"{law_name}增強處理統計" if law_name else "增強處理統計"
        table = Table(title=title)
        table.add_column("項目", style="cyan")
        table.add_column("數值", style="green")

        table.add_row("總關鍵字數", str(stats.get("total_keywords", 0)))
        table.add_row("總概念數", str(stats.get("total_concepts", 0)))
        table.add_row("總交叉引用數", str(stats.get("total_cross_references", 0)))
        table.add_row("平均重要性分數", f"{stats.get('avg_importance_score', 0):.3f}")
        table.add_row("高重要性塊數", str(stats.get("high_importance_chunks", 0)))
        table.add_row("引用圖節點數", str(stats.get("graph_nodes", 0)))
        table.add_row("引用圖邊數", str(stats.get("graph_edges", 0)))

        self.console.print(table)

    def display_query_result(self, result):
        """
        Display single-domain query result with answer and sources.

        Args:
            result: QueryResult object with attributes:
                    - answer: The generated answer
                    - confidence_score: Confidence score (0-1)
                    - query_type: Type of query
                    - sources: List of source documents with metadata
        """
        # Display answer
        self.console.print(Panel(
            result.answer,
            title="[bold green]法規解答[/bold green]",
            border_style="green"
        ))

        # Display confidence and query type
        confidence_color = "green" if result.confidence_score > 0.7 else "yellow" if result.confidence_score > 0.5 else "red"
        self.console.print(f"\n[{confidence_color}]信心度: {result.confidence_score:.3f}[/{confidence_color}] | 查詢類型: [blue]{result.query_type}[/blue]")

        # Display relevant sources
        if result.sources:
            self.console.print("\n[bold cyan]相關法條:[/bold cyan]")

            sources_table = Table()
            sources_table.add_column("條文", style="cyan")
            sources_table.add_column("相似度", style="yellow")
            sources_table.add_column("類型", style="green")
            sources_table.add_column("內容預覽", style="white")

            # Display top 3 most relevant sources
            for source in result.sources[:3]:
                similarity = f"{source['similarity_score']:.3f}"
                preview = source['text_preview'][:100] + "..." if len(source['text_preview']) > 100 else source['text_preview']

                sources_table.add_row(
                    f"第{source['article_number']}條",
                    similarity,
                    source['chunk_type'].replace('article_', ''),
                    preview
                )

            self.console.print(sources_table)

    def display_multi_domain_result(self, response):
        """
        Display multi-domain query result with routing information and fused response.

        Args:
            response: QueryResponse object with attributes:
                      - route_decision: RouteDecision with primary_kb, confidence_score, etc.
                      - fused_response: The fused answer from multiple domains
                      - responses: Dict of KB name -> KB response
        """
        # Display routing information
        kb_name = response.route_decision.primary_kb.value if response.route_decision.primary_kb else "conversational"
        self.console.print(f"\n[dim]查詢路由: {kb_name} " +
                          f"(信心度: {response.route_decision.confidence_score:.2f})[/dim]")

        if response.route_decision.secondary_kbs:
            secondary = ", ".join([kb.value for kb in response.route_decision.secondary_kbs])
            self.console.print(f"[dim]次要查詢: {secondary}[/dim]")

        # Display fused answer
        if response.fused_response:
            self.console.print(Panel(
                response.fused_response,
                title="[bold green]法規解答[/bold green]",
                border_style="green"
            ))
        else:
            self.console.print("[yellow]未找到相關資訊[/yellow]")

        # Display per-KB results summary if multiple KBs were queried
        if len(response.responses) > 1:
            meaningful_results = self._extract_meaningful_results(response.responses)

            # Only display table if there are meaningful results
            if meaningful_results:
                self.console.print("\n[bold cyan]各法規查詢結果:[/bold cyan]")

                results_table = Table()
                results_table.add_column("法規", style="cyan")
                results_table.add_column("相關文檔數", style="yellow")
                results_table.add_column("答案預覽", style="white")

                for result in meaningful_results:
                    results_table.add_row(
                        result['name'],
                        str(result['count']),
                        result['preview']
                    )

                self.console.print(results_table)
            elif self._count_total_docs(response.responses) == 0:
                # No documents found at all
                self.console.print("\n[yellow]查詢結果: 查詢不到任何相關法規結果[/yellow]")

    def _extract_meaningful_results(self, responses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract meaningful results from KB responses.

        Args:
            responses: Dictionary of KB name -> response

        Returns:
            List of dictionaries with meaningful result information
        """
        meaningful_results = []
        law_name_mapping = {
            "labor_law": "勞基法",
            "food_safety_act": "食品安全法",
            "civil_law": "民法"
        }

        for kb_name, kb_response in responses.items():
            if "error" not in kb_response:
                law_name = law_name_mapping.get(kb_name, "未知法規")
                doc_count = len(kb_response.get('metadata', {}).get('retrieved_nodes', []))
                answer_preview = kb_response.get('response', '')

                # Check if answer is meaningful (not a generic "not found" message)
                is_meaningful = (
                    doc_count > 0 and
                    answer_preview and
                    not answer_preview.startswith("I'm sorry, but based on the provided context") and
                    not answer_preview.startswith("很抱歉，根據提供的") and
                    len(answer_preview.strip()) > 20
                )

                if is_meaningful:
                    meaningful_results.append({
                        'name': law_name,
                        'count': doc_count,
                        'preview': answer_preview[:100] + "..." if len(answer_preview) > 100 else answer_preview
                    })

        return meaningful_results

    def _count_total_docs(self, responses: Dict[str, Any]) -> int:
        """
        Count total documents retrieved across all KB responses.

        Args:
            responses: Dictionary of KB name -> response

        Returns:
            Total document count
        """
        total_docs = 0
        for kb_response in responses.values():
            if "error" not in kb_response:
                doc_count = len(kb_response.get('metadata', {}).get('retrieved_nodes', []))
                total_docs += doc_count
        return total_docs

    def display_validation_result(self, result: ValidationResult):
        """
        Display environment validation results.

        Args:
            result: ValidationResult object with passed status and issues list
        """
        if result.passed:
            self.console.print("[green]環境檢查通過[/green]")
        else:
            self.console.print("\n[red]環境設置問題：[/red]")
            for issue in result.issues:
                self.console.print(f"  {issue}")

    def display_system_header(self, title: str = "台灣法規 RAG 知識檢索系統"):
        """
        Display system header/title.

        Args:
            title: Title text to display
        """
        self.console.print(f"[bold blue]{title}[/bold blue]\n")

    def display_success(self, message: str):
        """Display success message."""
        self.console.print(f"[green]{message}[/green]")

    def display_error(self, message: str):
        """Display error message."""
        self.console.print(f"[red]{message}[/red]")

    def display_warning(self, message: str):
        """Display warning message."""
        self.console.print(f"[yellow]{message}[/yellow]")

    def display_info(self, message: str):
        """Display info message."""
        self.console.print(f"[cyan]{message}[/cyan]")

    def display_searching(self):
        """Display searching indicator."""
        self.console.print("\n[yellow]搜尋相關法規...[/yellow]")

    def display_multi_domain_searching(self):
        """Display multi-domain searching indicator."""
        self.console.print("\n[yellow]智能路由查詢到相關法規...[/yellow]")

    def display_goodbye(self, message: str = "感謝使用法規知識檢索系統！"):
        """
        Display goodbye message.

        Args:
            message: Goodbye message text
        """
        self.console.print(f"\n[green]{message}[/green]")