"""
Data fetcher for Taiwan Civil Code (民法)
Extracts legal text from the Ministry of Justice website

Note: Civil Code uses "編" (books) instead of "章" (chapters) for major divisions
"""

from typing import List, Dict
from src.legal_base_fetcher import BaseLegalFetcher
from src.legal_models import LegalArticle


class CivilLawFetcher(BaseLegalFetcher):
    """
    Fetches and processes Taiwan Civil Code (民法)

    The Civil Code is organized into 5 main books (編):
    - 第一編 總則 (General Principles)
    - 第二編 債 (Obligations)
    - 第三編 物權 (Property Rights)
    - 第四編 親屬 (Family)
    - 第五編 繼承 (Succession)
    """

    def __init__(self, delay: float = 1.5):
        """
        Initialize Civil Code fetcher.

        Args:
            delay: Delay between requests (default: 1.5s due to 1229 articles)
        """
        super().__init__(
            pcode="B0000001",
            law_name="民法",
            delay=delay
        )

    def get_section_pattern(self) -> str:
        """
        Override to match "編" (books) instead of "章" (chapters).

        Returns:
            Regex pattern for book headers: 第一編, 第二編, etc.
        """
        return r'第\s*([一二三四五])\s*編\s*(.+)'

    def get_articles_by_book(self, book_number: str) -> List[LegalArticle]:
        """
        Get articles from a specific book (編).

        Args:
            book_number: The book number (e.g., "一", "二", "三", "四", "五")

        Returns:
            List of articles from that book
        """
        return [article for article in self.articles
                if article.section_number == book_number]

    def get_book_summary(self) -> Dict[str, int]:
        """
        Get summary of articles per book (編).

        Returns:
            Dictionary mapping book names to article counts
        """
        book_counts = {}
        for article in self.articles:
            if article.section:
                book_counts[article.section] = book_counts.get(article.section, 0) + 1
        return book_counts


def main():
    """Example usage of CivilLawFetcher with proper resource management"""
    # Use context manager to ensure session cleanup
    # Note: Civil Code has 1229 articles, so this will take ~30 minutes with 1.5s delay
    with CivilLawFetcher(delay=1.5) as fetcher:
        try:
            print("⚠️  Warning: Civil Code has 1229 articles. This will take approximately 30 minutes.")
            print("    Press Ctrl+C to cancel if needed.\n")

            # Fetch all articles
            articles = fetcher.fetch_all_articles()

            # Save to data directory
            fetcher.save_to_json('data/civil_code.json')

            print(f"\nSuccessfully fetched {len(articles)} articles")

            # Show book summary
            book_summary = fetcher.get_book_summary()
            print("\nBook (編) Summary:")
            for book, count in book_summary.items():
                print(f"- {book}: {count} articles")

            print("\nSample articles:")
            for article in articles[:3]:
                print(f"- {article.title} ({article.section})")
                print(f"  Content preview: {article.content[:100]}...")
                print()

        except KeyboardInterrupt:
            print("\n\nFetch cancelled by user.")
            print(f"Fetched {len(fetcher.articles)} articles so far.")
        except Exception as e:
            print(f"Error: {e}")
    # Session automatically closed here


if __name__ == "__main__":
    main()