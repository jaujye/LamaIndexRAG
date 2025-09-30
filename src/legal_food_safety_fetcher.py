"""
Data fetcher for Taiwan Food Safety and Sanitation Act (食品安全衛生管理法)
Extracts legal text from the Ministry of Justice website
"""

from typing import List, Dict
from src.legal_base_fetcher import BaseLegalFetcher
from src.legal_models import LegalArticle


class FoodSafetyActFetcher(BaseLegalFetcher):
    """Fetches and processes Taiwan Food Safety and Sanitation Act"""

    def __init__(self, delay: float = 1.0):
        """
        Initialize Food Safety Act fetcher.

        Args:
            delay: Delay between requests to be respectful to the server (default: 1.0s)
        """
        super().__init__(
            pcode="L0040001",
            law_name="食品安全衛生管理法",
            delay=delay
        )

    def get_articles_by_chapter(self, chapter_number: str) -> List[LegalArticle]:
        """
        Get articles from a specific chapter.

        Args:
            chapter_number: The chapter number (e.g., "一", "二", "三")

        Returns:
            List of articles from that chapter
        """
        return [article for article in self.articles
                if article.section_number == chapter_number]

    def get_chapter_summary(self) -> Dict[str, int]:
        """
        Get summary of articles per chapter.

        Returns:
            Dictionary mapping chapter names to article counts
        """
        chapter_counts = {}
        for article in self.articles:
            if article.section:
                chapter_counts[article.section] = chapter_counts.get(article.section, 0) + 1
        return chapter_counts


def main():
    """Example usage of FoodSafetyActFetcher with proper resource management"""
    # Use context manager to ensure session cleanup
    with FoodSafetyActFetcher(delay=1.0) as fetcher:
        try:
            # Fetch all articles
            articles = fetcher.fetch_all_articles()

            # Save to data directory
            fetcher.save_to_json('data/food_safety_act.json')

            print(f"\nSuccessfully fetched {len(articles)} articles")

            # Show chapter summary
            chapter_summary = fetcher.get_chapter_summary()
            print("\nChapter Summary:")
            for chapter, count in chapter_summary.items():
                print(f"- {chapter}: {count} articles")

            print("\nSample articles:")
            for article in articles[:3]:
                print(f"- {article.title} ({article.section})")
                print(f"  Content preview: {article.content[:100]}...")
                print()

        except Exception as e:
            print(f"Error: {e}")
    # Session automatically closed here


if __name__ == "__main__":
    main()