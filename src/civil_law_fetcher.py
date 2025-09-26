"""
Data fetcher for Taiwan Civil Code (民法)
Extracts legal text from the Ministry of Justice website
Based on the LaborLawFetcher architecture but adapted for Civil Law
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class CivilLegalArticle:
    """Represents a single civil law article"""
    article_number: str
    title: str
    content: str
    book: str  # 編 (e.g., 總則編, 債編, 物權編, 親屬編, 繼承編)
    book_number: str
    url: str


class CivilLawFetcher:
    """Fetches and processes Taiwan Civil Code (民法)"""

    BASE_URL = "https://law.moj.gov.tw"
    MAIN_URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=B0000001"

    def __init__(self, delay: float = 1.5):
        """
        Initialize fetcher
        Args:
            delay: Delay between requests to be respectful to the server
                  (Civil law has 1229 articles, so we use a longer delay)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.articles: List[CivilLegalArticle] = []

    def fetch_main_page(self) -> BeautifulSoup:
        """Fetch the main civil code page"""
        try:
            response = self.session.get(self.MAIN_URL, timeout=30)
            response.raise_for_status()

            # Try to detect encoding from response
            if response.encoding.lower() in ['iso-8859-1', 'windows-1252']:
                response.encoding = 'utf-8'
            elif not response.encoding:
                response.encoding = 'utf-8'

            return BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch main page: {e}")

    def extract_article_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract article links and metadata from main page"""
        article_links = []
        current_book = ""
        current_book_number = ""

        # Find all elements that could be books (編) or articles
        for element in soup.find_all(['a', 'div', 'span']):
            text = element.get_text(strip=True)

            # Check for book headers - Civil Code has 5 main books (編)
            # Pattern: 第一編 總則、第二編 債、第三編 物權、第四編 親屬、第五編 繼承
            book_match = re.match(r'第\s*([一二三四五])\s*編\s*(.+)', text)
            if book_match:
                current_book_number = book_match.group(1)
                current_book = text
                print(f"Found book (編): {current_book}")
                continue

            # Check for article links
            if element.name == 'a' and element.get('href'):
                href = element.get('href')
                # Civil Code uses pcode=B0000001
                if 'LawSingle.aspx' in href and 'pcode=B0000001' in href:
                    # Civil code articles can go beyond 999 (up to 1229)
                    article_match = re.match(r'第\s*(\d+)\s*條', text)
                    if article_match:
                        article_number = article_match.group(1)
                        # Skip deleted articles (marked as 已刪除)
                        if '已刪除' not in text:
                            article_links.append({
                                'number': article_number,
                                'text': text,
                                'url': self.BASE_URL + '/LawClass/' + href,
                                'book': current_book,
                                'book_number': current_book_number
                            })

        return article_links

    def fetch_article_content(self, article_url: str) -> Optional[str]:
        """Fetch individual article content"""
        try:
            time.sleep(self.delay)  # Respectful delay
            response = self.session.get(article_url, timeout=30)
            response.raise_for_status()

            # Fix encoding issues
            if response.encoding.lower() in ['iso-8859-1', 'windows-1252']:
                response.encoding = 'utf-8'
            elif not response.encoding:
                response.encoding = 'utf-8'

            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')

            # Find the article content - try multiple selectors
            content_selectors = [
                '.law-article',
                '.article-content',
                '[class*="article"]',
                'div.MsoNormal',
                'span',
                'div[style*="margin"]',
                '.content'
            ]

            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    # Look for substantial content that includes article number
                    if len(text) > 50 and '第' in text and '條' in text:
                        # Clean up the text
                        text = re.sub(r'\s+', ' ', text)
                        text = text.replace('\u3000', ' ')  # Full-width space
                        return text

            # Fallback: get all text and try to extract relevant content
            all_text = soup.get_text()
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]

            # Find lines that look like article content
            article_content = []
            capturing = False
            for line in lines:
                if re.match(r'第\s*\d+\s*條', line):
                    capturing = True
                    article_content.append(line)
                elif capturing and line:
                    if re.match(r'第\s*\d+\s*條', line) or 'Copyright' in line or '回首頁' in line:
                        break
                    article_content.append(line)

            return '\n'.join(article_content) if article_content else None

        except requests.RequestException as e:
            print(f"Failed to fetch article {article_url}: {e}")
            return None

    def fetch_all_articles(self, batch_size: int = 50, start_from: int = 1) -> List[CivilLegalArticle]:
        """
        Fetch all articles from the Civil Code
        Args:
            batch_size: Number of articles to process before saving checkpoint
            start_from: Article number to start from (for resuming interrupted fetches)
        """
        print("Fetching Civil Code main page...")
        soup = self.fetch_main_page()

        print("Extracting article links...")
        article_links = self.extract_article_links(soup)
        print(f"Found {len(article_links)} articles")

        # Filter articles to start from specified number
        if start_from > 1:
            article_links = [link for link in article_links if int(link['number']) >= start_from]
            print(f"Starting from article {start_from}, processing {len(article_links)} articles")

        articles = []
        failed_articles = []

        for i, link in enumerate(article_links, 1):
            article_num = link['number']
            print(f"Fetching article {i}/{len(article_links)}: 第{article_num}條")

            content = self.fetch_article_content(link['url'])
            if content:
                article = CivilLegalArticle(
                    article_number=article_num,
                    title=link['text'],
                    content=content,
                    book=link['book'],
                    book_number=link['book_number'],
                    url=link['url']
                )
                articles.append(article)
            else:
                print(f"Warning: Could not fetch content for article {article_num}")
                failed_articles.append(article_num)

            # Save checkpoint every batch_size articles
            if i % batch_size == 0:
                checkpoint_file = f'data/civil_code_checkpoint_{article_num}.json'
                self._save_checkpoint(articles, checkpoint_file)
                print(f"Checkpoint saved: {len(articles)} articles processed")

        self.articles = articles

        # Report results
        print(f"\nFetch completed:")
        print(f"- Successfully fetched: {len(articles)} articles")
        print(f"- Failed articles: {len(failed_articles)}")
        if failed_articles:
            print(f"- Failed article numbers: {failed_articles[:10]}...")  # Show first 10

        return articles

    def _save_checkpoint(self, articles: List[CivilLegalArticle], filepath: str) -> None:
        """Save checkpoint during fetch process"""
        data = {
            'law_name': '民法',
            'law_code': 'B0000001',
            'source_url': self.MAIN_URL,
            'total_articles': len(articles),
            'checkpoint_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'articles': []
        }

        for article in articles:
            data['articles'].append({
                'article_number': article.article_number,
                'title': article.title,
                'content': article.content,
                'book': article.book,
                'book_number': article.book_number,
                'url': article.url
            })

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_to_json(self, filepath: str) -> None:
        """Save articles to JSON file"""
        if not self.articles:
            raise ValueError("No articles to save. Run fetch_all_articles() first.")

        data = {
            'law_name': '民法',
            'law_code': 'B0000001',
            'source_url': self.MAIN_URL,
            'total_articles': len(self.articles),
            'fetch_completed': time.strftime('%Y-%m-%d %H:%M:%S'),
            'articles': []
        }

        for article in self.articles:
            data['articles'].append({
                'article_number': article.article_number,
                'title': article.title,
                'content': article.content,
                'book': article.book,
                'book_number': article.book_number,
                'url': article.url
            })

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.articles)} articles to {filepath}")

    def get_articles_by_book(self, book_number: str) -> List[CivilLegalArticle]:
        """Get articles from a specific book (編)"""
        return [article for article in self.articles
                if article.book_number == book_number]

    def get_book_summary(self) -> Dict[str, int]:
        """Get summary of articles per book (編)"""
        book_counts = {}
        for article in self.articles:
            if article.book:
                book_counts[article.book] = book_counts.get(article.book, 0) + 1
        return book_counts

    def get_articles_by_range(self, start: int, end: int) -> List[CivilLegalArticle]:
        """Get articles within a specific number range"""
        return [article for article in self.articles
                if start <= int(article.article_number) <= end]


def main():
    """Example usage"""
    fetcher = CivilLawFetcher(delay=2.0)  # Longer delay for large Civil Code

    try:
        # For testing, start with a small range
        print("Fetching Civil Code articles...")
        print("Note: Civil Code has 1229 articles. This may take 40+ minutes.")

        # Option to fetch in batches
        batch_choice = input("Fetch all articles? (y/N, or enter range like '1-50'): ").strip().lower()

        if batch_choice.startswith('y'):
            articles = fetcher.fetch_all_articles(batch_size=100)
        elif '-' in batch_choice:
            start, end = map(int, batch_choice.split('-'))
            print(f"Fetching articles {start}-{end} for testing...")
            # Modify fetch logic for range (simplified for demo)
            articles = fetcher.fetch_all_articles()
            articles = fetcher.get_articles_by_range(start, end)
        else:
            print("Fetching first 10 articles for demo...")
            articles = fetcher.fetch_all_articles()[:10]

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
            print(f"- {article.title} ({article.book})")
            print(f"  Content preview: {article.content[:100]}...")
            print()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()