"""
Base fetcher class for Taiwan legal documents.
This module provides a common foundation for all legal document fetchers,
eliminating code duplication and ensuring consistent behavior.
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from typing import List, Dict, Optional
from src.legal_models import LegalArticle


class BaseLegalFetcher:
    """
    Base class for fetching legal documents from law.moj.gov.tw

    This class implements all common fetching logic including:
    - HTTP session management with automatic cleanup
    - Encoding detection and handling
    - Article content extraction
    - Respectful rate limiting

    Subclasses should override:
    - get_pcode(): Return the law's pcode
    - get_law_name(): Return the law's name
    - get_section_pattern(): Return regex pattern for section headers
    - parse_section_name(): Customize section name parsing if needed
    """

    BASE_URL = "https://law.moj.gov.tw"

    def __init__(self, pcode: str, law_name: str, delay: float = 1.0):
        """
        Initialize the fetcher.

        Args:
            pcode: The law's pcode (e.g., "N0030001" for Labor Standards Act)
            law_name: Human-readable name of the law
            delay: Delay in seconds between requests (default: 1.0)
        """
        self.pcode = pcode
        self.law_name = law_name
        self.delay = delay
        self.MAIN_URL = f"{self.BASE_URL}/LawClass/LawAll.aspx?pcode={pcode}"

        # Create session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        self.articles: List[LegalArticle] = []

    def __enter__(self):
        """Context manager entry - allows 'with' statement usage"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures session cleanup"""
        self.close()
        return False

    def close(self):
        """Explicitly close the session to prevent resource leaks"""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __del__(self):
        """Destructor - cleanup session if not already closed"""
        self.close()

    def _fix_encoding(self, response: requests.Response) -> None:
        """
        Fix encoding issues with response.

        Args:
            response: The requests.Response object to fix
        """
        if response.encoding and response.encoding.lower() in ['iso-8859-1', 'windows-1252']:
            response.encoding = 'utf-8'
        elif not response.encoding:
            response.encoding = 'utf-8'

    def fetch_main_page(self) -> BeautifulSoup:
        """
        Fetch the main law page.

        Returns:
            BeautifulSoup object of the main page

        Raises:
            Exception: If the page cannot be fetched
        """
        try:
            response = self.session.get(self.MAIN_URL, timeout=30)
            response.raise_for_status()
            self._fix_encoding(response)
            return BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch main page for {self.law_name}: {e}")

    def get_section_pattern(self) -> str:
        """
        Get the regex pattern for identifying section headers.

        Override this in subclasses for different section patterns.
        Default matches: 第X章 (Chapter X)

        Returns:
            Regex pattern string
        """
        return r'第\s*([一二三四五六七八九十]+)\s*章\s*(.+)'

    def parse_section_name(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """
        Parse section header from text.

        Args:
            text: Text to parse

        Returns:
            Tuple of (section_number, full_section_text) or (None, None)
        """
        match = re.match(self.get_section_pattern(), text)
        if match:
            return match.group(1), text
        return None, None

    def extract_article_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Extract article links and metadata from the main page.

        Args:
            soup: BeautifulSoup object of the main page

        Returns:
            List of dictionaries with article metadata
        """
        article_links = []
        current_section = ""
        current_section_number = ""

        # Find all elements that could be sections or articles
        for element in soup.find_all(['a', 'div', 'span']):
            text = element.get_text(strip=True)

            # Check for section headers
            section_num, section_text = self.parse_section_name(text)
            if section_num:
                current_section_number = section_num
                current_section = section_text
                print(f"Found section: {current_section}")
                continue

            # Check for article links
            if element.name == 'a' and element.get('href'):
                href = element.get('href')
                if 'LawSingle.aspx' in href and f'pcode={self.pcode}' in href:
                    article_match = re.match(r'第\s*(\d+(?:-\d+)?)\s*條', text)
                    if article_match:
                        # Build full URL
                        if href.startswith('http'):
                            full_url = href
                        elif href.startswith('/'):
                            full_url = f"{self.BASE_URL}{href}"
                        else:
                            full_url = f"{self.BASE_URL}/LawClass/{href}"

                        article_links.append({
                            'number': article_match.group(1),
                            'title': text,
                            'url': full_url,
                            'section': current_section,
                            'section_number': current_section_number
                        })

        return article_links

    def fetch_article_content(self, article_url: str) -> Optional[str]:
        """
        Fetch individual article content.

        This method implements the common article fetching logic used by all fetchers.
        It tries multiple CSS selectors and parsing strategies to extract content.

        Args:
            article_url: Full URL to the article page

        Returns:
            Article content as string, or None if fetching failed
        """
        try:
            time.sleep(self.delay)  # Respectful delay
            response = self.session.get(article_url, timeout=30)
            response.raise_for_status()
            self._fix_encoding(response)

            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')

            # Try multiple selectors to find article content
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
                if re.match(r'第\s*\d+(?:-\d+)?\s*條', line):
                    capturing = True
                    article_content.append(line)
                elif capturing and line:
                    if re.match(r'第\s*\d+(?:-\d+)?\s*條', line) or 'Copyright' in line or '回首頁' in line:
                        break
                    article_content.append(line)

            return '\n'.join(article_content) if article_content else None

        except requests.RequestException as e:
            print(f"Failed to fetch article {article_url}: {e}")
            return None

    def fetch_all_articles(self) -> List[LegalArticle]:
        """
        Fetch all articles from the law.

        Returns:
            List of LegalArticle objects
        """
        print(f"Fetching {self.law_name} main page...")
        soup = self.fetch_main_page()

        print("Extracting article links...")
        article_links = self.extract_article_links(soup)
        print(f"Found {len(article_links)} articles")

        if not article_links:
            print("Warning: No articles found!")
            return []

        print(f"Fetching article content (with {self.delay}s delay between requests)...")
        fetched_count = 0

        for idx, article_info in enumerate(article_links, 1):
            content = self.fetch_article_content(article_info['url'])

            if content:
                article = LegalArticle(
                    article_number=article_info['number'],
                    title=article_info['title'],
                    content=content,
                    section=article_info['section'],
                    section_number=article_info['section_number'],
                    url=article_info['url']
                )
                self.articles.append(article)
                fetched_count += 1

                # Progress indicator
                if idx % 10 == 0 or idx == len(article_links):
                    print(f"Progress: {idx}/{len(article_links)} articles processed")
            else:
                print(f"Warning: Failed to fetch content for article {article_info['number']}")

        print(f"\nSuccessfully fetched {fetched_count}/{len(article_links)} articles")
        return self.articles

    def save_to_json(self, filepath: str) -> None:
        """
        Save fetched articles to JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        import json

        if not self.articles:
            print("No articles to save!")
            return

        data = {
            'law_name': self.law_name,
            'pcode': self.pcode,
            'total_articles': len(self.articles),
            'articles': [article.to_dict() for article in self.articles]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.articles)} articles to {filepath}")