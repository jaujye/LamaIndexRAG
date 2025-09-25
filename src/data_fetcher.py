"""
Data fetcher for Taiwan Food Safety and Sanitation Act
Extracts legal text from the Ministry of Justice website
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
class LegalArticle:
    """Represents a single legal article"""
    article_number: str
    title: str
    content: str
    chapter: str
    chapter_number: str
    url: str


class FoodSafetyActFetcher:
    """Fetches and processes Taiwan Food Safety and Sanitation Act"""

    BASE_URL = "https://law.moj.gov.tw"
    MAIN_URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=L0040001"

    def __init__(self, delay: float = 1.0):
        """
        Initialize fetcher
        Args:
            delay: Delay between requests to be respectful to the server
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.articles: List[LegalArticle] = []

    def fetch_main_page(self) -> BeautifulSoup:
        """Fetch the main law page"""
        try:
            response = self.session.get(self.MAIN_URL, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch main page: {e}")

    def extract_article_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract article links and metadata from main page"""
        article_links = []
        current_chapter = ""
        current_chapter_number = ""

        # Find all elements that could be chapters or articles
        for element in soup.find_all(['a', 'div', 'span']):
            text = element.get_text(strip=True)

            # Check for chapter headers
            chapter_match = re.match(r'第\s*([一二三四五六七八九十]+)\s*章\s*(.+)', text)
            if chapter_match:
                current_chapter_number = chapter_match.group(1)
                current_chapter = text
                continue

            # Check for article links
            if element.name == 'a' and element.get('href'):
                href = element.get('href')
                if 'LawSingle.aspx' in href and 'pcode=L0040001' in href:
                    article_match = re.match(r'第\s*(\d+)\s*條', text)
                    if article_match:
                        article_links.append({
                            'number': article_match.group(1),
                            'text': text,
                            'url': self.BASE_URL + '/LawClass/' + href,
                            'chapter': current_chapter,
                            'chapter_number': current_chapter_number
                        })

        return article_links

    def fetch_article_content(self, article_url: str) -> Optional[str]:
        """Fetch individual article content"""
        try:
            time.sleep(self.delay)  # Respectful delay
            response = self.session.get(article_url, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the article content - typically in a div or span with article text
            content_selectors = [
                '.law-article',
                '.article-content',
                '[class*="article"]',
                'div.MsoNormal',
                'span'
            ]

            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if len(text) > 50 and '第' in text and '條' in text:
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
                    if re.match(r'第\s*\d+\s*條', line) or 'Copyright' in line:
                        break
                    article_content.append(line)

            return '\n'.join(article_content) if article_content else None

        except requests.RequestException as e:
            print(f"Failed to fetch article {article_url}: {e}")
            return None

    def fetch_all_articles(self) -> List[LegalArticle]:
        """Fetch all articles from the Food Safety Act"""
        print("Fetching main page...")
        soup = self.fetch_main_page()

        print("Extracting article links...")
        article_links = self.extract_article_links(soup)
        print(f"Found {len(article_links)} articles")

        articles = []
        for i, link in enumerate(article_links, 1):
            print(f"Fetching article {i}/{len(article_links)}: 第{link['number']}條")

            content = self.fetch_article_content(link['url'])
            if content:
                article = LegalArticle(
                    article_number=link['number'],
                    title=link['text'],
                    content=content,
                    chapter=link['chapter'],
                    chapter_number=link['chapter_number'],
                    url=link['url']
                )
                articles.append(article)
            else:
                print(f"Warning: Could not fetch content for article {link['number']}")

        self.articles = articles
        return articles

    def save_to_json(self, filepath: str) -> None:
        """Save articles to JSON file"""
        if not self.articles:
            raise ValueError("No articles to save. Run fetch_all_articles() first.")

        data = {
            'law_name': '食品安全衛生管理法',
            'law_code': 'L0040001',
            'source_url': self.MAIN_URL,
            'total_articles': len(self.articles),
            'articles': []
        }

        for article in self.articles:
            data['articles'].append({
                'article_number': article.article_number,
                'title': article.title,
                'content': article.content,
                'chapter': article.chapter,
                'chapter_number': article.chapter_number,
                'url': article.url
            })

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.articles)} articles to {filepath}")

    def get_articles_by_chapter(self, chapter_number: str) -> List[LegalArticle]:
        """Get articles from a specific chapter"""
        return [article for article in self.articles
                if article.chapter_number == chapter_number]


def main():
    """Example usage"""
    fetcher = FoodSafetyActFetcher()

    try:
        # Fetch all articles
        articles = fetcher.fetch_all_articles()

        # Save to data directory
        fetcher.save_to_json('data/food_safety_act.json')

        print(f"\nSuccessfully fetched {len(articles)} articles")
        print("\nSample articles:")
        for article in articles[:3]:
            print(f"- {article.title} ({article.chapter})")
            print(f"  Content preview: {article.content[:100]}...")
            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()