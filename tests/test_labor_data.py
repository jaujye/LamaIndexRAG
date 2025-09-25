"""
Test script to fetch Labor Law data (limited for testing)
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.labor_law_fetcher import LaborLawFetcher

def fetch_labor_sample():
    """Fetch sample Labor Law data (first 10 articles)"""
    fetcher = LaborLawFetcher(delay=1.5)

    try:
        print("Fetching Labor Law main page...")
        soup = fetcher.fetch_main_page()

        print("Extracting article links...")
        article_links = fetcher.extract_article_links(soup)
        print(f"Found {len(article_links)} articles")

        # Limit to first 10 articles for testing
        test_links = article_links[:10]
        print(f"Processing first {len(test_links)} articles for testing...")

        articles = []
        for i, link in enumerate(test_links, 1):
            print(f"[{i}/{len(test_links)}] Fetching: {link['text']}")

            content = fetcher.fetch_article_content(link['url'])
            if content:
                article = {
                    'article_number': link['number'],
                    'title': link['text'],
                    'content': content,
                    'chapter': link['chapter'],
                    'chapter_number': link['chapter_number'],
                    'url': link['url']
                }
                articles.append(article)
                print(f"[OK] Article {link['number']} fetched successfully")
            else:
                print(f"[FAIL] Could not fetch content for article {link['number']}")

        # Save sample data
        data = {
            'law_name': '勞動基準法',
            'law_code': 'N0030001',
            'source_url': fetcher.MAIN_URL,
            'total_articles': len(articles),
            'articles': articles
        }

        import json
        import os
        os.makedirs('data', exist_ok=True)

        with open('data/labor_law_sample.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] Saved {len(articles)} articles to data/labor_law_sample.json")
        return True

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fetch_labor_sample()