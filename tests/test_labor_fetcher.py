"""
Test script for Labor Law Fetcher
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.labor_law_fetcher import LaborLawFetcher

def test_labor_fetcher():
    """Test the labor law fetcher with limited articles"""
    fetcher = LaborLawFetcher(delay=1.0)

    try:
        print("Testing main page fetch...")
        soup = fetcher.fetch_main_page()
        print("[OK] Main page fetched successfully")

        print("\nTesting article links extraction...")
        article_links = fetcher.extract_article_links(soup)
        print(f"[OK] Found {len(article_links)} article links")

        # Show first 3 article links
        print("\nFirst 3 article links:")
        for i, link in enumerate(article_links[:3]):
            print(f"{i+1}. {link['text']}")
            print(f"   URL: {link['url']}")
            print(f"   Chapter: {link['chapter']}")
            print()

        # Test fetching content for first article only
        if article_links:
            print("Testing article content fetch (first article only)...")
            first_article = article_links[0]
            content = fetcher.fetch_article_content(first_article['url'])

            if content:
                print("[OK] Article content fetched successfully")
                print(f"Content preview (first 200 chars):\n{content[:200]}...")
            else:
                print("[FAIL] Failed to fetch article content")

        return True

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False

if __name__ == "__main__":
    success = test_labor_fetcher()
    if success:
        print("\n[SUCCESS] Test completed successfully!")
    else:
        print("\n[FAIL] Test failed!")