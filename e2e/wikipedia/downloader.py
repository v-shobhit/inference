"""
Wikipedia article download orchestration.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool
from tqdm import tqdm
from .extractor import WikipediaExtractor


class WikipediaDownloader:
    """
    Orchestrates parallel downloading of Wikipedia articles.
    Handles progress tracking, error handling, and statistics.
    """
    
    def __init__(self, extractor: WikipediaExtractor, workers: int = 10):
        """
        Initialize the downloader.
        
        Args:
            extractor: WikipediaExtractor instance
            workers: Number of parallel download workers
        """
        self.extractor = extractor
        self.workers = min(workers, 20)  # Cap at 20 to be respectful to Wikipedia
    
    def download_articles(
        self,
        urls: List[str],
        output_dir: Path,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Download multiple Wikipedia articles in parallel.
        
        Args:
            urls: List of Wikipedia URLs to download
            output_dir: Directory to save articles
            verbose: Print progress information
        
        Returns:
            Dictionary with download statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if verbose:
            print(f"\n=== DOWNLOADING WIKIPEDIA ARTICLES ===")
            print(f"Total URLs: {len(urls)}")
            print(f"Output directory: {output_dir}")
            print(f"Workers: {self.workers}")
        
        # Prepare arguments for multiprocessing
        process_args = [(url, output_dir, self.extractor) for url in urls]
        
        # Process URLs in parallel with progress bar
        start_time = time.time()
        
        results = []
        failed_urls = []
        
        with Pool(processes=self.workers) as pool:
            with tqdm(total=len(urls), desc="Downloading articles", unit="article", disable=not verbose) as pbar:
                # Use imap_unordered for immediate progress updates as workers complete
                for result in pool.imap_unordered(_process_single_url_worker, process_args):
                    success, url, content, message = result
                    results.append((success, url))
                    
                    if success:
                        if message == "Already exists":
                            pbar.set_description(f"⊙ Skipped (exists)")
                        else:
                            pbar.set_description(f"✓ Downloaded")
                    else:
                        pbar.set_description(f"✗ Failed")
                        failed_urls.append((url, message))
                        if message not in ["Already exists"] and verbose:
                            tqdm.write(f"\n❌ FAILED: {url}")
                            tqdm.write(f"   Error: {message}")
                    
                    pbar.update(1)
        
        # Count results
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        end_time = time.time()
        duration = end_time - start_time
        
        if verbose:
            print(f"\n=== DOWNLOAD COMPLETE ===")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print(f"Total time: {duration:.2f} seconds")
            print(f"Average time per URL: {duration/len(urls):.2f} seconds")
        
        # Print detailed failure report if there were failures
        if failed_urls and verbose:
            print(f"\n=== FAILED DOWNLOADS ===")
            for i, (url, message) in enumerate(failed_urls[:10], 1):  # Show first 10
                print(f"{i:2d}. {url}")
                print(f"    Error: {message}")
            if len(failed_urls) > 10:
                print(f"... and {len(failed_urls) - 10} more failures")
        
        return {
            'successful': successful,
            'failed': failed,
            'total': len(urls),
            'duration': duration,
            'avg_time_per_url': duration / len(urls) if urls else 0,
            'failed_urls': failed_urls
        }


def _process_single_url_worker(args_tuple) -> Tuple[bool, str, Optional[Dict], str]:
    """
    Worker function for processing a single Wikipedia URL.
    Designed for multiprocessing.
    
    Args:
        args_tuple: (url, output_dir, extractor)
    
    Returns:
        Tuple of (success, url, content_dict, error_message)
    """
    url, output_dir, extractor = args_tuple
    
    # Create safe filename from URL
    title = extractor.extract_article_title_from_url(url)
    if not title:
        return False, url, None, "Could not extract title from URL"
    
    # Create safe filename
    safe_filename = re.sub(r'[^\w\s-]', '_', title)
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
    if len(safe_filename) > 200:
        safe_filename = safe_filename[:200]
    
    output_path = output_dir / f"{safe_filename}.txt"
    json_path = output_dir / f"{safe_filename}.json"
    
    # Skip if file already exists
    if output_path.exists() and json_path.exists():
        return True, url, None, "Already exists"
    
    # Download content
    content = extractor.download_from_url(url)
    
    if not content:
        return False, url, None, "Failed to download or extract content"
    
    # Check if content is substantial
    if len(content['text']) < 100:
        return False, url, None, f"Content too short ({len(content['text'])} chars)"
    
    try:
        # Save as plain text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content['text'])
        
        # Save metadata as JSON
        metadata = {
            'title': content['title'],
            'pageid': content['pageid'],
            'url': content['url'],
            'source_url': content['source_url'],
            'length': content['length'],
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True, url, content, "Success"
        
    except Exception as e:
        # Clean up partial files
        if output_path.exists():
            output_path.unlink()
        if json_path.exists():
            json_path.unlink()
        return False, url, None, f"Error saving: {str(e)}"

