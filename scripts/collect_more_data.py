# File: scripts/collect_more_data.py
"""
Collect More Reddit Data
Target: 5000+ high-quality samples
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime
import time

from src.data.reddit_collector import RedditCollector
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main():
    """Collect more data"""
    
    logger.info("="*80)
    logger.info("COLLECTING MORE REDDIT DATA")
    logger.info("="*80)
    
    # Expanded subreddit list
    subreddits = [
        'cryptocurrency',
        'bitcoin',
        'ethereum',
        'cryptomarkets',
        'defi',
        'altcoin',
        'CryptoMoonShots',
        'satoshistreetbets',
        'ethtrader',
        'bitcoinbeginners'
    ]
    
    collector = RedditCollector()
    all_posts = []
    
    for subreddit in subreddits:
        logger.info(f"\nCollecting from r/{subreddit}...")
        
        try:
            # Collect from multiple time periods
            for time_filter in ['week', 'month']:
                logger.info(f"  Time filter: {time_filter}")
                
                posts_df = collector.collect_posts(
                    subreddit_name=subreddit,
                    limit=500,
                    time_filter=time_filter
                )
                
                # Convert DataFrame to list of dictionaries
                if not posts_df.empty:
                    posts_list = posts_df.to_dict('records')
                    all_posts.extend(posts_list)
                    logger.info(f"  Collected {len(posts_list)} posts")
                else:
                    logger.warning(f"  No posts collected from r/{subreddit}")
                
                time.sleep(2)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit}: {e}")
            continue
    
    # Create dataframe
    df = pd.DataFrame(all_posts)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['post_id'])
    
    logger.info(f"\n✅ Total collected: {len(df)} unique posts")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Config.RAW_DATA_DIR / f"reddit_posts_large_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()