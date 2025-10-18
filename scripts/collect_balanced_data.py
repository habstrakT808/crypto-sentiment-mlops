# File: scripts/collect_balanced_data.py
"""
🎯 Collect Balanced Dataset
Target: 1000+ samples per class
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
import time

from src.data.reddit_collector import RedditCollector
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

def main():
    """Collect balanced dataset"""
    
    logger.info("="*80)
    logger.info("COLLECTING BALANCED DATASET")
    logger.info("Target: 1000+ samples per sentiment class")
    logger.info("="*80)
    
    # Expanded subreddit list with diverse communities
    subreddits = {
        'positive_heavy': ['CryptoMoonShots', 'SatoshiStreetBets', 'ethtrader'],
        'neutral_heavy': ['CryptoCurrency', 'CryptoMarkets', 'bitcoinbeginners'],
        'negative_heavy': ['Buttcoin'],  # Known for skepticism
        'mixed': ['Bitcoin', 'ethereum', 'defi', 'altcoin']
    }
    
    collector = RedditCollector()
    all_posts = []
    
    # Collect from each category
    for category, subreddit_list in subreddits.items():
        logger.info(f"\n📊 Collecting from {category} subreddits...")
        
        for subreddit in subreddit_list:
            logger.info(f"  Collecting from r/{subreddit}...")
            
            try:
                # Collect from multiple time periods
                for time_filter in ['week', 'month', 'all']:
                    for sort_by in ['hot', 'top', 'controversial']:
                        logger.info(f"    {sort_by} ({time_filter})...")
                        
                        posts_df = collector.collect_posts(
                            subreddit_name=subreddit,
                            limit=200,
                            time_filter=time_filter,
                            sort_by=sort_by
                        )
                        
                        if not posts_df.empty:
                            posts_df['category'] = category
                            all_posts.append(posts_df)
                            logger.info(f"    ✅ Collected {len(posts_df)} posts")
                        
                        time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit}: {e}")
                continue
    
    # Combine all posts
    if all_posts:
        df = pd.concat(all_posts, ignore_index=True)
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['post_id'])
        logger.info(f"\n✅ Removed {initial_count - len(df)} duplicates")
        
        logger.info(f"\n📊 Total collected: {len(df)} unique posts")
        logger.info(f"Distribution by category:")
        logger.info(df['category'].value_counts())
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Config.RAW_DATA_DIR / f"reddit_balanced_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"\n✅ Saved to: {output_path}")
        logger.info("="*80)
        
        return output_path
    else:
        logger.error("❌ No posts collected!")
        return None

if __name__ == "__main__":
    main()