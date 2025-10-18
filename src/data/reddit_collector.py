"""
Reddit Data Collector
Collects cryptocurrency-related posts from Reddit using PRAW
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RedditCollector:
    """Collect data from Reddit API"""
    
    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize PRAW Reddit client"""
        try:
            self.reddit = praw.Reddit(
                client_id=Config.REDDIT_CLIENT_ID,
                client_secret=Config.REDDIT_CLIENT_SECRET,
                user_agent=Config.REDDIT_USER_AGENT,
                check_for_async=False
            )
            
            # Test authentication
            logger.info(f"Reddit API initialized. User: {self.reddit.user.me() if self.reddit.read_only else 'Read-only'}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def collect_posts(
        self,
        subreddit_name: str,
        limit: int = 100,
        time_filter: str = "day",
        sort_by: str = "hot"
    ) -> pd.DataFrame:
        """
        Collect posts from a specific subreddit
        
        Args:
            subreddit_name: Name of subreddit
            limit: Maximum number of posts to collect
            time_filter: Time filter (hour, day, week, month, year, all)
            sort_by: Sort method (hot, new, top, rising)
        
        Returns:
            DataFrame with collected posts
        """
        try:
            logger.info(f"Collecting {limit} posts from r/{subreddit_name} ({sort_by})")
            
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_data = []
            
            # Select sorting method
            if sort_by == "hot":
                posts = subreddit.hot(limit=limit)
            elif sort_by == "new":
                posts = subreddit.new(limit=limit)
            elif sort_by == "top":
                posts = subreddit.top(time_filter=time_filter, limit=limit)
            elif sort_by == "rising":
                posts = subreddit.rising(limit=limit)
            else:
                logger.warning(f"Unknown sort method: {sort_by}, using 'hot'")
                posts = subreddit.hot(limit=limit)
            
            # Collect post data
            for post in posts:
                try:
                    post_data = {
                        'post_id': post.id,
                        'title': post.title,
                        'content': post.selftext,
                        'author': str(post.author) if post.author else '[deleted]',
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'score': post.score,
                        'upvote_ratio': post.upvote_ratio,
                        'num_comments': post.num_comments,
                        'subreddit': subreddit_name,
                        'url': post.url,
                        'permalink': f"https://reddit.com{post.permalink}",
                        'is_self': post.is_self,
                        'link_flair_text': post.link_flair_text,
                        'collected_at': datetime.now()
                    }
                    posts_data.append(post_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing post {post.id}: {e}")
                    continue
            
            df = pd.DataFrame(posts_data)
            logger.info(f"Successfully collected {len(df)} posts from r/{subreddit_name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error collecting posts from r/{subreddit_name}: {e}")
            return pd.DataFrame()
    
    def collect_from_multiple_subreddits(
        self,
        subreddits: List[str],
        limit_per_subreddit: int = 100,
        time_filter: str = "day",
        sort_by: str = "hot"
    ) -> pd.DataFrame:
        """
        Collect posts from multiple subreddits
        
        Args:
            subreddits: List of subreddit names
            limit_per_subreddit: Posts to collect per subreddit
            time_filter: Time filter
            sort_by: Sort method
        
        Returns:
            Combined DataFrame with all posts
        """
        all_posts = []
        
        for subreddit in subreddits:
            try:
                df = self.collect_posts(
                    subreddit,
                    limit=limit_per_subreddit,
                    time_filter=time_filter,
                    sort_by=sort_by
                )
                
                if not df.empty:
                    all_posts.append(df)
                
                # Rate limiting - be nice to Reddit API
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to collect from r/{subreddit}: {e}")
                continue
        
        if all_posts:
            combined_df = pd.concat(all_posts, ignore_index=True)
            
            # Remove duplicates
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['post_id'], keep='first')
            duplicates_removed = initial_count - len(combined_df)
            
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate posts")
            
            logger.info(f"Total posts collected: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No posts collected from any subreddit")
            return pd.DataFrame()
    
    def save_to_csv(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        Save collected data to CSV
        
        Args:
            df: DataFrame to save
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if df.empty:
            logger.warning("DataFrame is empty, nothing to save")
            return ""
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reddit_posts_{timestamp}.csv"
            
            filepath = Config.RAW_DATA_DIR / filename
            df.to_csv(filepath, index=False)
            
            logger.info(f"Data saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving data to CSV: {e}")
            raise

def main():
    """Main function for testing"""
    logger.info("Starting Reddit data collection...")
    
    # Validate configuration
    validation = Config.validate()
    if not validation['valid']:
        logger.error(f"Configuration issues: {validation['issues']}")
        return
    
    # Initialize collector
    collector = RedditCollector()
    
    # Collect data from configured subreddits
    df = collector.collect_from_multiple_subreddits(
        subreddits=Config.SUBREDDITS,
        limit_per_subreddit=100,
        sort_by="hot"
    )
    
    # Save to CSV
    if not df.empty:
        filepath = collector.save_to_csv(df)
        logger.info(f"Collection complete! Data saved to: {filepath}")
        
        # Print summary
        print("\n" + "="*50)
        print("COLLECTION SUMMARY")
        print("="*50)
        print(f"Total posts: {len(df)}")
        print(f"Subreddits: {df['subreddit'].nunique()}")
        print(f"Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
        print("\nPosts per subreddit:")
        print(df['subreddit'].value_counts())
        print("="*50)
    else:
        logger.error("No data collected")

if __name__ == "__main__":
    main()