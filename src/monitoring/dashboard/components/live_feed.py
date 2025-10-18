# File: src/monitoring/dashboard/components/live_feed.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import requests
from typing import List, Dict

def show_live_reddit_feed(api: 'DashboardAPI', subreddits: List[str] = None):
    """Display live Reddit feed with real-time sentiment analysis"""
    
    st.markdown("## 🔴 Live Reddit Feed")
    st.markdown("Real-time sentiment analysis of cryptocurrency discussions")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if subreddits is None:
            subreddits = st.multiselect(
                "Subreddits to monitor:",
                ["cryptocurrency", "bitcoin", "ethereum", "cryptomarkets", "defi"],
                default=["cryptocurrency", "bitcoin"]
            )
    
    with col2:
        max_posts = st.slider("Posts to analyze:", 5, 50, 10)
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.number_input("Refresh (seconds):", 10, 60, 30)
    
    # Placeholder for live feed
    feed_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Start/Stop buttons
    col1, col2 = st.columns([1, 5])
    with col1:
        start_button = st.button("▶️ Start", type="primary")
    with col2:
        stop_button = st.button("⏹️ Stop")
    
    if start_button or auto_refresh:
        with st.spinner("🔄 Fetching live posts..."):
            # Simulate fetching posts (in production, use actual Reddit API)
            live_posts = fetch_reddit_posts(subreddits, max_posts)
            
            # Analyze each post
            analyzed_posts = []
            for post in live_posts:
                result = api.predict_sentiment(post['text'], include_features=False)
                
                if 'error' not in result:
                    analyzed_posts.append({
                        'title': post['title'],
                        'subreddit': post['subreddit'],
                        'sentiment': result['prediction'],
                        'confidence': result['confidence'],
                        'timestamp': post['timestamp'],
                        'score': post['score'],
                        'comments': post['comments']
                    })
            
            # Display metrics
            with metrics_placeholder.container():
                display_feed_metrics(analyzed_posts)
            
            # Display feed
            with feed_placeholder.container():
                display_live_posts(analyzed_posts)
        
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()

def fetch_reddit_posts(subreddits: List[str], max_posts: int) -> List[Dict]:
    """Fetch recent Reddit posts (mock data for demo)"""
    import random
    
    sample_titles = [
        "Bitcoin breaks $125,000! Is this the new ATH?",
        "Ethereum gas fees are getting ridiculous...",
        "Just bought my first crypto! Excited to join the community",
        "Warning: New scam targeting crypto investors",
        "DeFi yields are looking good this week",
        "Market analysis: Bull run continues",
        "Lost my wallet seed phrase, any advice?",
        "Crypto regulation news - what does it mean for us?",
        "Best altcoins to invest in 2025?",
        "Staking rewards comparison - which platform is best?"
    ]
    
    posts = []
    for i in range(max_posts):
        posts.append({
            'title': random.choice(sample_titles),
            'text': random.choice(sample_titles) + " " + "Some additional context here.",
            'subreddit': random.choice(subreddits),
            'timestamp': datetime.now(),
            'score': random.randint(10, 500),
            'comments': random.randint(5, 100)
        })
    
    return posts

def display_feed_metrics(posts: List[Dict]):
    """Display aggregated metrics for the feed"""
    if not posts:
        st.warning("No posts to display")
        return
    
    # Calculate metrics
    total = len(posts)
    positive = sum(1 for p in posts if p['sentiment'] == 'positive')
    negative = sum(1 for p in posts if p['sentiment'] == 'negative')
    neutral = sum(1 for p in posts if p['sentiment'] == 'neutral')
    avg_confidence = sum(p['confidence'] for p in posts) / total
    
    # Display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Posts", total)
    
    with col2:
        st.metric("😊 Positive", f"{positive} ({positive/total*100:.0f}%)")
    
    with col3:
        st.metric("😞 Negative", f"{negative} ({negative/total*100:.0f}%)")
    
    with col4:
        st.metric("💎 Avg Confidence", f"{avg_confidence*100:.1f}%")

def display_live_posts(posts: List[Dict]):
    """Display live posts in a feed format"""
    st.markdown("### 📰 Latest Posts")
    
    for post in posts:
        # Sentiment emoji
        emoji = "😊" if post['sentiment'] == 'positive' else "😞" if post['sentiment'] == 'negative' else "😐"
        
        # Confidence color
        if post['confidence'] > 0.8:
            conf_color = "🟢"
        elif post['confidence'] > 0.6:
            conf_color = "🟡"
        else:
            conf_color = "🔴"
        
        # Display post card
        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 4px solid {'#00ff88' if post['sentiment'] == 'positive' else '#ff4444' if post['sentiment'] == 'negative' else '#ffaa00'};
            ">
                <h4>{emoji} {post['title']}</h4>
                <p><strong>r/{post['subreddit']}</strong> • {post['timestamp'].strftime('%H:%M:%S')}</p>
                <p>
                    Sentiment: <strong>{post['sentiment'].upper()}</strong> {conf_color} 
                    Confidence: <strong>{post['confidence']*100:.1f}%</strong> • 
                    Score: <strong>{post['score']}</strong> • 
                    Comments: <strong>{post['comments']}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)