-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS processed_data;
CREATE SCHEMA IF NOT EXISTS ml_models;

-- Raw social media data table
CREATE TABLE IF NOT EXISTS raw_data.social_media_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(50) NOT NULL,
    post_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    author VARCHAR(255),
    created_utc TIMESTAMP NOT NULL,
    score INTEGER,
    num_comments INTEGER,
    subreddit VARCHAR(100),
    url TEXT,
    metadata JSONB,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_source CHECK (source IN ('reddit', 'twitter', 'news', 'kaggle'))
);

-- Processed features table
CREATE TABLE IF NOT EXISTS processed_data.text_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES raw_data.social_media_posts(id) ON DELETE CASCADE,
    cleaned_text TEXT NOT NULL,
    text_length INTEGER,
    word_count INTEGER,
    sentiment_polarity FLOAT,
    sentiment_subjectivity FLOAT,
    named_entities JSONB,
    topics JSONB,
    language VARCHAR(10),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model predictions table
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES raw_data.social_media_posts(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    sentiment_label VARCHAR(20) NOT NULL,
    confidence_score FLOAT NOT NULL,
    probabilities JSONB,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_sentiment CHECK (sentiment_label IN ('positive', 'negative', 'neutral'))
);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS ml_models.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset_type VARCHAR(20) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_dataset CHECK (dataset_type IN ('train', 'validation', 'test'))
);

-- Data quality metrics table
CREATE TABLE IF NOT EXISTS processed_data.quality_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    threshold_value FLOAT,
    status VARCHAR(20) NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_status CHECK (status IN ('passed', 'failed', 'warning'))
);

-- Create indexes for better query performance
CREATE INDEX idx_posts_source ON raw_data.social_media_posts(source);
CREATE INDEX idx_posts_created ON raw_data.social_media_posts(created_utc);
CREATE INDEX idx_posts_subreddit ON raw_data.social_media_posts(subreddit);
CREATE INDEX idx_predictions_model ON ml_models.predictions(model_name, model_version);
CREATE INDEX idx_predictions_sentiment ON ml_models.predictions(sentiment_label);
CREATE INDEX idx_predictions_timestamp ON ml_models.predictions(predicted_at);

-- Create views for easy querying
CREATE OR REPLACE VIEW processed_data.latest_predictions AS
SELECT 
    p.id,
    smp.post_id,
    smp.title,
    smp.content,
    smp.source,
    smp.created_utc,
    p.model_name,
    p.sentiment_label,
    p.confidence_score,
    p.predicted_at
FROM ml_models.predictions p
JOIN raw_data.social_media_posts smp ON p.post_id = smp.id
WHERE p.predicted_at = (
    SELECT MAX(predicted_at) 
    FROM ml_models.predictions p2 
    WHERE p2.post_id = p.post_id
);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA raw_data TO crypto_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA processed_data TO crypto_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO crypto_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA raw_data TO crypto_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA processed_data TO crypto_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO crypto_user;