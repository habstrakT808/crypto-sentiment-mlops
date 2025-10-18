"""
Database Connection Management
Handles PostgreSQL connections and session management
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import time

from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create base class for declarative models
Base = declarative_base()

class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self):
        """Initialize database connection"""
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Create database engine with connection pooling"""
        try:
            self.engine = create_engine(
                Config.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=Config.DEBUG    # Log SQL queries in debug mode
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def wait_for_db(self, max_retries: int = 30, retry_interval: int = 2):
        """
        Wait for database to be ready
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_interval: Seconds to wait between retries
        """
        for attempt in range(max_retries):
            try:
                if self.test_connection():
                    logger.info("Database is ready")
                    return
            except Exception as e:
                logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(retry_interval)
        
        raise ConnectionError("Database did not become ready in time")
    
    @contextmanager
    def get_session(self):
        """
        Get database session with automatic cleanup
        
        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all tables defined in models"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

# Create global database instance
db = DatabaseConnection()