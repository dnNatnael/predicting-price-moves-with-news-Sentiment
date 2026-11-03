"""
Unit Tests for Core Functions
Tests for EDA analyzer, topic modeling, and correlation analysis
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

try:
    import gensim
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


class TestEDAAnalyzer:
    """Tests for EDAAnalyzer class"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        
        df = pd.DataFrame({
            'headline': [f'Sample headline {i}' for i in range(n)],
            'publisher': np.random.choice(['Pub1', 'Pub2', 'Pub3'], n),
            'stock': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], n),
            'date': dates,
            'date_only': pd.to_datetime(dates).date,
            'headline_length': np.random.randint(20, 100, n),
            'headline_word_count': np.random.randint(3, 15, n),
            'day_of_week': [d.dayofweek for d in dates],
            'month': [d.month for d in dates],
            'year': [d.year for d in dates],
            'hour': np.random.randint(0, 24, n),
        })
        return df
    
    def test_compute_descriptive_stats(self, sample_df):
        """Test descriptive statistics computation"""
        from eda_analyzer import EDAAnalyzer
        
        analyzer = EDAAnalyzer(sample_df, output_dir="output")
        stats = analyzer.compute_descriptive_stats()
        
        assert 'headline_length' in stats
        assert 'headline_word_count' in stats
        assert stats['total_articles'] == 100
        assert stats['unique_publishers'] == 3
        assert 'mean' in stats['headline_length']
        assert 'median' in stats['headline_length']
    
    def test_analyze_top_publishers(self, sample_df):
        """Test top publishers analysis"""
        from eda_analyzer import EDAAnalyzer
        
        analyzer = EDAAnalyzer(sample_df, output_dir="output")
        top_pubs = analyzer.analyze_top_publishers(top_n=2)
        
        assert len(top_pubs) <= 2
        assert 'publisher' in top_pubs.columns
        assert 'article_count' in top_pubs.columns
    
    def test_detect_news_spikes(self, sample_df):
        """Test news spike detection"""
        from eda_analyzer import EDAAnalyzer
        
        # Add some duplicate dates to create potential spikes
        sample_df.loc[0:20, 'date_only'] = sample_df.iloc[0]['date_only']
        
        analyzer = EDAAnalyzer(sample_df, output_dir="output")
        spikes = analyzer.detect_news_spikes(threshold_std=2.0)
        
        assert isinstance(spikes, pd.DataFrame)
        assert 'date' in spikes.columns
        assert 'article_count' in spikes.columns


class TestTopicModeler:
    """Tests for TopicModeler class"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for topic modeling"""
        headlines = [
            "Apple reports record earnings for Q4",
            "Google announces new AI product launch",
            "Microsoft acquires startup in cloud deal",
            "Stock market surges on positive news",
            "Tech stocks rally amid optimism",
            "Earnings season begins next week",
            "FDA approves new drug treatment",
            "Analysts upgrade price target for Microsoft",
            "Mergers and acquisitions activity increases",
            "Market volatility expected to continue",
        ] * 10  # Repeat for more data
        
        return pd.DataFrame({'headline': headlines})
    
    def test_preprocess_text(self, sample_df):
        """Test text preprocessing"""
        from topic_modeling import TopicModeler
        
        modeler = TopicModeler(sample_df, output_dir="output")
        tokens = modeler.preprocess_text("Apple reports record earnings for Q4!")
        
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        # Check that stopwords are removed
        assert 'for' not in tokens
        assert 'the' not in tokens
    
    @pytest.mark.skipif(not GENSIM_AVAILABLE, reason="gensim not installed")
    def test_prepare_corpus(self, sample_df):
        """Test corpus preparation"""
        from topic_modeling import TopicModeler
        
        modeler = TopicModeler(sample_df, output_dir="output")
        modeler.prepare_corpus()
        
        assert modeler.processed_texts is not None
        assert len(modeler.processed_texts) > 0
        assert modeler.dictionary is not None
        assert modeler.corpus is not None
    
    @pytest.mark.skipif(not GENSIM_AVAILABLE, reason="gensim not installed")
    def test_extract_frequent_keywords(self, sample_df):
        """Test keyword extraction"""
        from topic_modeling import TopicModeler
        
        modeler = TopicModeler(sample_df, output_dir="output")
        keywords = modeler.extract_frequent_keywords(top_n=10)
        
        assert isinstance(keywords, pd.DataFrame)
        assert 'keyword' in keywords.columns
        assert 'frequency' in keywords.columns
        assert len(keywords) <= 10
    
    def test_identify_topic_categories(self, sample_df):
        """Test topic category identification"""
        from topic_modeling import TopicModeler
        
        modeler = TopicModeler(sample_df, output_dir="output")
        categories = modeler.identify_topic_categories()
        
        assert isinstance(categories, dict)
        assert 'earnings' in categories
        assert 'mergers' in categories
        assert isinstance(categories['earnings'], list)


class TestCorrelationAnalysis:
    """Tests for correlation analysis functions"""
    
    @pytest.fixture
    def sample_news_df(self):
        """Create sample news DataFrame"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        return pd.DataFrame({
            'headline': ['Positive news about stock' if i % 2 == 0 else 'Negative news about stock' for i in range(50)],
            'date': dates,
            'date_only': pd.to_datetime(dates).date,
            'stock': ['AAPL'] * 50,
            'hour': [10] * 50,  # 10 AM
        })
    
    @pytest.fixture
    def sample_prices_df(self):
        """Create sample price DataFrame"""
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 50)
        prices = base_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'date_only': pd.to_datetime(dates).date,
            'Close': prices,
            'daily_return': returns,
        })
    
    def test_score_sentiment(self, sample_news_df):
        """Test sentiment scoring"""
        from scripts.news_sentiment_stock_correlation import score_sentiment
        
        scored = score_sentiment(sample_news_df.copy())
        
        assert 'sentiment' in scored.columns
        assert scored['sentiment'].between(-1, 1).all()
    
    def test_aggregate_daily_sentiment_mean(self, sample_news_df):
        """Test mean aggregation of sentiment"""
        from scripts.news_sentiment_stock_correlation import (
            score_sentiment, aggregate_daily_sentiment
        )
        
        scored = score_sentiment(sample_news_df.copy())
        aggregated = aggregate_daily_sentiment(scored, method='mean')
        
        assert 'avg_daily_sentiment' in aggregated.columns
        assert len(aggregated) > 0
    
    def test_aggregate_daily_sentiment_median(self, sample_news_df):
        """Test median aggregation of sentiment"""
        from scripts.news_sentiment_stock_correlation import (
            score_sentiment, aggregate_daily_sentiment
        )
        
        scored = score_sentiment(sample_news_df.copy())
        aggregated = aggregate_daily_sentiment(scored, method='median')
        
        assert 'avg_daily_sentiment' in aggregated.columns
    
    def test_create_lagged_sentiment(self):
        """Test lagged sentiment creation"""
        from scripts.news_sentiment_stock_correlation import create_lagged_sentiment
        
        df = pd.DataFrame({
            'date_only': pd.date_range(start='2023-01-01', periods=10),
            'avg_daily_sentiment': np.random.randn(10)
        })
        
        lagged = create_lagged_sentiment(df, lag=1)
        
        assert 'sentiment_lag_1' in lagged.columns
        assert lagged['sentiment_lag_1'].isna().sum() >= 1  # First value should be NaN
    
    def test_compute_correlation_with_stats(self):
        """Test correlation computation with statistics"""
        from scripts.news_sentiment_stock_correlation import (
            compute_correlation_with_stats
        )
        
        # Create correlated data
        np.random.seed(42)
        sentiment = pd.DataFrame({
            'date_only': pd.date_range(start='2023-01-01', periods=50),
            'avg_daily_sentiment': np.random.randn(50)
        })
        
        returns = pd.DataFrame({
            'date_only': pd.date_range(start='2023-01-01', periods=50),
            'daily_return': sentiment['avg_daily_sentiment'] * 0.5 + np.random.randn(50) * 0.5
        })
        
        result = compute_correlation_with_stats(sentiment, returns)
        
        assert hasattr(result, 'pearson_corr')
        assert hasattr(result, 'spearman_corr')
        assert hasattr(result, 'pearson_pvalue')
        assert hasattr(result, 'spearman_pvalue')
        assert hasattr(result, 'confidence_interval_95')
        assert hasattr(result, 'n_observations')
        assert -1 <= result.pearson_corr <= 1
    
    def test_timezone_handling(self, sample_news_df):
        """Test timezone handling"""
        from scripts.news_sentiment_stock_correlation import handle_timezone
        
        # Test news after market hours (should move to next day)
        sample_news_df.loc[0, 'hour'] = 20  # 8 PM
        sample_news_df.loc[0, 'date'] = pd.Timestamp('2023-01-01 20:00:00')
        
        tz_handled = handle_timezone(sample_news_df)
        
        assert 'effective_date' in tz_handled.columns
        # News after 4 PM should be attributed to next day
        assert tz_handled.iloc[0]['effective_date'] > tz_handled.iloc[0]['date_only']


class TestTechnicalAnalysis:
    """Tests for technical analysis functions"""
    
    def test_validate_data_sufficiency(self):
        """Test data validation"""
        from scripts.technical_analysis import validate_data_sufficiency
        
        # Valid data
        valid_df = pd.DataFrame({
            'Open': np.random.randn(150),
            'High': np.random.randn(150),
            'Low': np.random.randn(150),
            'Close': np.random.randn(150),
            'Volume': np.random.randint(1000, 10000, 150),
        }, index=pd.date_range('2023-01-01', periods=150))
        
        assert validate_data_sufficiency(valid_df) == True
        
        # Invalid - insufficient data
        invalid_df = pd.DataFrame({
            'Open': np.random.randn(50),
            'High': np.random.randn(50),
            'Low': np.random.randn(50),
            'Close': np.random.randn(50),
            'Volume': np.random.randint(1000, 10000, 50),
        }, index=pd.date_range('2023-01-01', periods=50))
        
        with pytest.raises(ValueError, match="Insufficient data"):
            validate_data_sufficiency(invalid_df)


class TestDataLoader:
    """Tests for DataLoader class"""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file for testing"""
        csv_path = tmp_path / "test_data.csv"
        csv_content = """headline,url,publisher,date,stock
Apple announces new product,http://example.com/1,Reuters,2023-01-01 10:00:00,AAPL
Google stock rises,http://example.com/2,CNBC,2023-01-01 11:00:00,GOOGL
Microsoft reports earnings,http://example.com/3,WSJ,2023-01-02 09:00:00,MSFT
"""
        csv_path.write_text(csv_content)
        return csv_path
    
    def test_load_data(self, sample_csv):
        """Test data loading"""
        from data_loader import DataLoader
        
        loader = DataLoader(str(sample_csv))
        df = loader.load_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'headline' in df.columns
        assert 'publisher' in df.columns
    
    def test_preprocess_data(self, sample_csv):
        """Test data preprocessing"""
        from data_loader import DataLoader
        
        loader = DataLoader(str(sample_csv))
        df = loader.load_data()
        processed = loader.preprocess_data()
        
        assert 'headline_length' in processed.columns
        assert 'headline_word_count' in processed.columns
        assert 'date_only' in processed.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
