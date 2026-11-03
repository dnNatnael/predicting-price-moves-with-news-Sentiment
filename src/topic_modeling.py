"""
Topic Modeling Module
Performs topic modeling using LDA and BERTopic on financial news headlines

Module Usage:
    >>> from topic_modeling import TopicModeler
    >>> modeler = TopicModeler(df, output_dir="output")
    >>> modeler.prepare_corpus()
    >>> lda_model = modeler.train_lda(num_topics=10, passes=10)
    >>> topics = modeler.get_lda_topics(num_words=10)

Parameter Choices:
    - num_topics (LDA): 10 topics is a good starting point. Can be tuned using
      coherence scores (see compute_coherence method).
    - min_topic_size (BERTopic): 10 is minimum; larger values (20-50) produce
      more stable topics but may miss niche topics.
    - passes: 10 passes provides good convergence; increase for larger datasets.
    - alpha/eta: 'auto' allows automatic tuning based on corpus size.

Preprocessing Steps:
    1. Lowercase conversion
    2. URL removal
    3. Special character removal
    4. Tokenization (NLTK word_tokenize)
    5. Stopword removal (NLTK english + financial custom stopwords)
    6. Lemmatization (WordNetLemmatizer)
    7. Short word removal (< 3 characters)

Author: Nova Financial Solutions
Version: 2.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Topic modeling imports
try:
    from gensim import corpora, models
    from gensim.models import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    corpora = None
    models = None
    CoherenceModel = None

# Optional imports
try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    pyLDAvis = None
    gensimvis = None

try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TopicModeler:
    """Perform topic modeling on financial news headlines"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "output"):
        """
        Initialize the Topic Modeler
        
        Args:
            df: DataFrame with headlines
            output_dir: Directory to save outputs
        """
        self.df = df
        self.output_dir = output_dir
        self.processed_texts = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.bertopic_model = None
        self.topics = {}
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Financial stopwords
        self.financial_stopwords = {
            'stock', 'stocks', 'shares', 'share', 'company', 'companies',
            'market', 'markets', 'trading', 'trade', 'investor', 'investors',
            'price', 'prices', 'financial', 'finance', 'news', 'report',
            'reports', 'said', 'says', 'say', 'according', 'could', 'would',
            'may', 'might', 'also', 'new', 'first', 'time', 'year', 'years'
        }
        
        # Get standard stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        self.stop_words.update(self.financial_stopwords)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess a single text
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words, lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def prepare_corpus(self):
        """Prepare corpus for topic modeling"""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed. Install it with: pip install gensim")
        print("Preprocessing headlines...")
        self.processed_texts = self.df['headline'].apply(self.preprocess_text).tolist()
        
        # Remove empty texts
        self.processed_texts = [text for text in self.processed_texts if len(text) > 0]
        
        # Create dictionary
        print("Creating dictionary...")
        self.dictionary = corpora.Dictionary(self.processed_texts)
        
        # Filter extremes
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        
        # Create corpus
        print("Creating corpus...")
        self.corpus = [self.dictionary.doc2bow(text) for text in self.processed_texts]
        
        print(f"Corpus prepared: {len(self.processed_texts)} documents, {len(self.dictionary)} unique terms")
    
    def extract_frequent_keywords(self, top_n: int = 50) -> pd.DataFrame:
        """
        Extract frequent keywords and phrases
        
        Args:
            top_n: Number of top keywords to return
            
        Returns:
            DataFrame with keywords and frequencies
        """
        if self.processed_texts is None:
            self.prepare_corpus()
        
        # Flatten all tokens
        all_tokens = [token for text in self.processed_texts for token in text]
        
        # Count frequencies
        from collections import Counter
        token_counts = Counter(all_tokens)
        
        keywords_df = pd.DataFrame(
            token_counts.most_common(top_n),
            columns=['keyword', 'frequency']
        )
        
        keywords_df['percentage'] = (keywords_df['frequency'] / len(all_tokens)) * 100
        
        return keywords_df
    
    def plot_frequent_keywords(self, top_n: int = 30, save: bool = True):
        """Plot frequent keywords"""
        keywords_df = self.extract_frequent_keywords(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Horizontal bar chart
        axes[0].barh(range(len(keywords_df)), keywords_df['frequency'], 
                    color=sns.color_palette("viridis", len(keywords_df)))
        axes[0].set_yticks(range(len(keywords_df)))
        axes[0].set_yticklabels(keywords_df['keyword'], fontsize=10)
        axes[0].set_xlabel('Frequency')
        axes[0].set_title(f'Top {top_n} Most Frequent Keywords')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Word cloud style visualization
        top_20 = keywords_df.head(20)
        axes[1].bar(range(len(top_20)), top_20['frequency'], 
                   color=sns.color_palette("coolwarm", len(top_20)))
        axes[1].set_xticks(range(len(top_20)))
        axes[1].set_xticklabels(top_20['keyword'], rotation=45, ha='right', fontsize=9)
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Top 20 Keywords Bar Chart')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/frequent_keywords.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_lda(self, num_topics: int = 10, passes: int = 10, 
                  alpha = 'auto', eta = 'auto') -> Any:
        """
        Train LDA model
        
        Args:
            num_topics: Number of topics
            passes: Number of passes through the corpus
            alpha: Alpha parameter ('auto' or float)
            eta: Eta parameter ('auto' or float)
            
        Returns:
            Trained LDA model
        """
        if self.corpus is None:
            self.prepare_corpus()
        
        print(f"Training LDA model with {num_topics} topics...")
        
        # LdaMulticore doesn't support 'auto' for alpha/eta, so use LdaModel if auto is specified
        # Otherwise use LdaMulticore for faster training
        if alpha == 'auto' or eta == 'auto':
            # Use LdaModel for auto-tuning
            self.lda_model = models.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=passes,
                alpha=alpha,
                eta=eta,
                random_state=42
            )
        else:
            # Use LdaMulticore for faster training with fixed alpha/eta
            self.lda_model = models.LdaMulticore(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                passes=passes,
                alpha=alpha,
                eta=eta,
                workers=3,
                random_state=42
            )
        
        print("LDA model trained successfully!")
        return self.lda_model
    
    def get_lda_topics(self, num_words: int = 10) -> Dict:
        """
        Extract topics from LDA model
        
        Args:
            num_words: Number of words per topic
            
        Returns:
            Dictionary with topics
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed. Install it with: pip install gensim")
        
        if self.lda_model is None:
            raise ValueError("LDA model not trained. Call train_lda() first.")
        
        topics = {}
        for idx, topic in self.lda_model.print_topics(num_words=num_words):
            topics[f'Topic_{idx}'] = {
                'words': topic,
                'top_words': [word.split('*')[1].strip('"') 
                            for word in topic.split('+')]
            }
        
        self.topics['lda'] = topics
        return topics
    
    def plot_lda_topics(self, num_words: int = 10, save: bool = True):
        """Visualize LDA topics"""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed. Install it with: pip install gensim")
        
        if self.lda_model is None:
            raise ValueError("LDA model not trained. Call train_lda() first.")
        
        topics = self.get_lda_topics(num_words)
        
        num_topics = len(topics)
        cols = 2
        rows = (num_topics + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if num_topics > 1 else [axes]
        
        for idx, (topic_name, topic_data) in enumerate(topics.items()):
            words = topic_data['top_words'][:num_words]
            # Extract weights (simplified - showing top words only)
            ax = axes[idx]
            
            # Create a simple visualization
            y_pos = np.arange(len(words))
            # Use equal weights for visualization (actual weights are in topic_data['words'])
            weights = [1.0] * len(words)  # Simplified
            
            ax.barh(y_pos, weights, color=sns.color_palette("Set2", len(words)))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=9)
            ax.set_xlabel('Relative Importance')
            ax.set_title(f'{topic_name}', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')
        
        # Hide unused subplots
        for idx in range(len(topics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/lda_topics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _sanitize_pyldavis_data(vis):
        """
        Ensure all numerical coordinates and metrics in pyLDAvis PreparedData
        are standard real floats. This eliminates TypeError: Object of type
        complex128 is not JSON serializable when negative eigenvalues occur in MDS/PCoA.
        """
        if vis is None:
            return vis
        try:
            # Topic coordinates (x, y) often contain complex128 when negative eigenvalues occur
            if hasattr(vis, 'topic_coordinates') and isinstance(vis.topic_coordinates, pd.DataFrame):
                coords = vis.topic_coordinates
                for col in ['x', 'y']:
                    if col in coords.columns:
                        coords[col] = coords[col].apply(
                            lambda v: float(v.real) if hasattr(v, 'real') else (float(np.real(v)) if isinstance(v, (complex, np.number)) else v)
                        ).astype(float)
            
            # Check other dataframes in PreparedData
            for attr in ['topic_info', 'token_table']:
                if hasattr(vis, attr):
                    df = getattr(vis, attr)
                    if isinstance(df, pd.DataFrame):
                        for col in df.columns:
                            if np.iscomplexobj(df[col]):
                                df[col] = np.real(df[col]).astype(float)
        except Exception as err:
            warnings.warn(f"Warning during pyLDAvis data sanitization: {err}")
        return vis

    def create_lda_visualization(self, save: bool = True):
        """Create interactive LDA visualization using pyLDAvis"""
        if not PYLDAVIS_AVAILABLE:
            raise ImportError(
                "pyLDAvis is not installed. Install it with: pip install pyLDAvis"
            )
        
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed. Install it with: pip install gensim")
        
        if self.lda_model is None:
            raise ValueError("LDA model not trained. Call train_lda() first.")
        
        print("Creating LDA visualization...")
        # Attempt preparation with default PCoA; fall back to metric MDS ('mmds') if needed
        try:
            vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary, sort_topics=False)
        except Exception as prep_err:
            print(f"PCoA projection encountered an issue ({prep_err}); falling back to mmds projection...")
            vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary, sort_topics=False, mds='mmds')
        
        # Sanitize any complex128 coordinates so they are strictly JSON-serializable
        vis = self._sanitize_pyldavis_data(vis)
        
        if save:
            try:
                pyLDAvis.save_html(vis, f'{self.output_dir}/lda_visualization.html')
            except TypeError as err:
                if 'complex' in str(err) or 'serializable' in str(err):
                    # Fallback to mmds if serialization still encountered complex types
                    vis = gensimvis.prepare(
                        self.lda_model, self.corpus, self.dictionary, sort_topics=False, mds='mmds'
                    )
                    vis = self._sanitize_pyldavis_data(vis)
                    pyLDAvis.save_html(vis, f'{self.output_dir}/lda_visualization.html')
                else:
                    raise err
            print(f"LDA visualization saved to {self.output_dir}/lda_visualization.html")
        
        return vis
    
    def train_bertopic(self, num_topics: Optional[int] = None, 
                      min_topic_size: int = 10) -> Optional[Any]:
        """
        Train BERTopic model
        
        Args:
            num_topics: Number of topics (None for automatic)
            min_topic_size: Minimum size of a topic
            
        Returns:
            Trained BERTopic model
        """
        if not BERTOPIC_AVAILABLE:
            print("BERTopic not available. Skipping BERTopic training.")
            return None
        
        print("Training BERTopic model...")
        print("This may take a while...")
        
        # Suppress Hugging Face unauthenticated request warnings and set environment
        import os
        import logging
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

        # Use standard text progress bars to prevent VS Code jupyter-ipywidget-renderer errors
        try:
            import tqdm.auto
            import tqdm.autonotebook
            import tqdm.std
            tqdm.auto.tqdm = tqdm.std.tqdm
            tqdm.auto.trange = tqdm.std.trange
            tqdm.autonotebook.tqdm = tqdm.std.tqdm
            tqdm.autonotebook.trange = tqdm.std.trange
        except Exception:
            pass

        # Prepare documents
        documents = self.df['headline'].tolist()
        
        # Initialize and train BERTopic
        self.bertopic_model = BERTopic(
            nr_topics=num_topics,
            min_topic_size=min_topic_size,
            verbose=True
        )
        
        topics, probs = self.bertopic_model.fit_transform(documents)
        
        # Add topics to dataframe
        self.df['bertopic_label'] = topics
        
        print("BERTopic model trained successfully!")
        return self.bertopic_model
    
    def get_bertopic_topics(self) -> Dict:
        """Extract topics from BERTopic model"""
        if self.bertopic_model is None:
            raise ValueError("BERTopic model not trained. Call train_bertopic() first.")
        
        topic_info = self.bertopic_model.get_topic_info()
        topics = {}
        
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            topic_words = self.bertopic_model.get_topic(topic_id)
            
            topics[f'Topic_{topic_id}'] = {
                'count': row['Count'],
                'name': row['Name'] if 'Name' in row else f'Topic {topic_id}',
                'words': [word[0] for word in topic_words],
                'scores': [word[1] for word in topic_words]
            }
        
        self.topics['bertopic'] = topics
        return topics
    
    def plot_bertopic_topics(self, save: bool = True):
        """Visualize BERTopic topics"""
        if self.bertopic_model is None:
            raise ValueError("BERTopic model not trained. Call train_bertopic() first.")
        
        try:
            fig = self.bertopic_model.visualize_topics()
            if save:
                fig.write_html(f'{self.output_dir}/bertopic_topics.html')
            
            fig = self.bertopic_model.visualize_barchart(top_n_topics=10)
            if save:
                fig.write_html(f'{self.output_dir}/bertopic_barchart.html')
        except Exception as e:
            print(f"Error creating BERTopic visualizations: {e}")
    
    def assign_topics_to_articles(self, method: str = 'lda') -> pd.DataFrame:
        """
        Assign topic labels to articles
        
        Args:
            method: 'lda' or 'bertopic'
            
        Returns:
            DataFrame with topic assignments
        """
        if method == 'lda' and self.lda_model is None:
            raise ValueError("LDA model not trained.")
        if method == 'bertopic' and self.bertopic_model is None:
            raise ValueError("BERTopic model not trained.")
        
        df_with_topics = self.df.copy()
        
        if method == 'lda':
            # Get topic distribution for each document
            topic_distributions = []
            for text in self.processed_texts:
                bow = self.dictionary.doc2bow(text)
                topic_dist = self.lda_model[bow]
                # Get dominant topic
                dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
                topic_distributions.append(dominant_topic)
            
            df_with_topics['lda_topic'] = topic_distributions
        
        elif method == 'bertopic':
            # Already assigned during training
            if 'bertopic_label' not in df_with_topics.columns:
                documents = df_with_topics['headline'].tolist()
                topics, _ = self.bertopic_model.transform(documents)
                df_with_topics['bertopic_label'] = topics
        
        return df_with_topics
    
    def identify_topic_categories(self) -> Dict:
        """
        Identify common financial topic categories
        
        Returns:
            Dictionary mapping topics to categories
        """
        categories = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarter', 'eps', 'guidance'],
            'mergers': ['merger', 'acquisition', 'deal', 'buyout', 'takeover', 'merge'],
            'fda_approval': ['fda', 'approval', 'drug', 'clinical', 'trial', 'regulatory'],
            'price_target': ['target', 'price', 'upgrade', 'downgrade', 'rating', 'analyst'],
            'product_launch': ['launch', 'product', 'release', 'announce', 'unveil'],
            'partnership': ['partnership', 'partner', 'collaboration', 'agreement', 'deal'],
            'regulation': ['regulation', 'regulatory', 'sec', 'investigation', 'lawsuit'],
            'market_movement': ['surge', 'plunge', 'rally', 'crash', 'volatility', 'trading']
        }
        
        return categories
    
    def compute_coherence(self, coherence_type: str = 'c_v') -> float:
        """
        Compute coherence score for the trained LDA model
        
        Args:
            coherence_type: Type of coherence measure ('c_v', 'c_uci', 'c_npmi')
            
        Returns:
            Coherence score (higher is better, typically 0-1)
            
        Note:
            - c_v: Uses normalized pointwise mutual information and cosine similarity
            - c_uci: Uses conditional probability and log ratio
            - c_npmi: Normalized PMI, better for shorter texts
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is not installed. Install it with: pip install gensim")
        
        if self.lda_model is None:
            raise ValueError("LDA model not trained. Call train_lda() first.")
        
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=self.processed_texts,
            dictionary=self.dictionary,
            coherence=coherence_type
        )
        
        coherence_score = coherence_model.get_coherence()
        return coherence_score


def run_topic_modeling(df: pd.DataFrame, output_dir: str = "output",
                       num_topics: int = 10, min_topic_size: int = 10,
                       run_lda: bool = True, run_bertopic: bool = False) -> Dict:
    """
    Convenience function to run complete topic modeling pipeline
    
    Args:
        df: DataFrame with 'headline' column
        output_dir: Directory for outputs
        num_topics: Number of topics for LDA
        min_topic_size: Minimum topic size for BERTopic
        run_lda: Whether to run LDA
        run_bertopic: Whether to run BERTopic
        
    Returns:
        Dictionary with trained models and results
    """
    modeler = TopicModeler(df, output_dir)
    results = {'modeler': modeler}
    
    # Prepare corpus
    modeler.prepare_corpus()
    
    # Extract keywords
    keywords = modeler.extract_frequent_keywords(50)
    modeler.plot_frequent_keywords(30)
    results['keywords'] = keywords
    
    # LDA
    if run_lda:
        lda_model = modeler.train_lda(num_topics=num_topics, passes=10)
        lda_topics = modeler.get_lda_topics(num_words=10)
        coherence = modeler.compute_coherence()
        modeler.plot_lda_topics(num_words=10)
        results['lda_model'] = lda_model
        results['lda_topics'] = lda_topics
        results['coherence_score'] = coherence
        print(f"LDA Coherence Score (c_v): {coherence:.4f}")
        
        try:
            modeler.create_lda_visualization()
        except Exception as e:
            print(f"LDA visualization skipped: {e}")
    
    # BERTopic
    if run_bertopic:
        try:
            bertopic_model = modeler.train_bertopic(min_topic_size=min_topic_size)
            if bertopic_model:
                bertopic_topics = modeler.get_bertopic_topics()
                modeler.plot_bertopic_topics()
                results['bertopic_model'] = bertopic_model
                results['bertopic_topics'] = bertopic_topics
        except Exception as e:
            print(f"BERTopic skipped: {e}")
    
    return results

