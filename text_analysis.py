import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Dict
import re
import nltk
from semantic_analyzer import SemanticAnalyzer

class TextAnalyzer:
    def __init__(self):
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.semantic_analyzer = SemanticAnalyzer()

        # Enhanced vectorization parameters
        self.description_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=15000,
            ngram_range=(1, 4),
            min_df=1,
            max_df=0.99,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

        self.eligibility_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=7000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.99,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

        self.additional_eligibility_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.99,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

        # Initialize matrices and data
        self.description_matrix = None
        self.eligibility_matrix = None
        self.additional_eligibility_matrix = None
        self.opportunity_numbers = None
        self.assistance_listings = None
        self.descriptions = None
        self.description_nn = None

    def _download_nltk_data(self):
        """Download required NLTK data with error handling."""
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            print(f"Error downloading NLTK data: {str(e)}")
            raise RuntimeError("Failed to download required NLTK data.")

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for funding descriptions."""
        try:
            text = str(text).lower()
            text = re.sub(r'[^\w\s$%()-]', ' ', text)
            text = re.sub(r'\d+', 'NUM', text)
            text = re.sub(r'pa-\d+', 'PA_NUMBER', text)
            text = re.sub(r'r\d+', 'R_NUMBER', text)

            tokens = word_tokenize(text)
            tokens = [t for t in tokens if (
                t.isalnum() or t in {'$', '%', '(', ')', '-'} and
                t not in self.stop_words and
                len(t) > 1
            )]

            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            return text

    def fit_transform_data(self, descriptions: List[str], eligibilities: List[str], 
                         additional_eligibilities: List[str], opportunity_numbers: List[str],
                         assistance_listings: List[str]) -> None:
        """Fit and transform all text data."""
        try:
            # Store metadata
            self.opportunity_numbers = opportunity_numbers
            self.assistance_listings = assistance_listings
            self.descriptions = descriptions

            # Process descriptions
            processed_descriptions = [self.preprocess_text(desc) for desc in descriptions]
            self.description_matrix = self.description_vectorizer.fit_transform(processed_descriptions)

            # Process eligibilities
            processed_eligibilities = [self.preprocess_text(elig) for elig in eligibilities]
            self.eligibility_matrix = self.eligibility_vectorizer.fit_transform(processed_eligibilities)

            # Process additional eligibility
            processed_add_elig = [self.preprocess_text(add_elig) for add_elig in additional_eligibilities]
            self.additional_eligibility_matrix = self.additional_eligibility_vectorizer.fit_transform(processed_add_elig)

            # Initialize nearest neighbors
            self.description_nn = NearestNeighbors(
                n_neighbors=min(100, len(descriptions)),
                metric='cosine',
                algorithm='brute'
            )
            self.description_nn.fit(self.description_matrix)

        except Exception as e:
            print(f"Error in fit_transform: {str(e)}")
            raise

    def calculate_vector_similarity(self, matrix1, matrix2) -> float:
        """Calculate cosine similarity between two sparse matrices."""
        similarity = cosine_similarity(matrix1, matrix2)[0][0]
        return float(similarity)

    def find_similar_opportunities(self, query_desc: str, query_elig: str, query_add_elig: str, 
                                 current_opp_number: str) -> List[Tuple[int, Dict]]:
        """Find similar opportunities using enhanced vector embeddings."""
        if not self.description_nn:
            raise ValueError("Model not fitted. Call fit_transform_data first.")

        try:
            # Process queries
            processed_desc = self.preprocess_text(query_desc)
            processed_elig = self.preprocess_text(query_elig)
            processed_add_elig = self.preprocess_text(query_add_elig)

            # Create query vectors
            query_desc_vector = self.description_vectorizer.transform([processed_desc])
            query_elig_vector = self.eligibility_vectorizer.transform([processed_elig])
            query_add_elig_vector = self.additional_eligibility_vectorizer.transform([processed_add_elig])

            # Find nearest neighbors
            distances, indices = self.description_nn.kneighbors(query_desc_vector)

            # Calculate similarities
            similar_opportunities = []
            for idx, desc_dist in zip(indices[0], distances[0]):
                # Skip self-matches
                if self.opportunity_numbers[idx] == current_opp_number:
                    continue

                # Calculate vector-based similarities
                desc_sim = 1 - desc_dist
                elig_sim = self.calculate_vector_similarity(
                    query_elig_vector,
                    self.eligibility_matrix[idx:idx+1]
                )
                add_elig_sim = self.calculate_vector_similarity(
                    query_add_elig_vector,
                    self.additional_eligibility_matrix[idx:idx+1]
                )

                # Calculate semantic similarity
                semantic_analysis = self.semantic_analyzer.analyze_opportunity_pair(
                    query_desc,
                    self.descriptions[idx]
                )
                semantic_sim = semantic_analysis['semantic_similarity']

                # Calculate combined score with semantic weight
                # Vector similarities: 70% (Description: 35%, Eligibility: 25%, Additional: 10%)
                # Semantic similarity: 30%
                combined_score = (
                    0.35 * desc_sim +
                    0.25 * elig_sim +
                    0.10 * add_elig_sim +
                    0.30 * semantic_sim
                )

                if combined_score > 0.01:  # 1% minimum similarity threshold
                    # Get semantic explanation
                    semantic_explanation = self.semantic_analyzer.get_semantic_explanation(
                        semantic_analysis['source_concepts'],
                        semantic_analysis['target_concepts']
                    )

                    similar_opportunities.append((
                        int(idx),
                        {
                            'opportunity_number': self.opportunity_numbers[idx],
                            'description_similarity': float(desc_sim),
                            'eligibility_similarity': float(elig_sim),
                            'additional_eligibility_similarity': float(add_elig_sim),
                            'semantic_similarity': float(semantic_sim),
                            'combined_score': float(combined_score),
                            'assistance_listing': self.assistance_listings[idx],
                            'semantic_explanation': semantic_explanation
                        }
                    ))

            # Sort by combined score
            similar_opportunities.sort(key=lambda x: x[1]['combined_score'], reverse=True)
            return similar_opportunities

        except Exception as e:
            print(f"Error finding similar opportunities: {str(e)}")
            return []
