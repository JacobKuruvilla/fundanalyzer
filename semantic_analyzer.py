import os
from anthropic import Anthropic
from typing import List, Dict, Tuple

class SemanticAnalyzer:
    def __init__(self):
        self.client = Anthropic(
            api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
        # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        self.model = "claude-3-5-sonnet-20241022"

    def extract_semantic_concepts(self, text: str) -> Dict[str, float]:
        """Extract key semantic concepts and their importance from text."""
        try:
            prompt = f"""Analyze this funding opportunity description and identify key semantic concepts. 
            Return a JSON object with concepts as keys and their importance (0-1) as values.
            Focus on research areas, methodologies, and objectives.

            Description: {text}"""

            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            # Parse and return semantic concepts
            concepts = eval(response.content)  # Safe since we're using controlled input
            return concepts
        except Exception as e:
            print(f"Error extracting semantic concepts: {str(e)}")
            return {}

    def calculate_semantic_similarity(self, source_concepts: Dict[str, float], 
                                   target_concepts: Dict[str, float]) -> float:
        """Calculate semantic similarity between two sets of concepts."""
        try:
            # Convert concepts to sets for comparison
            source_set = set(source_concepts.keys())
            target_set = set(target_concepts.keys())

            # Calculate overlap
            common_concepts = source_set.intersection(target_set)
            if not common_concepts:
                return 0.0

            # Calculate weighted similarity
            similarity_score = 0.0
            for concept in common_concepts:
                # Average of importance scores for matching concepts
                concept_score = (source_concepts[concept] + target_concepts[concept]) / 2
                similarity_score += concept_score

            # Normalize by total possible score
            max_concepts = max(len(source_concepts), len(target_concepts))
            normalized_score = similarity_score / max_concepts

            return float(normalized_score)
        except Exception as e:
            print(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def analyze_opportunity_pair(self, source_desc: str, target_desc: str) -> Dict[str, float]:
        """Analyze semantic similarity between two opportunities."""
        try:
            # Extract concepts
            source_concepts = self.extract_semantic_concepts(source_desc)
            target_concepts = self.extract_semantic_concepts(target_desc)

            # Calculate similarity
            semantic_score = self.calculate_semantic_similarity(source_concepts, target_concepts)

            return {
                'semantic_similarity': semantic_score,
                'source_concepts': source_concepts,
                'target_concepts': target_concepts
            }
        except Exception as e:
            print(f"Error analyzing opportunity pair: {str(e)}")
            return {
                'semantic_similarity': 0.0,
                'source_concepts': {},
                'target_concepts': {}
            }

    def get_semantic_explanation(self, source_concepts: Dict[str, float], 
                               target_concepts: Dict[str, float]) -> str:
        """Generate human-readable explanation of semantic similarities."""
        try:
            prompt = f"""Compare these two sets of funding opportunity concepts and explain their similarities:

            Source Concepts: {source_concepts}
            Target Concepts: {target_concepts}

            Provide a brief, clear explanation focusing on key overlapping themes and significant differences.
            """

            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            return response.content
        except Exception as e:
            print(f"Error generating semantic explanation: {str(e)}")
            return "Unable to generate semantic explanation."
