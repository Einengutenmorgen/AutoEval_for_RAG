#!/usr/bin/env python3
"""
Bias analysis tool for RAG evaluation datasets and results.
Detects potential biases in question distribution and system performance.
"""

import os
import sys
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/bias_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'get_system_info'):  # For RAGConnector
            return obj.get_system_info()
        return super().default(obj)


class RAGBiasAnalyzer:
    """Tool for detecting and analyzing biases in RAG evaluation."""
    
    def __init__(self, 
                questions_dir: str = "question_sets/validated_questions",
                results_dir: str = "evaluation/results",
                reports_dir: str = "reports/bias_analysis"):
        """
        Initialize the bias analyzer.
        
        Args:
            questions_dir: Directory containing question files
            results_dir: Directory containing evaluation results
            reports_dir: Directory to save bias analysis reports
        """
        self.questions_dir = Path(questions_dir)
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        
        # Create reports directory if it doesn't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.questions = []
        self.results = {}
        
        # Analysis results
        self.dataset_analysis = {}
        self.performance_analysis = {}
    
    def load_questions(self, questions_file: Optional[str] = None) -> None:
        """
        Load questions from file or directory.
        
        Args:
            questions_file: Path to specific questions file or None to load all
        """
        self.questions = []
        
        if questions_file:
            # Load from specific file
            file_path = Path(questions_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Questions file not found: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_questions = json.load(f)
                
                if isinstance(file_questions, dict):
                    self.questions.append(file_questions)
                else:
                    self.questions.extend(file_questions)
                
                logger.info(f"Loaded {len(self.questions)} questions from {file_path}")
            except Exception as e:
                logger.error(f"Error loading questions from {file_path}: {e}")
        else:
            # Load from all files in questions directory
            for file_path in self.questions_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_questions = json.load(f)
                    
                    if isinstance(file_questions, dict):
                        self.questions.append(file_questions)
                    else:
                        self.questions.extend(file_questions)
                    
                    logger.info(f"Loaded {len(file_questions) if isinstance(file_questions, list) else 1} questions from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading questions from {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.questions)} questions in total")
    
    def load_evaluation_results(self, results_file: Optional[str] = None) -> None:
        """
        Load evaluation results.
        
        Args:
            results_file: Path to specific results file or None to load all
        """
        self.results = {}
        
        if results_file:
            # Load from specific file
            file_path = Path(results_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Results file not found: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                system_name = results["system"]["name"]
                self.results[system_name] = results
                
                logger.info(f"Loaded results for system '{system_name}' from {file_path}")
            except Exception as e:
                logger.error(f"Error loading results from {file_path}: {e}")
        else:
            # Load from all files in results directory
            for file_path in self.results_dir.glob("*_results_*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    system_name = results["system"]["name"]
                    self.results[system_name] = results
                    
                    logger.info(f"Loaded results for system '{system_name}' from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading results from {file_path}: {e}")
        
        logger.info(f"Loaded results for {len(self.results)} systems in total")
    
    def analyze_dataset_bias(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze potential biases in the question dataset.
        
        Args:
            output_file: Path to save analysis results
            
        Returns:
            Dataset bias analysis results
        """
        if not self.questions:
            logger.warning("No questions loaded for analysis")
            return {}
        
        logger.info("Analyzing dataset bias...")
        
        # Initialize analysis
        analysis = {
            "dataset_size": len(self.questions),
            "distributions": {},
            "correlations": {},
            "diversity": {},
            "potential_biases": []
        }
        
        # Analyze distributions of metadata fields
        metadata_fields = [
            "query_type",
            "complexity",
            "special_categories"
        ]
        
        # Extract paper-specific fields if available
        if self.questions and "paper_id" in self.questions[0]:
            metadata_fields.append("paper_id")
        
        # Calculate distributions
        for field in metadata_fields:
            distribution = self._calculate_distribution(field)
            analysis["distributions"][field] = distribution
            
            # Check for uneven distribution
            if field != "special_categories":  # Skip special categories as they're optional
                evenness = self._calculate_evenness(distribution)
                analysis["diversity"][f"{field}_evenness"] = evenness
                
                # Flag potential bias if distribution is highly uneven
                if evenness < 0.7:  # Arbitrary threshold
                    dominant_category = max(distribution.items(), key=lambda x: x[1]["count"])[0]
                    analysis["potential_biases"].append({
                        "field": field,
                        "issue": "uneven_distribution",
                        "evenness_score": evenness,
                        "dominant_category": dominant_category,
                        "dominant_percentage": distribution[dominant_category]["percentage"]
                    })
        
        # Analyze cross-correlations between fields
        for i, field1 in enumerate(metadata_fields[:-1]):
            if field1 == "special_categories":
                continue  # Skip special categories for correlations
                
            for field2 in metadata_fields[i+1:]:
                if field2 == "special_categories":
                    continue
                
                correlation = self._calculate_field_correlation(field1, field2)
                analysis["correlations"][f"{field1}_{field2}"] = correlation
                
                # Flag potential bias if correlation is strong
                if correlation["cramers_v"] > 0.5:  # Arbitrary threshold
                    analysis["potential_biases"].append({
                        "field": f"{field1}_{field2}",
                        "issue": "strong_correlation",
                        "correlation_measure": "cramers_v",
                        "correlation_value": correlation["cramers_v"]
                    })
        
        # Analyze text characteristics
        text_analysis = self._analyze_text_characteristics()
        analysis["text_characteristics"] = text_analysis
        
        # Flag potential text biases
        for measure, value in text_analysis["averages"].items():
            if measure == "question_length_chars" and value < 20:
                analysis["potential_biases"].append({
                    "field": "question_text",
                    "issue": "too_short_questions",
                    "average_length": value
                })
            elif measure == "question_length_words" and value < 5:
                analysis["potential_biases"].append({
                    "field": "question_text",
                    "issue": "too_few_words",
                    "average_word_count": value
                })
        
        # Save the dataset analysis
        self.dataset_analysis = analysis
        
        # Generate visualizations
        if output_file:
            self._generate_dataset_report(output_file)
        
        logger.info("Dataset bias analysis complete")
        
        return analysis
    
    def _calculate_distribution(self, field: str) -> Dict[str, Dict[str, Any]]:
        """
        Calculate distribution of values for a metadata field.
        
        Args:
            field: Metadata field name
            
        Returns:
            Distribution dictionary
        """
        counts = Counter()
        total = 0
        
        for question in self.questions:
            metadata = question.get("metadata", {})
            
            if field == "special_categories":
                # Special case for special_categories which is a list
                categories = metadata.get(field, [])
                for category in categories:
                    counts[category] += 1
                    total += 1
                
                # Also count questions with no special categories
                if not categories:
                    counts["none"] += 1
                    total += 1
            else:
                # Regular metadata field
                value = metadata.get(field, "unknown")
                if field == "paper_id":
                    value = question.get("paper_id", "unknown")
                
                counts[value] += 1
                total += 1
        
        # Convert to percentage distribution
        distribution = {}
        for value, count in counts.items():
            distribution[value] = {
                "count": count,
                "percentage": count / total if total > 0 else 0
            }
        
        return distribution
    
    def _calculate_evenness(self, distribution: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate evenness of a distribution (normalized entropy).
        
        Args:
            distribution: Distribution dictionary
            
        Returns:
            Evenness score between 0 and 1
        """
        values = [item["count"] for item in distribution.values()]
        total = sum(values)
        
        if total == 0 or len(values) <= 1:
            return 1.0  # Only one category or no data
        
        # Calculate Shannon entropy
        probabilities = [v / total for v in values if v > 0]
        entropy = -sum(p * np.log(p) for p in probabilities)
        
        # Normalize by maximum entropy (log of number of categories)
        max_entropy = np.log(len(values))
        evenness = entropy / max_entropy if max_entropy > 0 else 1.0
        
        return evenness
    
    def _calculate_field_correlation(self, field1: str, field2: str) -> Dict[str, Any]:
        """
        Calculate correlation between two metadata fields using contingency table.
        
        Args:
            field1: First metadata field
            field2: Second metadata field
            
        Returns:
            Correlation dictionary
        """
        # Create contingency table
        field1_values = set()
        field2_values = set()
        counts = defaultdict(lambda: defaultdict(int))
        
        for question in self.questions:
            metadata = question.get("metadata", {})
            
            # Get values for each field
            value1 = metadata.get(field1, "unknown")
            if field1 == "paper_id":
                value1 = question.get("paper_id", "unknown")
                
            value2 = metadata.get(field2, "unknown")
            if field2 == "paper_id":
                value2 = question.get("paper_id", "unknown")
            
            field1_values.add(value1)
            field2_values.add(value2)
            counts[value1][value2] += 1
        
        # Convert to numpy array
        field1_values = sorted(field1_values)
        field2_values = sorted(field2_values)
        
        contingency_table = np.zeros((len(field1_values), len(field2_values)))
        for i, v1 in enumerate(field1_values):
            for j, v2 in enumerate(field2_values):
                contingency_table[i, j] = counts[v1][v2]
        
        # Calculate Cramer's V (measure of association between categorical variables)
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        
        if n == 0 or min_dim == 0:
            cramers_v = 0
        else:
            cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        return {
            "contingency_table": contingency_table.tolist(),
            "field1_values": field1_values,
            "field2_values": field2_values,
            "cramers_v": cramers_v
        }
    
    def _analyze_text_characteristics(self) -> Dict[str, Any]:
        """
        Analyze text characteristics of questions.
        
        Returns:
            Text analysis dictionary
        """
        analysis = {
            "averages": {},
            "distributions": {},
            "by_category": {}
        }
        
        # Calculate overall averages
        question_lengths_chars = []
        question_lengths_words = []
        
        for question in self.questions:
            question_text = question["question"]
            
            # Calculate lengths
            char_length = len(question_text)
            word_length = len(question_text.split())
            
            question_lengths_chars.append(char_length)
            question_lengths_words.append(word_length)
        
        # Calculate averages
        analysis["averages"]["question_length_chars"] = np.mean(question_lengths_chars) if question_lengths_chars else 0
        analysis["averages"]["question_length_words"] = np.mean(question_lengths_words) if question_lengths_words else 0
        
        # Calculate distributions
        analysis["distributions"]["question_length_chars"] = {
            "min": min(question_lengths_chars) if question_lengths_chars else 0,
            "max": max(question_lengths_chars) if question_lengths_chars else 0,
            "median": np.median(question_lengths_chars) if question_lengths_chars else 0,
            "std_dev": np.std(question_lengths_chars) if question_lengths_chars else 0
        }
        
        analysis["distributions"]["question_length_words"] = {
            "min": min(question_lengths_words) if question_lengths_words else 0,
            "max": max(question_lengths_words) if question_lengths_words else 0,
            "median": np.median(question_lengths_words) if question_lengths_words else 0,
            "std_dev": np.std(question_lengths_words) if question_lengths_words else 0
        }
        
        # Calculate by query type
        by_query_type = defaultdict(lambda: {"chars": [], "words": []})
        
        for question in self.questions:
            query_type = question.get("metadata", {}).get("query_type", "unknown")
            
            by_query_type[query_type]["chars"].append(len(question["question"]))
            by_query_type[query_type]["words"].append(len(question["question"].split()))
        
        # Calculate averages by query type
        for query_type, data in by_query_type.items():
            analysis["by_category"][query_type] = {
                "avg_length_chars": np.mean(data["chars"]) if data["chars"] else 0,
                "avg_length_words": np.mean(data["words"]) if data["words"] else 0
            }
        
        return analysis
    
    def _generate_dataset_report(self, output_file: str) -> None:
        """
        Generate dataset bias analysis report with visualizations.
        
        Args:
            output_file: Path to save the report
        """
        # Check if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except ImportError:
            logger.warning("Matplotlib not available for visualizations")
            
            # Save JSON report only
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.dataset_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Dataset analysis saved to {output_file}")
            return
        
        # Create base output filename without extension
        output_base = str(Path(output_file).with_suffix(''))
        
        # Save JSON report
        with open(f"{output_base}.json", 'w', encoding='utf-8') as f:
            json.dump(self.dataset_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Create visualizations
        try:
            self._create_dataset_visualizations(output_base)
            logger.info(f"Dataset analysis and visualizations saved to {output_base}.*")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_dataset_visualizations(self, output_base: str) -> None:
        """
        Create visualizations for dataset analysis.
        
        Args:
            output_base: Base filename for output files
        """
        # Set style
        plt.style.use('ggplot')
        
        # Distribution plots
        for field, distribution in self.dataset_analysis["distributions"].items():
            if field == "special_categories":
                # Skip special categories visualization for now
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Sort by count descending
            categories = sorted(distribution.keys(), key=lambda x: distribution[x]["count"], reverse=True)
            counts = [distribution[cat]["count"] for cat in categories]
            
            # Create bar chart
            bars = plt.bar(range(len(categories)), counts, color='skyblue')
            plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}',
                        ha='center', va='bottom')
            
            plt.title(f"Distribution of {field}")
            plt.xlabel(field.capitalize())
            plt.ylabel("Count")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{output_base}_{field}_distribution.png")
            plt.close()
        
        # Correlation heatmap if we have correlations
        if self.dataset_analysis["correlations"]:
            plt.figure(figsize=(10, 8))
            
            # Extract correlation values
            fields = []
            values = []
            
            for key, data in self.dataset_analysis["correlations"].items():
                logger.info(f"Ky {key}: {data}")
                if isinstance(data, dict) and "field1_values" in data and "field2_values" in data and "cramers_v" in data:
                    field1, field2 = key.split('_', 1)  # More robust splitting, by limiting the amount 
                    fields.append((field1, field2))
                    values.append(data["cramers_v"])
                else:
                    logger.warning(f"Skipping malformed correlation data for {key}")
            
            # Create unique list of all fields
            all_fields = sorted(set([f for pair in fields for f in pair]))
            
            # Create correlation matrix
            corr_matrix = np.zeros((len(all_fields), len(all_fields)))
            
            for (f1, f2), value in zip(fields, values):
                i = all_fields.index(f1)
                j = all_fields.index(f2)
                corr_matrix[i, j] = value
                corr_matrix[j, i] = value  # Matrix is symmetric
            
            # Add 1s on the diagonal
            for i in range(len(all_fields)):
                corr_matrix[i, i] = 1.0
            
            # Create heatmap
            ax = sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', xticklabels=all_fields, yticklabels=all_fields)
            plt.title("Correlations between Metadata Fields (Cramer's V)")
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{output_base}_correlations.png")
            plt.close()
        
        # Text characteristics
        if "text_characteristics" in self.dataset_analysis:
            # Create histogram of question lengths
            question_lengths = []
            for question in self.questions:
                question_lengths.append(len(question["question"]))
            
            plt.figure(figsize=(10, 6))
            plt.hist(question_lengths, bins=20, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(question_lengths), color='red', linestyle='dashed', linewidth=1)
            plt.text(np.mean(question_lengths) * 1.1, plt.ylim()[1] * 0.9, f'Mean: {np.mean(question_lengths):.1f}')
            
            plt.title("Distribution of Question Lengths (Characters)")
            plt.xlabel("Number of Characters")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{output_base}_question_lengths.png")
            plt.close()
            
            # Create bar chart of average lengths by query type
            if "by_category" in self.dataset_analysis["text_characteristics"]:
                plt.figure(figsize=(12, 6))
                
                query_types = []
                avg_chars = []
                avg_words = []
                
                for qt, data in self.dataset_analysis["text_characteristics"]["by_category"].items():
                    query_types.append(qt)
                    avg_chars.append(data["avg_length_chars"])
                    avg_words.append(data["avg_length_words"] * 5)  # Scale words for better visualization
                
                # Sort by average character length
                sorted_indices = np.argsort(avg_chars)[::-1]
                query_types = [query_types[i] for i in sorted_indices]
                avg_chars = [avg_chars[i] for i in sorted_indices]
                avg_words = [avg_words[i] for i in sorted_indices]
                
                x = np.arange(len(query_types))
                width = 0.35
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot character lengths
                bars1 = ax1.bar(x - width/2, avg_chars, width, label='Characters', color='skyblue')
                ax1.set_ylabel('Average Characters')
                ax1.set_title('Average Question Length by Query Type')
                
                # Create second y-axis for words
                ax2 = ax1.twinx()
                bars2 = ax2.bar(x + width/2, [w/5 for w in avg_words], width, label='Words', color='lightgreen')
                ax2.set_ylabel('Average Words')
                
                # Set x-axis
                ax1.set_xticks(x)
                ax1.set_xticklabels(query_types, rotation=45, ha='right')
                
                # Add legend
                ax1.legend([bars1, bars2], ['Characters', 'Words'], loc='upper right')
                
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{output_base}_lengths_by_query_type.png")
                plt.close()
    
    def analyze_performance_bias(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze biases in system performance across question categories.
        
        Args:
            output_file: Path to save analysis results
            
        Returns:
            Performance bias analysis results
        """
        if not self.results:
            logger.warning("No evaluation results loaded for analysis")
            return {}
        
        logger.info("Analyzing performance bias...")
        
        # Initialize analysis
        analysis = {
            "systems": list(self.results.keys()),
            "performance_by_category": {},
            "statistical_tests": {},
            "potential_biases": []
        }
        
        # Metadata fields to analyze
        metadata_fields = [
            "query_type",
            "complexity",
            "special_categories"
        ]
        
        # Extract paper-specific fields if available
        if self.results and "responses" in next(iter(self.results.values())):
            first_response = self.results[list(self.results.keys())[0]]["responses"][0]
            if "question" in first_response and "paper_id" in first_response["question"]:
                metadata_fields.append("paper_id")
        
        # Analyze performance by category for each system
        for system_name, results in self.results.items():
            analysis["performance_by_category"][system_name] = {}
            
            # Get responses
            responses = results.get("responses", [])
            successful_responses = [r for r in responses if r.get("success", False)]
            
            if not successful_responses:
                logger.warning(f"No successful responses for system '{system_name}'")
                continue
            
            # Calculate performance scores for each response
            # We'll use the system's reported metrics when available
            response_scores = self._calculate_response_scores(successful_responses)
            
            # Analyze by metadata field
            for field in metadata_fields:
                analysis["performance_by_category"][system_name][field] = self._analyze_performance_by_field(
                    field, successful_responses, response_scores
                )
            
            # Statistical tests for significant differences
            analysis["statistical_tests"][system_name] = {}
            
            for field in metadata_fields:
                if field == "special_categories":
                    continue  # Skip special categories for statistical tests
                
                # Perform ANOVA test to check for significant differences
                try:
                    field_data = analysis["performance_by_category"][system_name][field]
                    
                    if len(field_data) > 1:  # Need at least 2 categories for testing
                        groups = []
                        group_names = []
                        
                        for category, data in field_data.items():
                            if "individual_scores" in data and data["individual_scores"]:
                                groups.append(data["individual_scores"])
                                group_names.append(category)
                        
                        if len(groups) > 1:  # Need at least 2 groups with data
                            # Perform ANOVA
                            f_val, p_val = stats.f_oneway(*groups)
                            
                            analysis["statistical_tests"][system_name][field] = {
                                "test": "ANOVA",
                                "f_value": float(f_val),
                                "p_value": float(p_val),
                                "significant": p_val < 0.05
                            }
                            
                            # Flag potential bias if difference is significant
                            if p_val < 0.05:
                                # Find best and worst performing categories
                                category_scores = [(cat, data["average_score"]) 
                                                 for cat, data in field_data.items()]
                                
                                if category_scores:
                                    best_category = max(category_scores, key=lambda x: x[1])[0]
                                    worst_category = min(category_scores, key=lambda x: x[1])[0]
                                    
                                    analysis["potential_biases"].append({
                                        "system": system_name,
                                        "field": field,
                                        "issue": "performance_disparity",
                                        "p_value": float(p_val),
                                        "best_category": best_category,
                                        "worst_category": worst_category,
                                        "performance_gap": field_data[best_category]["average_score"] - 
                                                         field_data[worst_category]["average_score"]
                                    })
                
                except Exception as e:
                    logger.warning(f"Error performing statistical test for {field}: {e}")
        
        # Save the performance analysis
        self.performance_analysis = analysis
        
        # Generate visualizations
        if output_file:
            self._generate_performance_report(output_file)
        
        logger.info("Performance bias analysis complete")
        
        return analysis
    
    def _calculate_response_scores(self, responses: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate performance scores for responses.
        
        Args:
            responses: List of successful responses
            
        Returns:
            List of scores
        """
        scores = []
        
        for response in responses:
            # Try to extract ROUGE score or other metrics
            score = None
            
            # Check if a score is already available in the response
            if "score" in response:
                score = response["score"]
            elif "evaluation" in response and "score" in response["evaluation"]:
                score = response["evaluation"]["score"]
            elif "metrics" in response and "rouge" in response["metrics"]:
                # Use ROUGE-L F1 if available
                rouge_metrics = response["metrics"]["rouge"]
                if "rougeL_fmeasure" in rouge_metrics:
                    score = rouge_metrics["rougeL_fmeasure"]
            
            # If no score is available, use a basic text similarity
            if score is None and "answer" in response and "question" in response:
                gold_answer = response["question"].get("gold_answer", {}).get("text", "")
                pred_answer = response.get("answer", "")
                
                if gold_answer and pred_answer:
                    score = self._calculate_text_similarity(gold_answer, pred_answer)
            
            # Default score if we couldn't calculate one
            if score is None:
                score = 0.5  # Neutral score
            
            scores.append(score)
        
        return scores
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on token overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize and lowercase
        tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0
        
        # Calculate Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def _analyze_performance_by_field(self, field: str, 
                                    responses: List[Dict[str, Any]],
                                    scores: List[float]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by metadata field.
        
        Args:
            field: Metadata field to analyze
            responses: List of successful responses
            scores: List of performance scores
            
        Returns:
            Analysis by category
        """
        # Group responses by field value
        grouped_responses = defaultdict(list)
        grouped_scores = defaultdict(list)
        
        for response, score in zip(responses, scores):
            question = response.get("question", {})
            metadata = question.get("metadata", {})
            
            if field == "special_categories":
                # Special case for special_categories which is a list
                categories = metadata.get(field, [])
                
                if not categories:
                    grouped_responses["none"].append(response)
                    grouped_scores["none"].append(score)
                else:
                    for category in categories:
                        grouped_responses[category].append(response)
                        grouped_scores[category].append(score)
            else:
                # Regular metadata field
                value = metadata.get(field, "unknown")
                if field == "paper_id":
                    value = question.get("paper_id", "unknown")
                
                grouped_responses[value].append(response)
                grouped_scores[value].append(score)
        
        # Calculate statistics for each group
        analysis = {}
        
        for category, category_scores in grouped_scores.items():
            if not category_scores:
                continue
                
            analysis[category] = {
                "count": len(category_scores),
                "average_score": np.mean(category_scores) if category_scores else 0,
                "median_score": np.median(category_scores) if category_scores else 0,
                "min_score": min(category_scores) if category_scores else 0,
                "max_score": max(category_scores) if category_scores else 0,
                "std_dev": np.std(category_scores) if len(category_scores) > 1 else 0,
                "individual_scores": category_scores
            }
        
        return analysis
    
    def _generate_performance_report(self, output_file: str) -> None:
        """
        Generate performance bias analysis report with visualizations.
        
        Args:
            output_file: Path to save the report
        
        """
        output_base = str(Path(output_file).with_suffix(''))
        # Check if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
        except ImportError:
            logger.warning("Matplotlib not available for visualizations")
            
            # Save JSON report only
            with open(f"{output_base}.json", 'w', encoding='utf-8') as f:
                json.dump(self.performance_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Performance analysis saved to {output_file}")
            return
        
        # Create base output filename without extension
        output_base = str(Path(output_file).with_suffix(''))
        
        # Save JSON report
        with open(f"{output_base}.json", 'w', encoding='utf-8') as f:
            json.dump(self.performance_analysis, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # Create visualizations
        try:
            self._create_performance_visualizations(output_base)
            logger.info(f"Performance analysis and visualizations saved to {output_base}.*")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _create_performance_visualizations(self, output_base: str) -> None:
        """
        Create visualizations for performance analysis.
        
        Args:
            output_base: Base filename for output files
        """
        # Set style
        plt.style.use('ggplot')
        
        # Create visualizations for each system
        for system_name, performance_data in self.performance_analysis["performance_by_category"].items():
            # Performance by query type
            if "query_type" in performance_data:
                plt.figure(figsize=(12, 6))
                
                categories = []
                avg_scores = []
                counts = []
                
                for category, data in performance_data["query_type"].items():
                    categories.append(category)
                    avg_scores.append(data["average_score"])
                    counts.append(data["count"])
                
                # Sort by average score
                sorted_indices = np.argsort(avg_scores)[::-1]
                categories = [categories[i] for i in sorted_indices]
                avg_scores = [avg_scores[i] for i in sorted_indices]
                counts = [counts[i] for i in sorted_indices]
                
                # Create bar chart
                x = np.arange(len(categories))
                width = 0.7
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Plot average scores
                bars = ax1.bar(x, avg_scores, width, color='skyblue')
                
                # Add count labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'n={counts[i]}',
                            ha='center', va='bottom', fontsize=9)
                
                # Add score labels
                for i, v in enumerate(avg_scores):
                    ax1.text(i, v/2, f'{v:.2f}', ha='center', va='center', 
                         color='black', fontweight='bold')
                
                # Set labels and title
                ax1.set_xlabel('Query Type')
                ax1.set_ylabel('Average Score')
                ax1.set_title(f'Performance by Query Type: {system_name}')
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories, rotation=45, ha='right')
                ax1.set_ylim(0, max(avg_scores) * 1.1)
                
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{output_base}_{system_name}_query_type.png")
                plt.close()
            
            # Performance by complexity
            if "complexity" in performance_data:
                plt.figure(figsize=(10, 6))
                
                categories = []
                avg_scores = []
                counts = []
                
                for category, data in performance_data["complexity"].items():
                    categories.append(category)
                    avg_scores.append(data["average_score"])
                    counts.append(data["count"])
                
                # Sort by complexity level if possible
                if all(c in ["L1", "L2", "L3"] for c in categories):
                    sorted_indices = [categories.index(c) for c in ["L1", "L2", "L3"] if c in categories]
                else:
                    # Sort alphabetically
                    sorted_indices = np.argsort(categories)
                
                categories = [categories[i] for i in sorted_indices]
                avg_scores = [avg_scores[i] for i in sorted_indices]
                counts = [counts[i] for i in sorted_indices]
                
                # Create bar chart
                x = np.arange(len(categories))
                width = 0.7
                
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Plot average scores with gradient colors
                colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(categories)))
                bars = ax1.bar(x, avg_scores, width, color=colors)
                
                # Add count labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'n={counts[i]}',
                            ha='center', va='bottom', fontsize=9)
                
                # Add score labels
                for i, v in enumerate(avg_scores):
                    ax1.text(i, v/2, f'{v:.2f}', ha='center', va='center', 
                         color='black', fontweight='bold')
                
                # Set labels and title
                ax1.set_xlabel('Complexity Level')
                ax1.set_ylabel('Average Score')
                ax1.set_title(f'Performance by Complexity: {system_name}')
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories)
                ax1.set_ylim(0, max(avg_scores) * 1.1)
                
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{output_base}_{system_name}_complexity.png")
                plt.close()
            
            # Performance by paper (if available)
            if "paper_id" in performance_data:
                # Only show if we have a reasonable number of papers
                if len(performance_data["paper_id"]) <= 20:
                    plt.figure(figsize=(12, 6))
                    
                    categories = []
                    avg_scores = []
                    counts = []
                    
                    for category, data in performance_data["paper_id"].items():
                        categories.append(category)
                        avg_scores.append(data["average_score"])
                        counts.append(data["count"])
                    
                    # Sort by average score
                    sorted_indices = np.argsort(avg_scores)[::-1]
                    categories = [categories[i] for i in sorted_indices]
                    avg_scores = [avg_scores[i] for i in sorted_indices]
                    counts = [counts[i] for i in sorted_indices]
                    
                    # Create bar chart
                    x = np.arange(len(categories))
                    width = 0.7
                    
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    
                    # Plot average scores
                    bars = ax1.bar(x, avg_scores, width, color='lightgreen')
                    
                    # Add count labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'n={counts[i]}',
                                ha='center', va='bottom', fontsize=9)
                    
                    # Set labels and title
                    ax1.set_xlabel('Paper ID')
                    ax1.set_ylabel('Average Score')
                    ax1.set_title(f'Performance by Paper: {system_name}')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(categories, rotation=45, ha='right')
                    ax1.set_ylim(0, max(avg_scores) * 1.1)
                    
                    plt.tight_layout()
                    
                    # Save figure
                    plt.savefig(f"{output_base}_{system_name}_paper_id.png")
                    plt.close()
        
        # System comparison if we have multiple systems
        if len(self.performance_analysis["systems"]) > 1:
            plt.figure(figsize=(12, 8))
            
            # Compare systems across query types
            systems = self.performance_analysis["systems"]
            query_types = set()
            
            # Collect all query types
            for system_name in systems:
                if system_name in self.performance_analysis["performance_by_category"]:
                    performance_data = self.performance_analysis["performance_by_category"][system_name]
                    if "query_type" in performance_data:
                        query_types.update(performance_data["query_type"].keys())
            
            # Sort query types
            query_types = sorted(query_types)
            
            # Create data for plotting
            all_data = []
            
            for system_name in systems:
                system_scores = []
                
                if system_name in self.performance_analysis["performance_by_category"]:
                    performance_data = self.performance_analysis["performance_by_category"][system_name]
                    
                    if "query_type" in performance_data:
                        for qt in query_types:
                            if qt in performance_data["query_type"]:
                                system_scores.append(performance_data["query_type"][qt]["average_score"])
                            else:
                                system_scores.append(0)
                    else:
                        system_scores = [0] * len(query_types)
                else:
                    system_scores = [0] * len(query_types)
                
                all_data.append(system_scores)
            
            # Create grouped bar chart
            x = np.arange(len(query_types))
            width = 0.8 / len(systems)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot bars for each system
            for i, (system_name, scores) in enumerate(zip(systems, all_data)):
                offset = (i - len(systems) / 2 + 0.5) * width
                bars = ax.bar(x + offset, scores, width, label=system_name)
            
            # Set labels and title
            ax.set_xlabel('Query Type')
            ax.set_ylabel('Average Score')
            ax.set_title('System Performance Comparison by Query Type')
            ax.set_xticks(x)
            ax.set_xticklabels(query_types, rotation=45, ha='right')
            ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{output_base}_system_comparison.png")
            plt.close()
    
    def generate_adversarial_examples(self, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate adversarial examples based on bias analysis.
        
        Args:
            output_file: Path to save generated examples
            
        Returns:
            List of adversarial question dictionaries
        """
        if not self.questions:
            logger.warning("No questions loaded for generating adversarial examples")
            return []
        
        logger.info("Generating adversarial examples...")
        
        # Create adversarial questions
        adversarial_questions = []
        
        # Get performance biases if available
        performance_biases = []
        if self.performance_analysis and "potential_biases" in self.performance_analysis:
            performance_biases = self.performance_analysis["potential_biases"]
        
        # Find the most vulnerable categories
        vulnerable_categories = {}
        
        for bias in performance_biases:
            field = bias.get("field")
            category = bias.get("worst_category")
            
            if field and category:
                if field not in vulnerable_categories:
                    vulnerable_categories[field] = []
                
                if category not in vulnerable_categories[field]:
                    vulnerable_categories[field].append(category)
        
        # Generate adversarial examples targeting vulnerable categories
        logger.info(f"Found vulnerable categories: {vulnerable_categories}")
        
        # Generate paraphrased examples for the most vulnerable query types
        if "query_type" in vulnerable_categories:
            vulnerable_query_types = vulnerable_categories["query_type"]
            
            # Select some questions of these types
            for query_type in vulnerable_query_types:
                candidate_questions = [q for q in self.questions 
                                      if q.get("metadata", {}).get("query_type") == query_type]
                
                # Select up to 5 questions of this type
                selected_questions = candidate_questions[:5] if len(candidate_questions) > 5 else candidate_questions
                
                for question in selected_questions:
                    # Create paraphrased version
                    adv_question = self._create_paraphrased_question(question)
                    if adv_question:
                        adversarial_questions.append(adv_question)
                    
                    # Create question with irrelevant context
                    adv_question = self._create_question_with_irrelevant_context(question)
                    if adv_question:
                        adversarial_questions.append(adv_question)
        
        # Generate examples with ambiguity for complex questions
        if "complexity" in vulnerable_categories:
            vulnerable_complexities = vulnerable_categories["complexity"]
            
            for complexity in vulnerable_complexities:
                candidate_questions = [q for q in self.questions 
                                      if q.get("metadata", {}).get("complexity") == complexity]
                
                # Select up to 5 questions of this complexity
                selected_questions = candidate_questions[:5] if len(candidate_questions) > 5 else candidate_questions
                
                for question in selected_questions:
                    # Create ambiguous version
                    adv_question = self._create_ambiguous_question(question)
                    if adv_question:
                        adversarial_questions.append(adv_question)
        
        # Generate examples with impossible questions
        impossible_count = min(5, len(self.questions))
        for i in range(impossible_count):
            if i < len(self.questions):
                # Create impossible version
                adv_question = self._create_impossible_question(self.questions[i])
                if adv_question:
                    adversarial_questions.append(adv_question)
        
        logger.info(f"Generated {len(adversarial_questions)} adversarial examples")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(adversarial_questions, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Adversarial examples saved to {output_file}")
        
        return adversarial_questions
    
    def _create_paraphrased_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a paraphrased version of a question.
        
        Args:
            question: Original question
            
        Returns:
            Paraphrased question or None if not possible
        """
        # Create a copy of the question
        adv_question = dict(question)
        
        # Get the original question text
        original_text = question["question"]
        
        # Create simple paraphrase
        paraphrase = None
        
        # Technique 1: Change question prefix
        if original_text.startswith("What is"):
            paraphrase = f"Could you explain what {original_text[8:].rstrip('?')} is?"
        elif original_text.startswith("How"):
            paraphrase = f"In what way {original_text[4:].rstrip('?')}?"
        elif original_text.startswith("Why"):
            paraphrase = f"For what reason {original_text[4:].rstrip('?')}?"
        elif original_text.startswith("Where"):
            paraphrase = f"In which location {original_text[6:].rstrip('?')}?"
        elif original_text.startswith("When"):
            paraphrase = f"At what time {original_text[5:].rstrip('?')}?"
        elif original_text.startswith("Can you"):
            paraphrase = f"Would you be able to {original_text[8:].rstrip('?')}?"
        
        # If technique 1 didn't work, try technique 2: add a prefix
        if not paraphrase:
            prefixes = [
                "I'm wondering:",
                "I'd like to know:",
                "Can you tell me:",
                "I'm interested in learning:",
                "Please explain:"
            ]
            
            paraphrase = f"{prefixes[hash(original_text) % len(prefixes)]} {original_text}"
        
        # If we have a paraphrase, update the question
        if paraphrase and paraphrase != original_text:
            # Update question ID and text
            adv_question["question_id"] = f"{question.get('question_id', 'q')}_para"
            adv_question["question"] = paraphrase
            
            # Update metadata
            adv_question["metadata"] = dict(question.get("metadata", {}))
            adv_question["metadata"]["adversarial"] = {
                "technique": "paraphrase",
                "original_id": question.get("question_id", ""),
                "original_text": original_text
            }
            
            return adv_question
        
        return None
    
    def _create_ambiguous_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create an ambiguous version of a question.
        
        Args:
            question: Original question
            
        Returns:
            Ambiguous question or None if not possible
        """
        # Create a copy of the question
        adv_question = dict(question)
        
        # Get the original question text
        original_text = question["question"]
        
        # Create ambiguous version
        ambiguous_text = None
        
        # Technique 1: Replace specific terms with pronouns
        nouns = ["it", "this", "that", "these", "those"]
        
        words = original_text.split()
        if len(words) > 4:
            # Find a good candidate for replacement (skip first and last two words)
            for i in range(2, len(words) - 2):
                # If word is capitalized, it might be an important term
                if words[i][0].isupper() and len(words[i]) > 3:
                    modified_words = words.copy()
                    modified_words[i] = nouns[hash(words[i]) % len(nouns)]
                    ambiguous_text = " ".join(modified_words)
                    break
        
        # Technique 2: Remove specific context
        if not ambiguous_text:
            for pattern in [
                r' in the [A-Za-z]+ of [A-Za-z]+',  # "in the context of X"
                r' for [A-Za-z]+ [A-Za-z]+',        # "for X Y"
                r' from [A-Za-z]+ \d+',             # "from X 2021"
                r' according to [A-Za-z]+'          # "according to X"
            ]:
                match = re.search(pattern, original_text)
                if match:
                    ambiguous_text = original_text.replace(match.group(0), "")
                    break
        
        # Technique 3: Add ambiguity modifiers
        if not ambiguous_text:
            if original_text.startswith("What"):
                ambiguous_text = original_text.replace("What", "What potential")
            elif original_text.startswith("How"):
                ambiguous_text = original_text.replace("How", "How might")
            elif original_text.startswith("Why"):
                ambiguous_text = original_text.replace("Why", "Why might")
        
        # If we have an ambiguous version, update the question
        if ambiguous_text and ambiguous_text != original_text:
            # Update question ID and text
            adv_question["question_id"] = f"{question.get('question_id', 'q')}_ambig"
            adv_question["question"] = ambiguous_text
            
            # Update metadata
            adv_question["metadata"] = dict(question.get("metadata", {}))
            adv_question["metadata"]["adversarial"] = {
                "technique": "ambiguity",
                "original_id": question.get("question_id", ""),
                "original_text": original_text
            }
            
            # Add to special categories
            if "special_categories" not in adv_question["metadata"]:
                adv_question["metadata"]["special_categories"] = []
            
            if "ambiguous" not in adv_question["metadata"]["special_categories"]:
                adv_question["metadata"]["special_categories"].append("ambiguous")
            
            return adv_question
        
        return None
    
    def _create_question_with_irrelevant_context(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a question with irrelevant context.
        
        Args:
            question: Original question
            
        Returns:
            Question with irrelevant context or None if not possible
        """
        # Create a copy of the question
        adv_question = dict(question)
        
        # Get the original question text
        original_text = question["question"]
        
        # Irrelevant context templates
        irrelevant_contexts = [
            "While thinking about a completely different topic, ",
            "Although it may not be directly addressed in the paper, ",
            "Without specific evidence in the text, ",
            "Setting aside the main focus of the paper, ",
            "As a hypothetical exploration not covered in the paper, ",
            "Though tangential to the main research, ",
            "From a different perspective than what's presented, ",
            "If we assume information not present in the document, "
        ]
        
        # Add irrelevant context
        context = irrelevant_contexts[hash(original_text) % len(irrelevant_contexts)]
        
        # Make first letter lowercase if needed
        modified_text = original_text
        if modified_text[0].isupper():
            modified_text = modified_text[0].lower() + modified_text[1:]
        
        irrelevant_text = f"{context}{modified_text}"
        
        # Update question ID and text
        adv_question["question_id"] = f"{question.get('question_id', 'q')}_irrel"
        adv_question["question"] = irrelevant_text
        
        # Update metadata
        adv_question["metadata"] = dict(question.get("metadata", {}))
        adv_question["metadata"]["adversarial"] = {
            "technique": "irrelevant_context",
            "original_id": question.get("question_id", ""),
            "original_text": original_text
        }
        
        return adv_question
    
    def _create_impossible_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create an impossible version of a question.
        
        Args:
            question: Original question
            
        Returns:
            Impossible question or None if not possible
        """
        # Create a copy of the question
        adv_question = dict(question)
        
        # Get the original question text
        original_text = question["question"]
        
        # Impossible question templates
        impossible_prefixes = [
            "What classified information about ",
            "What unreported aspects of ",
            "What will be the future impact of ",
            "What undocumented features of ",
            "What proprietary details of "
        ]
        
        # Generate impossible question
        impossible_text = None
        
        # Technique 1: Ask about information from the future
        if "in the future" not in original_text.lower() and "will" not in original_text.lower():
            if original_text.startswith("What"):
                impossible_text = original_text.replace("What", "What will")
            elif original_text.startswith("How"):
                impossible_text = original_text.replace("How", "How will")
            else:
                # Extract noun phrase to use in template
                words = original_text.split()
                if len(words) > 3:
                    noun_phrase = " ".join(words[1:3])
                    prefix = impossible_prefixes[hash(original_text) % len(impossible_prefixes)]
                    impossible_text = f"{prefix}{noun_phrase}?"
        
        # Technique 2: Ask about information outside the paper's scope
        if not impossible_text:
            if "paper" in original_text or "study" in original_text or "research" in original_text:
                impossible_text = original_text.replace("?", " that was deliberately excluded from the paper?")
            else:
                # Add specific request for non-existent information
                impossible_text = original_text.replace("?", " according to unpublished data?")
        
        # If we have an impossible version, update the question
        if impossible_text and impossible_text != original_text:
            # Update question ID and text
            adv_question["question_id"] = f"{question.get('question_id', 'q')}_imp"
            adv_question["question"] = impossible_text
            
            # Update metadata
            adv_question["metadata"] = dict(question.get("metadata", {}))
            adv_question["metadata"]["adversarial"] = {
                "technique": "impossible",
                "original_id": question.get("question_id", ""),
                "original_text": original_text
            }
            
            # Add to special categories
            if "special_categories" not in adv_question["metadata"]:
                adv_question["metadata"]["special_categories"] = []
            
            adv_question["metadata"]["special_categories"].append("impossible")
            
            # Mark gold answer if it exists
            if "gold_answer" in adv_question:
                adv_question["gold_answer"] = {
                    "text": "This question cannot be answered based on the information provided in the paper.",
                    "is_validated": True,
                    "supporting_passages": []
                }
            
            return adv_question
        
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Bias Analysis Tool")
    parser.add_argument("--questions", help="Path to questions file or directory")
    parser.add_argument("--results", help="Path to evaluation results file or directory")
    parser.add_argument("--output-dir", default="reports/bias_analysis", 
                       help="Directory to save analysis reports")
    parser.add_argument("--dataset-analysis", action="store_true",
                       help="Analyze dataset bias")
    parser.add_argument("--performance-analysis", action="store_true",
                       help="Analyze performance bias")
    parser.add_argument("--generate-adversarial", action="store_true",
                       help="Generate adversarial examples")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create analyzer
    analyzer = RAGBiasAnalyzer(
        reports_dir=args.output_dir
    )
    
    # Load data
    if args.questions:
        analyzer.load_questions(args.questions)
    else:
        analyzer.load_questions()
    
    if args.results:
        analyzer.load_evaluation_results(args.results)
    
    # Run requested analyses
    if args.dataset_analysis or (not args.performance_analysis and not args.generate_adversarial):
        output_file = os.path.join(args.output_dir, "dataset_bias_analysis.json")
        analyzer.analyze_dataset_bias(output_file)
    
    if args.performance_analysis and analyzer.results:
        output_file = os.path.join(args.output_dir, "performance_bias_analysis.json")
        analyzer.analyze_performance_bias(output_file)
    
    if args.generate_adversarial:
        output_file = os.path.join(args.output_dir, "adversarial_examples.json")
        analyzer.generate_adversarial_examples(output_file)