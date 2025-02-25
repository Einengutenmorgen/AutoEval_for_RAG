#!/usr/bin/env python3
"""
Evaluation framework for RAG systems.
Runs RAG applications against test questions and scores the responses.
"""

import os
import sys
import json
import yaml
import logging
import time
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
import re
from collections import defaultdict
import hashlib
import traceback

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGEvaluationFramework:
    """Framework for evaluating RAG systems against question datasets."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluation framework.
        
        Args:
            config_path: Path to evaluation configuration file
        """
        # Set default paths
        self.base_dir = Path.cwd()
        self.questions_dir = self.base_dir / "question_sets" / "gold_answers"
        self.results_dir = self.base_dir / "evaluation" / "results"
        self.cache_dir = self.base_dir / "evaluation" / "cache"
        self.reports_dir = self.base_dir / "reports"
        
        # Create directories if they don't exist
        for dir_path in [self.results_dir, self.cache_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up metrics
        self.metrics = self._initialize_metrics()
        
        # Initialize result storage
        self.results = {}
        
        # Set up caching if enabled
        self.cache_enabled = self.config["advanced"].get("caching", {}).get("enabled", False)
        if self.cache_enabled:
            self.cache_ttl = self.config["advanced"]["caching"].get("ttl", 86400)  # Default: 24 hours
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load evaluation configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "metrics": {
                "retrieval": {
                    "use_precision_at_k": True,
                    "k_values": [1, 3, 5, 10],
                    "use_recall_at_k": True,
                    "use_mrr": True,
                    "use_ndcg": True,
                    "relevance_threshold": 0.7
                },
                "context": {
                    "use_context_precision": True,
                    "use_context_recall": True,
                    "use_context_conciseness": True,
                    "use_context_coverage": True
                },
                "answer": {
                    "use_factual_correctness": True,
                    "use_comprehensiveness": True,
                    "use_conciseness": True,
                    "use_rouge": True,
                    "rouge_types": ["rouge1", "rouge2", "rougeL"],
                    "use_bleu": True,
                    "use_bertscore": True,
                    "bertscore_model": "microsoft/deberta-xlarge-mnli"
                },
                "robustness": {
                    "use_query_variation": True,
                    "number_of_variations": 3,
                    "use_edge_case_handling": True,
                    "use_impossible_detection": True
                }
            },
            "target_systems": [
                {
                    "name": "default_rag",
                    "description": "Default RAG system for testing",
                    "api_endpoint": "http://localhost:8000/query",
                    "request_format": "json",
                    "authentication": {
                        "type": "none"
                    }
                }
            ],
            "results": {
                "output_directory": "evaluation/results",
                "formats": ["json", "csv"],
                "visualization": {
                    "enabled": True
                }
            },
            "advanced": {
                "parallelism": 4,
                "timeout": 30,
                "retry": {
                    "attempts": 3,
                    "backoff_factor": 2
                },
                "caching": {
                    "enabled": True,
                    "directory": "evaluation/cache",
                    "ttl": 86400
                },
                "logging": {
                    "level": "INFO"
                }
            }
        }
        
        # Load custom config if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    # Load based on file extension
                    if config_path.endswith(('.yaml', '.yml')):
                        custom_config = yaml.safe_load(f)
                    else:
                        custom_config = json.load(f)
                
                # Merge with default config (recursive update)
                self._deep_update(default_config, custom_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Set logging level
        logging_level = default_config["advanced"]["logging"].get("level", "INFO")
        logging.getLogger().setLevel(getattr(logging, logging_level))
        
        return default_config
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively update a dictionary.
        
        Args:
            d: Base dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """
        Initialize evaluation metrics based on configuration.
        
        Returns:
            Dictionary of metric functions
        """
        # Import metrics as needed based on configuration
        metrics = {}
        
        # Import NLP metrics if needed
        nlp_metrics_needed = False
        answer_metrics = self.config["metrics"]["answer"]
        
        if answer_metrics.get("use_rouge") or answer_metrics.get("use_bleu") or answer_metrics.get("use_bertscore"):
            nlp_metrics_needed = True
        
        if nlp_metrics_needed:
            try:
                # Import only when needed
                import nltk
                from rouge_score import rouge_scorer
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                
                # Download NLTK resources if needed
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                # Set up ROUGE
                if answer_metrics.get("use_rouge"):
                    rouge_types = answer_metrics.get("rouge_types", ["rouge1"])
                    metrics["rouge_scorer"] = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
                
                # Set up BLEU
                if answer_metrics.get("use_bleu"):
                    metrics["bleu_smoothing"] = SmoothingFunction().method1
                
                # Set up BERTScore if enabled
                if answer_metrics.get("use_bertscore"):
                    try:
                        import bert_score
                        metrics["bert_score"] = bert_score
                    except ImportError:
                        logger.warning("BERTScore requested but not installed. Disabling BERTScore metric.")
                        answer_metrics["use_bertscore"] = False
            
            except ImportError as e:
                logger.warning(f"Could not import NLP metrics: {e}. Some metrics will be disabled.")
                answer_metrics["use_rouge"] = False
                answer_metrics["use_bleu"] = False
                answer_metrics["use_bertscore"] = False
        
        return metrics
    
    def evaluate_system(self, system_name: str, 
                      questions_file: Optional[str] = None,
                      output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a RAG system using the question dataset.
        
        Args:
            system_name: Name of the system to evaluate (must be in config)
            questions_file: Path to questions file (if None, use all files in questions directory)
            output_file: Path to save results (default: auto-generated based on system name)
            
        Returns:
            Evaluation results
        """
        # Find system configuration
        system_config = None
        for sys in self.config["target_systems"]:
            if sys["name"] == system_name:
                system_config = sys
                break
        
        if not system_config:
            raise ValueError(f"System '{system_name}' not found in configuration")
        
        logger.info(f"Evaluating system: {system_name}")
        
        # Load questions
        questions = self._load_questions(questions_file)
        
        if not questions:
            raise ValueError("No questions loaded for evaluation")
        
        logger.info(f"Loaded {len(questions)} questions for evaluation")
        
        # Initialize results for this evaluation
        evaluation_id = f"{system_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results[evaluation_id] = {
            "system": system_config,
            "timestamp": datetime.now().isoformat(),
            "questions": len(questions),
            "responses": [],
            "metrics": {},
            "summary": {}
        }
        
        # Run parallel evaluation if configured
        parallelism = self.config["advanced"].get("parallelism", 1)
        
        if parallelism > 1:
            self._run_parallel_evaluation(system_config, questions, evaluation_id, parallelism)
        else:
            self._run_sequential_evaluation(system_config, questions, evaluation_id)
        
        # Calculate metrics
        self._calculate_metrics(evaluation_id)
        
        # Save results
        if output_file:
            results_file = Path(output_file)
        else:
            results_file = self.results_dir / f"{system_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results[evaluation_id], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation complete. Results saved to {results_file}")
        
        return self.results[evaluation_id]
    
    def _load_questions(self, questions_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load questions from file or directory.
        
        Args:
            questions_file: Path to questions file or None to load from default directory
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        if questions_file:
            # Load from specific file
            file_path = Path(questions_file)
            if not file_path.exists():
                raise FileNotFoundError(f"Questions file not found: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_questions = json.load(f)
                
                # Handle both single questions and lists
                if isinstance(file_questions, dict):
                    file_questions = [file_questions]
                
                # Only use questions with gold answers
                questions.extend([q for q in file_questions if "gold_answer" in q])
                
            except Exception as e:
                logger.error(f"Error loading questions from {file_path}: {e}")
        else:
            # Load from all files in questions directory
            for file_path in self.questions_dir.glob("*_with_gold.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_questions = json.load(f)
                    
                    # Handle both single questions and lists
                    if isinstance(file_questions, dict):
                        file_questions = [file_questions]
                    
                    # Only use questions with gold answers
                    questions.extend([q for q in file_questions if "gold_answer" in q])
                    
                except Exception as e:
                    logger.error(f"Error loading questions from {file_path}: {e}")
        
        return questions
    
    def _run_sequential_evaluation(self, system_config: Dict[str, Any], 
                                 questions: List[Dict[str, Any]], 
                                 evaluation_id: str) -> None:
        """
        Run evaluation sequentially.
        
        Args:
            system_config: System configuration
            questions: List of questions
            evaluation_id: Evaluation identifier
        """
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            
            try:
                response = self._query_system(system_config, question)
                self.results[evaluation_id]["responses"].append(response)
            except Exception as e:
                logger.error(f"Error querying system for question {i+1}: {e}")
                # Add error response
                self.results[evaluation_id]["responses"].append({
                    "question": question,
                    "error": str(e),
                    "success": False,
                    "latency": 0
                })
    
    def _run_parallel_evaluation(self, system_config: Dict[str, Any], 
                               questions: List[Dict[str, Any]], 
                               evaluation_id: str,
                               parallelism: int) -> None:
        """
        Run evaluation in parallel.
        
        Args:
            system_config: System configuration
            questions: List of questions
            evaluation_id: Evaluation identifier
            parallelism: Number of parallel threads
        """
        # For concurrent.futures implementation
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [executor.submit(self._query_system, system_config, question) 
                      for question in questions]
            
            for i, future in enumerate(futures):
                try:
                    response = future.result()
                    self.results[evaluation_id]["responses"].append(response)
                    logger.info(f"Completed question {i+1}/{len(questions)}")
                except Exception as e:
                    logger.error(f"Error querying system for question {i+1}: {e}")
                    # Add error response
                    self.results[evaluation_id]["responses"].append({
                        "question": questions[i],
                        "error": str(e),
                        "success": False,
                        "latency": 0
                    })
    
    def _query_system(self, system_config: Dict[str, Any], 
                    question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            system_config: System configuration
            question: Question dictionary
            
        Returns:
            Response dictionary
        """
        # Check cache first if enabled
        if self.cache_enabled:
            cache_key = self._get_cache_key(system_config["name"], question)
            cached_response = self._get_from_cache(cache_key)
            if cached_response:
                logger.debug(f"Using cached response for question: {question['question'][:50]}...")
                return cached_response
        
        # Prepare the request
        api_endpoint = system_config["api_endpoint"]
        request_format = system_config.get("request_format", "json")
        auth = system_config.get("authentication", {"type": "none"})
        
        # Convert environment variables in the endpoint
        if "$" in api_endpoint:
            for var in re.findall(r'\${([^}]+)}', api_endpoint):
                api_endpoint = api_endpoint.replace(f"${{{var}}}", os.environ.get(var, ""))
        
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        
        # Add authentication
        if auth["type"] == "bearer":
            token = auth["token"]
            # Replace environment variable if present
            if token.startswith("${") and token.endswith("}"):
                env_var = token[2:-1]
                token = os.environ.get(env_var, "")
            headers["Authorization"] = f"Bearer {token}"
        elif auth["type"] == "api_key":
            key = auth["key"]
            # Replace environment variable if present
            if key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1]
                key = os.environ.get(env_var, "")
            header_name = auth.get("header_name", "X-API-Key")
            headers[header_name] = key
        
        # Prepare data
        if request_format == "json":
            data = {
                "question": question["question"],
                "metadata": question.get("metadata", {})
            }
        else:
            # Default to simple text format
            data = {"query": question["question"]}
        
        # Record start time
        start_time = time.time()
        
        # Set up for retries
        max_attempts = self.config["advanced"]["retry"].get("attempts", 1)
        backoff_factor = self.config["advanced"]["retry"].get("backoff_factor", 2)
        timeout = self.config["advanced"].get("timeout", 30)
        
        # Send request with retries
        response_data = None
        success = False
        error_message = ""
        
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    api_endpoint,
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                
                response.raise_for_status()  # Raise error for bad status codes
                response_data = response.json()
                success = True
                break
            
            except requests.RequestException as e:
                error_message = str(e)
                logger.warning(f"Request failed (attempt {attempt+1}/{max_attempts}): {error_message}")
                
                if attempt < max_attempts - 1:
                    # Wait before retrying (exponential backoff)
                    sleep_time = backoff_factor ** attempt
                    time.sleep(sleep_time)
        
        # Record end time
        end_time = time.time()
        latency = end_time - start_time
        
        # Prepare response object
        result = {
            "question": question,
            "success": success,
            "latency": latency,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            result["response"] = response_data
            
            # Extract answer if available
            if "answer" in response_data:
                result["answer"] = response_data["answer"]
            elif "text" in response_data:
                result["answer"] = response_data["text"]
            elif "result" in response_data:
                result["answer"] = response_data["result"]
            else:
                # Fallback: use entire response
                result["answer"] = str(response_data)
            
            # Extract retrieved contexts if available
            if "contexts" in response_data:
                result["contexts"] = response_data["contexts"]
            elif "context" in response_data:
                result["contexts"] = [response_data["context"]]
        else:
            result["error"] = error_message
        
        # Cache successful responses if enabled
        if self.cache_enabled and success:
            cache_key = self._get_cache_key(system_config["name"], question)
            self._save_to_cache(cache_key, result)
        
        return result
    
    def _get_cache_key(self, system_name: str, question: Dict[str, Any]) -> str:
        """
        Generate a cache key for a question.
        
        Args:
            system_name: Name of the system
            question: Question dictionary
            
        Returns:
            Cache key string
        """
        # Create a deterministic hash based on system and question
        key_data = f"{system_name}:{question['question']}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a response from the cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached response or None if not found or expired
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check TTL
            if self.cache_ttl > 0:
                cache_time = datetime.fromisoformat(cached_data.get("timestamp", "2000-01-01T00:00:00"))
                now = datetime.now()
                age = (now - cache_time).total_seconds()
                
                if age > self.cache_ttl:
                    # Cache expired
                    return None
            
            return cached_data
        
        except Exception as e:
            logger.warning(f"Error reading from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        Save a response to the cache.
        
        Args:
            cache_key: Cache key
            data: Response data
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _calculate_metrics(self, evaluation_id: str) -> None:
        """
        Calculate evaluation metrics for the results.
        
        Args:
            evaluation_id: Evaluation identifier
        """
        logger.info("Calculating evaluation metrics")
        
        responses = self.results[evaluation_id]["responses"]
        
        # Skip if no successful responses
        successful_responses = [r for r in responses if r.get("success", False)]
        if not successful_responses:
            logger.warning("No successful responses to calculate metrics")
            return
        
        metrics = {}
        
        # Calculate retrieval metrics if contexts available
        if self.config["metrics"]["retrieval"].get("use_precision_at_k") or \
           self.config["metrics"]["retrieval"].get("use_recall_at_k"):
            retrieval_metrics = self._calculate_retrieval_metrics(successful_responses)
            metrics["retrieval"] = retrieval_metrics
        
        # Calculate answer metrics
        if self.config["metrics"]["answer"].get("use_rouge") or \
           self.config["metrics"]["answer"].get("use_bleu") or \
           self.config["metrics"]["answer"].get("use_bertscore"):
            answer_metrics = self._calculate_answer_metrics(successful_responses)
            metrics["answer"] = answer_metrics
        
        # Calculate robustness metrics if available
        if "robustness" in self.config["metrics"] and \
           self.config["metrics"]["robustness"].get("use_impossible_detection"):
            robustness_metrics = self._calculate_robustness_metrics(successful_responses)
            metrics["robustness"] = robustness_metrics
        
        # Store metrics in results
        self.results[evaluation_id]["metrics"] = metrics
        
        # Calculate summary statistics
        summary = self._calculate_summary(successful_responses, metrics)
        self.results[evaluation_id]["summary"] = summary
        
        logger.info("Metrics calculation complete")
    
    def _calculate_retrieval_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate retrieval metrics like precision and recall.
        
        Args:
            responses: List of successful responses
            
        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {}
        
        # Only consider responses that have contexts
        responses_with_contexts = [r for r in responses if "contexts" in r]
        if not responses_with_contexts:
            logger.warning("No responses with context to calculate retrieval metrics")
            return metrics
        
        # Calculate precision@k and recall@k
        k_values = self.config["metrics"]["retrieval"].get("k_values", [1, 3, 5, 10])
        
        precision_at_k = {}
        recall_at_k = {}
        
        for k in k_values:
            precision_sum = 0
            recall_sum = 0
            count = 0
            
            for response in responses_with_contexts:
                # Get retrieved contexts
                retrieved_contexts = response.get("contexts", [])
                if not retrieved_contexts:
                    continue
                
                # Get gold contexts from the question
                gold_contexts = response["question"].get("gold_answer", {}).get("supporting_passages", [])
                if not gold_contexts:
                    continue
                
                # Limit to top k contexts
                top_k_contexts = retrieved_contexts[:k]
                
                # Calculate precision and recall
                gold_context_ids = set(gc.get("id", "") for gc in gold_contexts)
                relevant_count = 0
                
                for ctx in top_k_contexts:
                    # Check if this context appears in gold contexts
                    ctx_id = ctx.get("id", "")
                    
                    # If no ID, try text matching
                    if not ctx_id:
                        ctx_text = ctx.get("text", "")
                        if any(self._text_similarity(ctx_text, gc.get("text", "")) > 0.8 for gc in gold_contexts):
                            relevant_count += 1
                    elif ctx_id in gold_context_ids:
                        relevant_count += 1
                
                # Calculate precision and recall
                precision = relevant_count / len(top_k_contexts) if top_k_contexts else 0
                recall = relevant_count / len(gold_contexts) if gold_contexts else 0
                
                precision_sum += precision
                recall_sum += recall
                count += 1
            
            # Calculate averages
            precision_at_k[f"p@{k}"] = precision_sum / count if count > 0 else 0
            recall_at_k[f"r@{k}"] = recall_sum / count if count > 0 else 0
        
        # Add metrics to results
        if self.config["metrics"]["retrieval"].get("use_precision_at_k"):
            metrics["precision_at_k"] = precision_at_k
        
        if self.config["metrics"]["retrieval"].get("use_recall_at_k"):
            metrics["recall_at_k"] = recall_at_k
        
        # Calculate MRR if requested
        if self.config["metrics"]["retrieval"].get("use_mrr"):
            mrr_sum = 0
            mrr_count = 0
            
            for response in responses_with_contexts:
                retrieved_contexts = response.get("contexts", [])
                gold_contexts = response["question"].get("gold_answer", {}).get("supporting_passages", [])
                
                if not retrieved_contexts or not gold_contexts:
                    continue
                
                # Look for the first relevant context
                gold_context_ids = set(gc.get("id", "") for gc in gold_contexts)
                
                for i, ctx in enumerate(retrieved_contexts):
                    ctx_id = ctx.get("id", "")
                    
                    # If no ID, try text matching
                    found_match = False
                    if not ctx_id:
                        ctx_text = ctx.get("text", "")
                        if any(self._text_similarity(ctx_text, gc.get("text", "")) > 0.8 for gc in gold_contexts):
                            found_match = True
                    elif ctx_id in gold_context_ids:
                        found_match = True
                    
                    if found_match:
                        mrr_sum += 1 / (i + 1)
                        break
                
                mrr_count += 1
            
            metrics["mrr"] = mrr_sum / mrr_count if mrr_count > 0 else 0
        
        return metrics
    
    def _calculate_answer_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate answer quality metrics like ROUGE, BLEU, and BERTScore.
        
        Args:
            responses: List of successful responses
            
        Returns:
            Dictionary of answer metrics
        """
        metrics = {}
        
        # Only consider responses that have answers
        responses_with_answers = [r for r in responses if "answer" in r]
        if not responses_with_answers:
            logger.warning("No responses with answers to calculate answer metrics")
            return metrics
        
        # Calculate ROUGE scores if enabled
        if self.config["metrics"]["answer"].get("use_rouge") and "rouge_scorer" in self.metrics:
            rouge_scores = defaultdict(list)
            
            for response in responses_with_answers:
                pred_answer = response.get("answer", "")
                gold_answer = response["question"].get("gold_answer", {}).get("text", "")
                
                if not pred_answer or not gold_answer:
                    continue
                
                # Calculate ROUGE scores
                try:
                    scores = self.metrics["rouge_scorer"].score(gold_answer, pred_answer)
                    
                    # Extract scores
                    for rouge_type, score in scores.items():
                        rouge_scores[f"{rouge_type}_precision"].append(score.precision)
                        rouge_scores[f"{rouge_type}_recall"].append(score.recall)
                        rouge_scores[f"{rouge_type}_fmeasure"].append(score.fmeasure)
                except Exception as e:
                    logger.warning(f"Error calculating ROUGE score: {e}")
            
            # Calculate averages
            rouge_metrics = {}
            for metric, values in rouge_scores.items():
                if values:
                    rouge_metrics[metric] = sum(values) / len(values)
            
            metrics["rouge"] = rouge_metrics
        
        # Calculate BLEU scores if enabled
        if self.config["metrics"]["answer"].get("use_bleu") and "bleu_smoothing" in self.metrics:
            bleu_scores = []
            
            for response in responses_with_answers:
                pred_answer = response.get("answer", "")
                gold_answer = response["question"].get("gold_answer", {}).get("text", "")
                
                if not pred_answer or not gold_answer:
                    continue
                
                # Tokenize
                pred_tokens = self._tokenize(pred_answer)
                gold_tokens = self._tokenize(gold_answer)
                
                # Calculate BLEU score
                try:
                    # Use 1-4 grams with smoothing
                    bleu_score = sentence_bleu([gold_tokens], pred_tokens, 
                                             smoothing_function=self.metrics["bleu_smoothing"])
                    bleu_scores.append(bleu_score)
                except Exception as e:
                    logger.warning(f"Error calculating BLEU score: {e}")
            
            if bleu_scores:
                metrics["bleu"] = sum(bleu_scores) / len(bleu_scores)
        
        # Calculate BERTScore if enabled
        if self.config["metrics"]["answer"].get("use_bertscore") and "bert_score" in self.metrics:
            try:
                # Prepare lists of predicted and reference answers
                pred_answers = []
                gold_answers = []
                
                for response in responses_with_answers:
                    pred_answer = response.get("answer", "")
                    gold_answer = response["question"].get("gold_answer", {}).get("text", "")
                    
                    if not pred_answer or not gold_answer:
                        continue
                    
                    pred_answers.append(pred_answer)
                    gold_answers.append(gold_answer)
                
                # Calculate BERTScores in batch
                if pred_answers and gold_answers:
                    model = self.config["metrics"]["answer"].get("bertscore_model", "microsoft/deberta-xlarge-mnli")
                    P, R, F1 = self.metrics["bert_score"].score(pred_answers, gold_answers, model_type=model)
                    
                    # Convert torch tensors to numpy arrays and calculate averages
                    metrics["bertscore"] = {
                        "precision": float(P.mean().item()),
                        "recall": float(R.mean().item()),
                        "f1": float(F1.mean().item())
                    }
            except Exception as e:
                logger.warning(f"Error calculating BERTScore: {e}")
        
        return metrics
    
    def _calculate_robustness_metrics(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate robustness metrics like impossible question detection.
        
        Args:
            responses: List of successful responses
            
        Returns:
            Dictionary of robustness metrics
        """
        metrics = {}
        
        # Calculate impossible question detection metrics
        if self.config["metrics"]["robustness"].get("use_impossible_detection"):
            # Identify impossible questions
            impossible_questions = []
            possible_questions = []
            
            for response in responses:
                question = response["question"]
                
                # Check if question is marked as impossible
                is_impossible = False
                if "special_categories" in question.get("metadata", {}):
                    if "impossible" in question["metadata"]["special_categories"]:
                        is_impossible = True
                
                if is_impossible:
                    impossible_questions.append(response)
                else:
                    possible_questions.append(response)
            
            # Calculate metrics if we have any impossible questions
            if impossible_questions:
                # Analyze how the system handled impossible questions
                impossible_correct = 0
                
                for response in impossible_questions:
                    answer = response.get("answer", "").lower()
                    
                    # Check if the answer indicates impossibility
                    # This is a simple heuristic - you may want to use a more sophisticated approach
                    impossibility_indicators = [
                        "cannot answer", "unable to answer", "not possible to answer",
                        "insufficient information", "not enough information",
                        "cannot determine", "cannot be determined"
                    ]
                    
                    if any(indicator in answer for indicator in impossibility_indicators):
                        impossible_correct += 1
                
                # Calculate accuracy
                impossible_accuracy = impossible_correct / len(impossible_questions) if impossible_questions else 0
                
                metrics["impossible_detection"] = {
                    "accuracy": impossible_accuracy,
                    "impossible_count": len(impossible_questions),
                    "possible_count": len(possible_questions)
                }
        
        return metrics
    
    def _calculate_summary(self, responses: List[Dict[str, Any]], 
                         metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics.
        
        Args:
            responses: List of successful responses
            metrics: Calculated metrics
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_questions": len(self.results[list(self.results.keys())[0]]["responses"]),
            "successful_queries": len(responses),
            "success_rate": len(responses) / len(self.results[list(self.results.keys())[0]]["responses"]) if self.results[list(self.results.keys())[0]]["responses"] else 0,
            "average_latency": sum(r.get("latency", 0) for r in responses) / len(responses) if responses else 0
        }
        
        # Calculate overall effectiveness score if we have both retrieval and answer metrics
        if "retrieval" in metrics and "answer" in metrics:
            retrieval_score = 0
            answer_score = 0
            
            # Calculate retrieval score (average of precision@k)
            if "precision_at_k" in metrics["retrieval"]:
                p_at_k = metrics["retrieval"]["precision_at_k"]
                retrieval_score = sum(p_at_k.values()) / len(p_at_k) if p_at_k else 0
            
            # Calculate answer score (use ROUGE-L F1 or BERTScore F1 if available)
            if "rouge" in metrics["answer"] and "rougeL_fmeasure" in metrics["answer"]["rouge"]:
                answer_score = metrics["answer"]["rouge"]["rougeL_fmeasure"]
            elif "bertscore" in metrics["answer"] and "f1" in metrics["answer"]["bertscore"]:
                answer_score = metrics["answer"]["bertscore"]["f1"]
            
            # Calculate overall score (weighted average)
            overall_score = 0.4 * retrieval_score + 0.6 * answer_score
            summary["rag_effectiveness_score"] = overall_score
        
        # Add best and worst performing questions
        if responses:
            # Sort responses by answer quality (use ROUGE-L if available)
            scored_responses = []
            
            for i, response in enumerate(responses):
                score = 0
                pred_answer = response.get("answer", "")
                gold_answer = response["question"].get("gold_answer", {}).get("text", "")
                
                if pred_answer and gold_answer and "rouge_scorer" in self.metrics:
                    try:
                        rouge_scores = self.metrics["rouge_scorer"].score(gold_answer, pred_answer)
                        if "rougeL" in rouge_scores:
                            score = rouge_scores["rougeL"].fmeasure
                    except:
                        pass
                
                scored_responses.append((i, response, score))
            
            # Sort by score
            scored_responses.sort(key=lambda x: x[2], reverse=True)
            
            # Get best and worst examples
            best_count = min(5, len(scored_responses))
            best_examples = [{"index": idx, "question": r["question"]["question"], "score": score} 
                           for idx, r, score in scored_responses[:best_count]]
            
            worst_examples = [{"index": idx, "question": r["question"]["question"], "score": score} 
                            for idx, r, score in scored_responses[-best_count:]]
            
            summary["best_examples"] = best_examples
            summary["worst_examples"] = worst_examples
        
        return summary
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Remove punctuation and lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity based on token overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        tokens1 = set(self._tokenize(text1))
        tokens2 = set(self._tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def generate_report(self, evaluation_id: str, 
                      formats: Optional[List[str]] = None,
                      output_file: Optional[str] = None) -> str:
        """
        Generate a report for an evaluation.
        
        Args:
            evaluation_id: Evaluation identifier
            formats: List of output formats ("json", "csv", "html")
            output_file: Base path for output files (without extension)
            
        Returns:
            Path to the main report file
        """
        if evaluation_id not in self.results:
            raise ValueError(f"Evaluation '{evaluation_id}' not found")
        
        # Get results
        results = self.results[evaluation_id]
        
        # Default formats if not specified
        if not formats:
            formats = self.config["results"].get("formats", ["json"])
        
        # Default output file
        if not output_file:
            system_name = results["system"]["name"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.reports_dir / f"{system_name}_report_{timestamp}")
        
        # Generate reports in each format
        report_files = {}
        
        for fmt in formats:
            if fmt == "json":
                report_file = f"{output_file}.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                report_files["json"] = report_file
            
            elif fmt == "csv":
                # Generate CSV for responses
                responses_file = f"{output_file}_responses.csv"
                
                # Convert responses to DataFrame
                rows = []
                for response in results["responses"]:
                    row = {
                        "question_id": response.get("question", {}).get("question_id", ""),
                        "question": response.get("question", {}).get("question", ""),
                        "success": response.get("success", False),
                        "latency": response.get("latency", 0),
                        "answer": response.get("answer", ""),
                        "gold_answer": response.get("question", {}).get("gold_answer", {}).get("text", "")
                    }
                    rows.append(row)
                
                # Save to CSV
                pd.DataFrame(rows).to_csv(responses_file, index=False)
                report_files["csv_responses"] = responses_file
                
                # Generate CSV for metrics
                metrics_file = f"{output_file}_metrics.csv"
                
                # Flatten metrics to DataFrame
                metric_rows = []
                
                for category, category_metrics in results["metrics"].items():
                    if isinstance(category_metrics, dict):
                        for metric_name, value in category_metrics.items():
                            if isinstance(value, dict):
                                for sub_name, sub_value in value.items():
                                    metric_rows.append({
                                        "category": category,
                                        "metric": f"{metric_name}.{sub_name}",
                                        "value": sub_value
                                    })
                            else:
                                metric_rows.append({
                                    "category": category,
                                    "metric": metric_name,
                                    "value": value
                                })
                
                # Save to CSV
                pd.DataFrame(metric_rows).to_csv(metrics_file, index=False)
                report_files["csv_metrics"] = metrics_file
            
            elif fmt == "html":
                # Generate HTML report
                html_file = f"{output_file}.html"
                
                # Simple HTML generation - you can make this more sophisticated
                html_content = self._generate_html_report(results)
                
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                report_files["html"] = html_file
        
        logger.info(f"Generated reports: {', '.join(report_files.values())}")
        
        # Return the main report file (prefer HTML > JSON > CSV)
        for fmt in ["html", "json", "csv_responses"]:
            if fmt in report_files:
                return report_files[fmt]
        
        return list(report_files.values())[0] if report_files else ""
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """
        Generate an HTML report.
        
        Args:
            results: Evaluation results
            
        Returns:
            HTML content
        """
        system_name = results["system"]["name"]
        timestamp = results["timestamp"]
        summary = results.get("summary", {})
        metrics = results.get("metrics", {})
        
        # Start HTML content
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report: {system_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metrics {{ margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        .chart {{ width: 100%; height: 400px; margin-bottom: 20px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>RAG Evaluation Report</h1>
        <p>System: <strong>{system_name}</strong></p>
        <p>Generated: {timestamp}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Questions: <strong>{summary.get('total_questions', 0)}</strong></p>
            <p>Successful Queries: <strong>{summary.get('successful_queries', 0)}</strong> ({summary.get('success_rate', 0)*100:.1f}%)</p>
            <p>Average Latency: <strong>{summary.get('average_latency', 0):.2f} seconds</strong></p>
            <p>RAG Effectiveness Score: <strong>{summary.get('rag_effectiveness_score', 0):.2f}</strong></p>
        </div>
        
        <div class="metrics">
            <h2>Metrics</h2>
        """
        
        # Add retrieval metrics
        if "retrieval" in metrics:
            html += """
            <h3>Retrieval Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            # Add precision@k
            if "precision_at_k" in metrics["retrieval"]:
                for k, value in metrics["retrieval"]["precision_at_k"].items():
                    html += f"""
                <tr>
                    <td>Precision@{k.split('@')[1]}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
            
            # Add recall@k
            if "recall_at_k" in metrics["retrieval"]:
                for k, value in metrics["retrieval"]["recall_at_k"].items():
                    html += f"""
                <tr>
                    <td>Recall@{k.split('@')[1]}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
            
            # Add MRR
            if "mrr" in metrics["retrieval"]:
                html += f"""
                <tr>
                    <td>Mean Reciprocal Rank (MRR)</td>
                    <td>{metrics["retrieval"]["mrr"]:.4f}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        # Add answer metrics
        if "answer" in metrics:
            html += """
            <h3>Answer Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            # Add ROUGE metrics
            if "rouge" in metrics["answer"]:
                for metric, value in metrics["answer"]["rouge"].items():
                    html += f"""
                <tr>
                    <td>ROUGE {metric}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
            
            # Add BLEU
            if "bleu" in metrics["answer"]:
                html += f"""
                <tr>
                    <td>BLEU</td>
                    <td>{metrics["answer"]["bleu"]:.4f}</td>
                </tr>
                """
            
            # Add BERTScore
            if "bertscore" in metrics["answer"]:
                for metric, value in metrics["answer"]["bertscore"].items():
                    html += f"""
                <tr>
                    <td>BERTScore {metric}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        # Add robustness metrics
        if "robustness" in metrics:
            html += """
            <h3>Robustness Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            # Add impossible question detection
            if "impossible_detection" in metrics["robustness"]:
                html += f"""
                <tr>
                    <td>Impossible Question Detection Accuracy</td>
                    <td>{metrics["robustness"]["impossible_detection"]["accuracy"]:.4f}</td>
                </tr>
                <tr>
                    <td>Number of Impossible Questions</td>
                    <td>{metrics["robustness"]["impossible_detection"]["impossible_count"]}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
        
        # Add charts if we have the right metrics
        if "retrieval" in metrics and "precision_at_k" in metrics["retrieval"]:
            # Prepare data for precision@k chart
            p_at_k = metrics["retrieval"]["precision_at_k"]
            k_values = [k.split('@')[1] for k in p_at_k.keys()]
            p_values = list(p_at_k.values())
            
            html += """
        <div class="chart-container">
            <h3>Precision@K</h3>
            <canvas id="precisionChart"></canvas>
        </div>
        
        <script>
            var ctx = document.getElementById('precisionChart').getContext('2d');
            var precisionChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: """ + str(k_values) + """,
                    datasets: [{
                        label: 'Precision@K',
                        data: """ + str(p_values) + """,
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }]
                },
                options: {
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
        </script>
        """
        
        # Add best and worst examples
        if "best_examples" in summary:
            html += """
        <div class="examples">
            <h3>Best Performing Questions</h3>
            <table>
                <tr>
                    <th>Index</th>
                    <th>Question</th>
                    <th>Score</th>
                </tr>
            """
            
            for example in summary["best_examples"]:
                html += f"""
                <tr>
                    <td>{example["index"]}</td>
                    <td>{example["question"]}</td>
                    <td>{example["score"]:.4f}</td>
                </tr>
                """
            
            html += """
            </table>
            
            <h3>Worst Performing Questions</h3>
            <table>
                <tr>
                    <th>Index</th>
                    <th>Question</th>
                    <th>Score</th>
                </tr>
            """
            
            for example in summary["worst_examples"]:
                html += f"""
                <tr>
                    <td>{example["index"]}</td>
                    <td>{example["question"]}</td>
                    <td>{example["score"]:.4f}</td>
                </tr>
                """
            
            html += """
            </table>
        </div>
            """
        
        # Close HTML
        html += """
    </div>
</body>
</html>
        """
        
        return html
    
    def compare_systems(self, evaluation_ids: List[str], 
                      output_file: Optional[str] = None) -> str:
        """
        Generate a comparison report for multiple systems.
        
        Args:
            evaluation_ids: List of evaluation identifiers
            output_file: Path to output file
            
        Returns:
            Path to the comparison report
        """
        # Validate evaluation IDs
        for eval_id in evaluation_ids:
            if eval_id not in self.results:
                raise ValueError(f"Evaluation '{eval_id}' not found")
        
        # Create comparison data
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "systems": [],
            "metrics_comparison": {},
            "success_rates": {}
        }
        
        # Add system info and success rates
        for eval_id in evaluation_ids:
            results = self.results[eval_id]
            system_name = results["system"]["name"]
            
            comparison["systems"].append({
                "id": eval_id,
                "name": system_name,
                "description": results["system"].get("description", "")
            })
            
            # Add success rate
            total = len(results["responses"])
            successful = len([r for r in results["responses"] if r.get("success", False)])
            comparison["success_rates"][system_name] = successful / total if total else 0
        
        # Compare metrics
        metric_categories = ["retrieval", "answer", "robustness"]
        
        for category in metric_categories:
            comparison["metrics_comparison"][category] = {}
            
            # Collect metrics from all systems
            for eval_id in evaluation_ids:
                results = self.results[eval_id]
                system_name = results["system"]["name"]
                
                if category in results.get("metrics", {}):
                    for metric_name, value in self._flatten_metrics(results["metrics"][category]):
                        if metric_name not in comparison["metrics_comparison"][category]:
                            comparison["metrics_comparison"][category][metric_name] = {}
                        
                        comparison["metrics_comparison"][category][metric_name][system_name] = value
        
        # Add overall scores
        comparison["overall_scores"] = {}
        
        for eval_id in evaluation_ids:
            results = self.results[eval_id]
            system_name = results["system"]["name"]
            
            if "rag_effectiveness_score" in results.get("summary", {}):
                comparison["overall_scores"][system_name] = results["summary"]["rag_effectiveness_score"]
        
        # Save comparison report
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.reports_dir / f"systems_comparison_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison report generated: {output_file}")
        
        return output_file
    
    def _flatten_metrics(self, metrics: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Flatten nested metrics into a list of (name, value) tuples.
        
        Args:
            metrics: Metrics dictionary
            
        Returns:
            List of (metric_name, value) tuples
        """
        flattened = []
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened.append((f"{key}.{subkey}", subvalue))
            else:
                flattened.append((key, value))
        
        return flattened
    
    def analyze_bias(self, evaluation_id: str, 
                   group_by: List[str] = ["query_type", "complexity"],
                   output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze potential biases in evaluation results.
        
        Args:
            evaluation_id: Evaluation identifier
            group_by: List of metadata fields to group by
            output_file: Path to save the analysis
            
        Returns:
            Bias analysis dictionary
        """
        if evaluation_id not in self.results:
            raise ValueError(f"Evaluation '{evaluation_id}' not found")
        
        results = self.results[evaluation_id]
        responses = results["responses"]
        
        # Create analysis structure
        analysis = {
            "system": results["system"]["name"],
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": results.get("metrics", {}),
            "groups": {}
        }
        
        # Group responses by metadata fields
        for field in group_by:
            analysis["groups"][field] = {}
            
            # Count questions in each group
            group_counts = defaultdict(int)
            for response in responses:
                question = response.get("question", {})
                metadata = question.get("metadata", {})
                
                # Get the group value
                if field in metadata:
                    group_value = str(metadata[field])
                else:
                    group_value = "unknown"
                
                group_counts[group_value] += 1
            
            # Calculate success rates and scores for each group
            for group_value, count in group_counts.items():
                group_responses = [r for r in responses if r.get("question", {}).get("metadata", {}).get(field) == group_value]
                successful = [r for r in group_responses if r.get("success", False)]
                
                success_rate = len(successful) / len(group_responses) if group_responses else 0
                
                # Calculate average scores (use ROUGE-L if available)
                rouge_scores = []
                
                for response in successful:
                    pred_answer = response.get("answer", "")
                    gold_answer = response.get("question", {}).get("gold_answer", {}).get("text", "")
                    
                    if pred_answer and gold_answer and "rouge_scorer" in self.metrics:
                        try:
                            rouge_scores.append(
                                self.metrics["rouge_scorer"].score(gold_answer, pred_answer)["rougeL"].fmeasure
                            )
                        except:
                            pass
                
                avg_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
                
                analysis["groups"][field][group_value] = {
                    "count": count,
                    "percentage": count / len(responses) if responses else 0,
                    "success_rate": success_rate,
                    "average_score": avg_score
                }
        
        # Check for significant performance disparities
        analysis["potential_biases"] = []
        
        for field, groups in analysis["groups"].items():
            if len(groups) <= 1:
                continue
            
            # Get average and std dev of scores
            scores = [g["average_score"] for g in groups.values()]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Calculate std dev
            if len(scores) > 1:
                squared_diffs = [(score - avg_score) ** 2 for score in scores]
                variance = sum(squared_diffs) / (len(scores) - 1)
                std_dev = variance ** 0.5
            else:
                std_dev = 0
            
            # Check for outliers (more than 2 std devs from mean)
            for group_value, stats in groups.items():
                if abs(stats["average_score"] - avg_score) > 2 * std_dev and std_dev > 0:
                    analysis["potential_biases"].append({
                        "field": field,
                        "group": group_value,
                        "score": stats["average_score"],
                        "difference_from_mean": stats["average_score"] - avg_score,
                        "standard_deviations": (stats["average_score"] - avg_score) / std_dev if std_dev > 0 else 0
                    })
        
        # Save analysis if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Bias analysis saved to {output_file}")
        
        return analysis
    
    def generate_adversarial_examples(self, questions: List[Dict[str, Any]], 
                                   techniques: List[str] = ["paraphrase", "ambiguity", "irrelevant_context"],
                                   output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate adversarial examples to test robustness.
        
        Args:
            questions: List of question dictionaries
            techniques: List of adversarial techniques to apply
            output_file: Path to save generated examples
            
        Returns:
            List of adversarial question dictionaries
        """
        adversarial_questions = []
        
        # Process each question
        for question in questions:
            orig_question_text = question["question"]
            
            # Apply each technique
            for technique in techniques:
                if technique == "paraphrase":
                    # Simple paraphrasing
                    new_questions = self._generate_paraphrases(orig_question_text)
                    
                    for i, new_question_text in enumerate(new_questions):
                        # Create new question object based on original
                        adv_question = dict(question)
                        adv_question["question_id"] = f"{question.get('question_id', 'q')}_para_{i+1}"
                        adv_question["question"] = new_question_text
                        
                        # Update metadata
                        adv_question["metadata"] = dict(question.get("metadata", {}))
                        adv_question["metadata"]["adversarial"] = {
                            "technique": "paraphrase",
                            "original_id": question.get("question_id", ""),
                            "variant": i+1
                        }
                        
                        adversarial_questions.append(adv_question)
                
                elif technique == "ambiguity":
                    # Introduce ambiguity
                    new_question_text = self._introduce_ambiguity(orig_question_text)
                    
                    if new_question_text and new_question_text != orig_question_text:
                        # Create new question object
                        adv_question = dict(question)
                        adv_question["question_id"] = f"{question.get('question_id', 'q')}_ambig"
                        adv_question["question"] = new_question_text
                        
                        # Update metadata
                        adv_question["metadata"] = dict(question.get("metadata", {}))
                        adv_question["metadata"]["adversarial"] = {
                            "technique": "ambiguity",
                            "original_id": question.get("question_id", "")
                        }
                        
                        # Add to special categories
                        if "special_categories" not in adv_question["metadata"]:
                            adv_question["metadata"]["special_categories"] = []
                        
                        if "ambiguous" not in adv_question["metadata"]["special_categories"]:
                            adv_question["metadata"]["special_categories"].append("ambiguous")
                        
                        adversarial_questions.append(adv_question)
                
                elif technique == "irrelevant_context":
                    # Add irrelevant context
                    new_question_text = self._add_irrelevant_context(orig_question_text)
                    
                    if new_question_text and new_question_text != orig_question_text:
                        # Create new question object
                        adv_question = dict(question)
                        adv_question["question_id"] = f"{question.get('question_id', 'q')}_irrel"
                        adv_question["question"] = new_question_text
                        
                        # Update metadata
                        adv_question["metadata"] = dict(question.get("metadata", {}))
                        adv_question["metadata"]["adversarial"] = {
                            "technique": "irrelevant_context",
                            "original_id": question.get("question_id", "")
                        }
                        
                        adversarial_questions.append(adv_question)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(adversarial_questions, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated {len(adversarial_questions)} adversarial examples saved to {output_file}")
        
        return adversarial_questions
    
    def _generate_paraphrases(self, question: str) -> List[str]:
        """
        Generate paraphrased versions of a question.
        
        Args:
            question: Original question
            
        Returns:
            List of paraphrased questions
        """
        # Simple rule-based paraphrasing
        paraphrases = []
        
        # Technique 1: Reordering phrases
        if question.startswith("What is"):
            paraphrases.append(f"Can you explain what {question[8:].rstrip('?')} is?")
        
        # Technique 2: Synonym substitution
        substitutions = [
            ("What", "Which"),
            ("How", "In what way"),
            ("Why", "For what reason"),
            ("When", "At what time"),
            ("Where", "In which location"),
            ("important", "significant"),
            ("difference", "distinction"),
            ("similar", "comparable"),
            ("explain", "describe"),
            ("advantage", "benefit"),
            ("disadvantage", "drawback")
        ]
        
        for orig, repl in substitutions:
            if orig in question:
                paraphrases.append(question.replace(orig, repl))
        
        # Technique 3: Active/passive conversion
        if "does" in question and "what" in question.lower():
            # E.g., "What does X do?" -> "What is done by X?"
            match = re.search(r"What does ([^?]+) do\?", question)
            if match:
                subject = match.group(1)
                paraphrases.append(f"What is done by {subject}?")
        
        # If no paraphrases were generated, add a simple one
        if not paraphrases:
            paraphrases.append(f"Please tell me: {question}")
        
        # De-duplicate and remove original
        return list(set(p for p in paraphrases if p != question))
    
    def _introduce_ambiguity(self, question: str) -> str:
        """
        Introduce ambiguity into a question.
        
        Args:
            question: Original question
            
        Returns:
            Ambiguous question
        """
        # Technique 1: Remove specific references
        for entity_pattern in [r'(in|of|for) the ([A-Z][a-z]+ ?)+', r'(in|of|for) ([0-9]{4})', r'(in|of|for) this [a-z]+']:
            match = re.search(entity_pattern, question)
            if match:
                return question.replace(match.group(0), "")
        
        # Technique 2: Add ambiguous modifiers
        if question.startswith("What"):
            return question.replace("What", "What potential")
        elif question.startswith("How"):
            return question.replace("How", "How might")
        elif question.startswith("Why"):
            return question.replace("Why", "Why might")
        
        # Technique 3: Replace specific terms with pronouns
        nouns = ["it", "this", "that", "these", "those"]
        
        tokens = question.split()
        if len(tokens) > 4:
            # Replace a noun in the middle of the sentence
            idx = len(tokens) // 2
            tokens[idx] = random.choice(nouns)
            return " ".join(tokens)
        
        # If no techniques worked, return original
        return question
    
    def _add_irrelevant_context(self, question: str) -> str:
        """
        Add irrelevant context to a question.
        
        Args:
            question: Original question
            
        Returns:
            Question with irrelevant context
        """
        irrelevant_contexts = [
            "While thinking about something else, ",
            "I was reading another paper and wondered, ",
            "Although it might not be directly addressed, ",
            "In addition to other topics in the paper, ",
            "Besides the main contributions, ",
            "Considering the broader implications, ",
            "Without specific evidence, ",
            "In a tangentially related domain, "
        ]
        
        context = random.choice(irrelevant_contexts)
        
        # Make first letter lowercase if needed
        if question[0].isupper():
            question = question[0].lower() + question[1:]
        
        return f"{context}{question}"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument("--config", help="Path to evaluation configuration file")
    parser.add_argument("--system", help="Name of the system to evaluate")
    parser.add_argument("--questions", help="Path to questions file (optional)")
    parser.add_argument("--output", help="Path to output file (optional)")
    parser.add_argument("--report", action="store_true", help="Generate report after evaluation")
    parser.add_argument("--report-format", nargs="+", choices=["json", "csv", "html"], 
                       default=["json"], help="Report format(s)")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize framework
        framework = RAGEvaluationFramework(config_path=args.config)
        
        if args.system:
            # Run evaluation
            evaluation_id = f"{args.system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = framework.evaluate_system(
                args.system,
                questions_file=args.questions,
                output_file=args.output
            )
            
            # Generate report if requested
            if args.report:
                report_file = framework.generate_report(
                    evaluation_id,
                    formats=args.report_format
                )
                print(f"Report generated: {report_file}")
        else:
            print("No system specified for evaluation. Use --system to specify a system.")
    
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        traceback.print_exc()
        sys.exit(1)