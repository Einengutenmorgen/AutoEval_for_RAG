#!/usr/bin/env python3
"""
Mock RAG server for testing the evaluation framework.
Provides a simple REST API endpoint that simulates a RAG system.
"""

import os
import json
import random
import time
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Mock RAG Server", description="Simple mock RAG server for testing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings
QUESTIONS_DIR = Path("question_sets")
DELAY_MIN = 0.5  # Minimum response delay in seconds
DELAY_MAX = 2.0  # Maximum response delay in seconds
FAILURE_RATE = 0.05  # Rate of simulated failures (0-1)
QUESTIONS_CACHE = {}  # Cache to store loaded questions


def load_questions(questions_dir: Path = QUESTIONS_DIR) -> Dict[str, Any]:
    """
    Load all questions from the questions directory.
    
    Args:
        questions_dir: Directory containing question files
        
    Returns:
        Dictionary of questions by ID
    """
    global QUESTIONS_CACHE
    
    if QUESTIONS_CACHE:
        return QUESTIONS_CACHE
    
    all_questions = {}
    
    # Find all question files
    for file_path in questions_dir.glob("**/*_questions*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # Handle both list and single question formats
            if isinstance(questions, dict):
                questions = [questions]
            
            # Add questions to collection
            for question in questions:
                question_id = question.get("question_id", str(len(all_questions)))
                all_questions[question_id] = question
        except Exception as e:
            logger.error(f"Error loading questions from {file_path}: {e}")
    
    logger.info(f"Loaded {len(all_questions)} questions")
    
    # Cache the questions
    QUESTIONS_CACHE = all_questions
    
    return all_questions


def get_relevant_context(question_text: str, gold_answer: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Generate or retrieve relevant context for a question.
    
    Args:
        question_text: Question text
        gold_answer: Gold answer with supporting passages if available
        
    Returns:
        List of context dictionaries
    """
    contexts = []
    
    # Use gold answer contexts if available
    if gold_answer and "supporting_passages" in gold_answer:
        for i, passage in enumerate(gold_answer["supporting_passages"]):
            contexts.append({
                "id": passage.get("id", f"ctx_{i+1}"),
                "text": passage.get("text", ""),
                "score": random.uniform(0.7, 0.95),
                "metadata": {
                    "section": passage.get("section", "")
                }
            })
    else:
        # Generate mock contexts
        num_contexts = random.randint(2, 5)
        
        for i in range(num_contexts):
            # Create random text related to the question
            words = question_text.split()
            if words:
                # Use some words from the question
                sample_size = min(3, len(words))
                sample_words = random.sample(words, sample_size)
                
                # Generate context text
                context_text = f"This is a simulated context passage {i+1} that mentions {' '.join(sample_words)}. "
                context_text += "The context continues with additional information that might be relevant to the query. "
                context_text += "This is mock data for testing purposes only."
                
                contexts.append({
                    "id": f"ctx_{i+1}",
                    "text": context_text,
                    "score": random.uniform(0.5, 0.9),
                    "metadata": {
                        "section": f"Section {i+1}"
                    }
                })
    
    return contexts


def generate_answer(question_text: str, contexts: List[Dict[str, Any]], 
                   gold_answer: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate an answer based on the question and contexts.
    
    Args:
        question_text: Question text
        contexts: Retrieved contexts
        gold_answer: Gold answer if available
        
    Returns:
        Generated answer text
    """
    # Use gold answer with some noise if available
    if gold_answer and gold_answer.get("text"):
        gold_text = gold_answer["text"]
        
        # Sometimes return the perfect answer
        if random.random() < 0.2:
            return gold_text
        
        # Otherwise add some noise to the gold answer
        words = gold_text.split()
        if len(words) > 10:
            # Remove some words
            remove_count = random.randint(1, min(3, len(words) // 5))
            for _ in range(remove_count):
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
            
            # Add some filler words
            filler_words = ["actually", "basically", "in fact", "specifically", "generally"]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, len(words))
                words.insert(idx, random.choice(filler_words))
            
            return " ".join(words)
    
    # No gold answer available, generate a mock answer from contexts
    if contexts:
        answer = "Based on the retrieved information, "
        
        # Extract some text from contexts
        for ctx in contexts[:2]:  # Use up to 2 contexts
            ctx_text = ctx["text"]
            words = ctx_text.split()
            
            if len(words) > 10:
                # Extract a random segment
                start = random.randint(0, len(words) - 10)
                end = min(start + random.randint(5, 10), len(words))
                snippet = " ".join(words[start:end])
                answer += snippet + ". "
        
        return answer.strip()
    
    # Fallback for no contexts
    return f"I don't have enough information to answer the question about '{question_text}'."


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {"status": "ok", "message": "Mock RAG server is running"}


@app.post("/query")
async def query(request: Request):
    """
    Query endpoint - simulates a RAG system.
    
    Request body should contain a 'question' field.
    """
    try:
        # Parse request body
        body = await request.json()
        question = body.get("question")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question field is required")
        
        # Log the incoming question
        logger.info(f"Received question: {question}")
        
        # Add random delay to simulate processing
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        time.sleep(delay)
        
        # Simulate random failures
        if random.random() < FAILURE_RATE:
            raise HTTPException(status_code=500, detail="Simulated server error")
        
        # Load questions to check if we have a gold answer
        all_questions = load_questions()
        
        # Find matching question if available
        matching_question = None
        for q in all_questions.values():
            if q.get("question") == question:
                matching_question = q
                break
        
        # Get gold answer if available
        gold_answer = matching_question.get("gold_answer") if matching_question else None
        
        # Generate contexts
        contexts = get_relevant_context(question, gold_answer)
        
        # Generate answer
        answer = generate_answer(question, contexts, gold_answer)
        
        # Create response
        response = {
            "answer": answer,
            "contexts": contexts,
            "metadata": {
                "model": "mock-rag-model",
                "version": "1.0",
                "processing_time": delay
            }
        }
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock RAG Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--questions-dir", default="question_sets", help="Directory containing question files")
    parser.add_argument("--delay-min", type=float, default=0.5, help="Minimum response delay in seconds")
    parser.add_argument("--delay-max", type=float, default=2.0, help="Maximum response delay in seconds")
    parser.add_argument("--failure-rate", type=float, default=0.05, help="Rate of simulated failures (0-1)")
    
    args = parser.parse_args()
    
    # Update global settings
    QUESTIONS_DIR = Path(args.questions_dir)
    DELAY_MIN = args.delay_min
    DELAY_MAX = args.delay_max
    FAILURE_RATE = args.failure_rate
    
    # Pre-load questions
    load_questions(QUESTIONS_DIR)
    
    # Run the server
    logger.info(f"Starting mock RAG server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)