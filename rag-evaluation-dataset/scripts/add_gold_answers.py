#!/usr/bin/env python3
"""
Utility script to add gold answers to questions.
Takes a questions file and adds mock gold answers to make it compatible with the evaluation framework.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def add_gold_answers(input_file, output_file=None):
    """
    Add gold answers to questions.
    
    Args:
        input_file: Path to input questions file
        output_file: Path to output file (if None, will use input_file with _with_gold suffix)
    """
    # Set default output file
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_stem(f"{input_path.stem}_with_gold")
    
    # Load questions
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            questions = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {input_file} is not a valid JSON file")
            return None
    
    # Handle single question case
    if isinstance(questions, dict):
        questions = [questions]
    
    # Add gold answers
    modified_questions = []
    for question in questions:
        # Skip if already has gold answer
        if "gold_answer" in question:
            modified_questions.append(question)
            continue
        
        # Create a copy
        modified_question = dict(question)
        
        # Add mock gold answer
        question_text = question.get("question", "")
        modified_question["gold_answer"] = {
            "text": f"This is a mock gold answer for: {question_text}",
            "is_validated": True,
            "supporting_passages": [
                {
                    "id": "P1",
                    "section": "Introduction",
                    "text": f"This is a mock supporting passage for: {question_text}"
                }
            ]
        }
        
        modified_questions.append(modified_question)
    
    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(modified_questions, f, indent=2)
    
    print(f"Added gold answers to {len(modified_questions)} questions")
    print(f"Saved to {output_file}")
    
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add gold answers to questions")
    parser.add_argument("input_file", help="Path to input questions file")
    parser.add_argument("--output-file", help="Path to output file")
    
    args = parser.parse_args()
    
    add_gold_answers(args.input_file, args.output_file)