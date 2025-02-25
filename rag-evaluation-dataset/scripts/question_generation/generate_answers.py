#!/usr/bin/env python3
"""
Gold standard answer generator for RAG evaluation questions.
Generates comprehensive answers with citations to source passages.
"""

import os
import json
import logging
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/answer_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


class GoldAnswerGenerator:
    """Generator for gold standard answers with citations to source passages."""
    
    def __init__(self, 
                papers_dir: str = 'data/processed',
                output_dir: str = 'question_sets/gold_answers'):
        """
        Initialize the gold answer generator.
        
        Args:
            papers_dir: Directory containing processed paper data
            output_dir: Directory to save gold standard answers
        """
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded papers
        self.paper_cache = {}
    
    def generate_answers_for_file(self, question_file: Union[str, Path], 
                                interactive: bool = True) -> None:
        """
        Generate gold standard answers for questions in a file.
        
        Args:
            question_file: Path to question file (JSON)
            interactive: Whether to run in interactive mode
        """
        question_file = Path(question_file)
        if not question_file.exists():
            raise FileNotFoundError(f"Question file not found: {question_file}")
        
        # Load questions
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if not isinstance(questions, list):
            questions = [questions]  # Handle single question case
        
        logger.info(f"Generating answers for {len(questions)} questions from {question_file}")
        console.print(f"\n[bold green]Generating answers for {len(questions)} questions from [/bold green][cyan]{question_file}[/cyan]")
        
        # Process each question
        for i, question in enumerate(questions):
            console.print(f"\n[bold]Question {i+1}/{len(questions)}[/bold]")
            
            # Skip questions that already have a gold answer
            if question.get("gold_answer") and not interactive:
                console.print("[yellow]Question already has a gold answer. Skipping...[/yellow]")
                continue
            
            # Load paper data if needed
            paper_id = question.get("paper_id")
            if not paper_id:
                console.print("[bold red]Question is missing paper_id. Skipping...[/bold red]")
                continue
            
            paper_data = self._load_paper(paper_id)
            if not paper_data:
                console.print(f"[bold red]Could not load paper data for {paper_id}. Skipping...[/bold red]")
                continue
            
            # Generate answer
            if interactive:
                self._interactive_generate_answer(question, paper_data)
            else:
                self._automatic_generate_answer(question, paper_data)
        
        # Save updated questions
        output_file = self.output_dir / question_file.name.replace('.json', '_with_gold.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved questions with gold answers to {output_file}")
        console.print(f"\n[bold green]Saved questions with gold answers to [/bold green][cyan]{output_file}[/cyan]")
    
    def _load_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Load paper data from cache or file.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Paper data dictionary or None if not found
        """
        # Check cache first
        if paper_id in self.paper_cache:
            return self.paper_cache[paper_id]
        
        # Try to load from file
        paper_file = self.papers_dir / f"{paper_id}.json"
        if not paper_file.exists():
            logger.error(f"Paper file not found: {paper_file}")
            return None
        
        try:
            with open(paper_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Cache for future use
            self.paper_cache[paper_id] = paper_data
            return paper_data
        except Exception as e:
            logger.error(f"Error loading paper data: {e}")
            return None
    
    def _interactive_generate_answer(self, question: Dict[str, Any], 
                                  paper_data: Dict[str, Any]) -> None:
        """
        Interactively generate a gold standard answer.
        
        Args:
            question: Question dictionary
            paper_data: Paper data dictionary
        """
        # Display question
        console.print(Panel(question["question"], title="Question", border_style="blue"))
        
        # Display current answer if it exists
        if question.get("answer"):
            console.print(Panel(question["answer"], title="Current Answer", border_style="green"))
        
        # Display gold answer if it exists
        if question.get("gold_answer"):
            console.print(Panel(question["gold_answer"]["text"], title="Current Gold Answer", border_style="yellow"))
            
            if Confirm.ask("Do you want to keep the current gold answer?", default=True):
                return
        
        # Find relevant passages
        search_term = self._extract_search_term(question)
        relevant_passages = self._find_relevant_passages(search_term, paper_data)
        
        # Display relevant passages
        console.print("\n[bold yellow]Relevant Passages:[/bold yellow]")
        for i, passage in enumerate(relevant_passages):
            console.print(f"\n[bold cyan]Passage {i+1}:[/bold cyan]")
            console.print(f"Section: {passage['section']}")
            console.print(passage["text"])
        
        # Ask for gold answer
        console.print("\n[bold yellow]Enter Gold Standard Answer:[/bold yellow]")
        console.print("[italic](Include citations to passages where appropriate, e.g., [P1], [P2], etc.)[/italic]")
        gold_answer = Prompt.ask("Gold Answer", default=question.get("answer", ""))
        
        # Select supporting passages
        supporting_passages = []
        console.print("\n[bold yellow]Select Supporting Passages:[/bold yellow]")
        
        for i, passage in enumerate(relevant_passages):
            if Confirm.ask(f"Include passage {i+1} as a supporting passage?", default=True):
                supporting_passages.append({
                    "id": f"P{i+1}",
                    "section": passage["section"],
                    "text": passage["text"]
                })
        
        # Allow adding custom passages
        if Confirm.ask("Would you like to add a custom passage not shown above?", default=False):
            console.print("\n[bold yellow]Enter Custom Passage:[/bold yellow]")
            section = Prompt.ask("Section name")
            text = Prompt.ask("Passage text")
            
            supporting_passages.append({
                "id": f"P{len(supporting_passages) + 1}",
                "section": section,
                "text": text
            })
        
        # Create gold answer object
        question["gold_answer"] = {
            "text": gold_answer,
            "supporting_passages": supporting_passages,
            "is_validated": True
        }
        
        console.print("\n[bold green]Gold answer added successfully![/bold green]")
    
    def _automatic_generate_answer(self, question: Dict[str, Any], 
                                paper_data: Dict[str, Any]) -> None:
        """
        Automatically generate a gold standard answer.
        
        Args:
            question: Question dictionary
            paper_data: Paper data dictionary
        """
        # Skip if it already has a gold answer
        if question.get("gold_answer"):
            return
        
        # Use the existing answer as a starting point if it exists
        answer_text = question.get("answer", "")
        
        # Find relevant passages
        search_term = self._extract_search_term(question)
        relevant_passages = self._find_relevant_passages(search_term, paper_data)
        
        # Create supporting passages
        supporting_passages = []
        for i, passage in enumerate(relevant_passages[:3]):  # Limit to top 3
            supporting_passages.append({
                "id": f"P{i+1}",
                "section": passage["section"],
                "text": passage["text"]
            })
        
        # Create gold answer object
        question["gold_answer"] = {
            "text": answer_text,
            "supporting_passages": supporting_passages,
            "is_validated": False  # Requires manual validation
        }
        
        logger.info(f"Automatically generated answer for question: {question['question'][:50]}...")
    
    def _extract_search_term(self, question: Dict[str, Any]) -> str:
        """
        Extract a search term from the question for finding passages.
        
        Args:
            question: Question dictionary
            
        Returns:
            Search term string
        """
        # Get the question text
        question_text = question["question"]
        
        # Remove stop words and punctuation
        stop_words = {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'on', 'for', 'to', 'as', 'with', 'by', 'how', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose', 'that', 'this', 'these', 'those', 'and', 'but', 'or', 'if', 'then', 'else', 'so', 'not', 'no', 'nor', 'at', 'from', 'upon', 'after', 'before', 'above', 'below', 'between', 'among', 'through', 'during', 'under', 'within', 'along', 'across', 'behind', 'beyond', 'over', 'under', 'above', 'into'}
        
        words = question_text.lower().split()
        content_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        
        # Check if there are specific entities or concepts in question metadata
        metadata = question.get("metadata", {})
        if "template_slots" in metadata:
            slots = metadata["template_slots"]
            if "concept" in slots:
                return slots["concept"]
            elif "entity" in slots:
                return slots["entity"]
            elif "method" in slots:
                return slots["method"]
        
        # If no specific entity found, use the most frequent content words
        if content_words:
            # Count word frequency in content words
            word_counts = {}
            for word in content_words:
                word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
                if word:
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get the most frequent content word
            if word_counts:
                return max(word_counts.items(), key=lambda x: x[1])[0]
        
        # Fallback: just use the first 3 words
        return ' '.join(question_text.split()[:3])
    
    def _find_relevant_passages(self, search_term: str, 
                              paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find passages relevant to the search term.
        
        Args:
            search_term: Search term string
            paper_data: Paper data dictionary
            
        Returns:
            List of relevant passage dictionaries
        """
        relevant_passages = []
        
        # Look in sections
        for section in paper_data.get("sections", []):
            section_text = section["content"]
            section_heading = section["heading"]
            
            # Skip certain sections like references
            if any(x in section_heading.lower() for x in ["reference", "bibliography", "acknowledgement"]):
                continue
            
            # Find occurrences of the search term
            if search_term.lower() in section_text.lower():
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', section_text)
                
                # Find sentences containing the search term
                matching_sentences = []
                for i, sentence in enumerate(sentences):
                    if search_term.lower() in sentence.lower():
                        # Get context (surrounding sentences)
                        start = max(0, i - 1)
                        end = min(len(sentences), i + 2)
                        context = ' '.join(sentences[start:end])
                        matching_sentences.append(context)
                
                # Add each matching context as a passage
                for context in matching_sentences:
                    relevant_passages.append({
                        "section": section_heading,
                        "text": context
                    })
        
        # Also check for relevant concepts
        for concept in paper_data.get("concepts", []):
            concept_text = concept.get("text", "")
            if (search_term.lower() in concept_text.lower() or 
                concept_text.lower() in search_term.lower()):
                
                # Add examples as passages
                for example in concept.get("examples", []):
                    relevant_passages.append({
                        "section": example.get("section", "Concept"),
                        "text": example.get("context", "")
                    })
        
        # Deduplicate passages
        unique_passages = []
        seen_texts = set()
        
        for passage in relevant_passages:
            text = passage["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                unique_passages.append(passage)
        
        return unique_passages
    
    def generate_answers_for_directory(self, directory: Union[str, Path], 
                                    interactive: bool = True) -> None:
        """
        Generate gold standard answers for all question files in a directory.
        
        Args:
            directory: Directory containing question files
            interactive: Whether to run in interactive mode
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        # Find all question files
        question_files = list(directory.glob("*_questions.json"))
        if not question_files:
            logger.warning(f"No question files found in {directory}")
            return
        
        logger.info(f"Found {len(question_files)} question files in {directory}")
        console.print(f"\n[bold green]Found {len(question_files)} question files in [/bold green][cyan]{directory}[/cyan]")
        
        # Generate answers for each file
        for file in question_files:
            try:
                self.generate_answers_for_file(file, interactive=interactive)
            except Exception as e:
                logger.error(f"Error generating answers for {file}: {e}")
                console.print(f"[bold red]Error generating answers for {file}: {e}[/bold red]")

    def get_answer_coverage_stats(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Calculate statistics on gold answer coverage.
        
        Args:
            directory: Directory containing question files with gold answers
            
        Returns:
            Dictionary with coverage statistics
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        # Find all question files
        question_files = list(directory.glob("*_with_gold.json"))
        if not question_files:
            logger.warning(f"No gold answer files found in {directory}")
            return {}
        
        stats = {
            "total_questions": 0,
            "questions_with_gold": 0,
            "validated_gold": 0,
            "by_type": {},
            "by_complexity": {},
            "by_paper": {}
        }
        
        # Process each file
        for file in question_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                
                if not isinstance(questions, list):
                    questions = [questions]
                
                stats["total_questions"] += len(questions)
                
                for question in questions:
                    # Track by paper
                    paper_id = question.get("paper_id", "unknown")
                    if paper_id not in stats["by_paper"]:
                        stats["by_paper"][paper_id] = {
                            "total": 0,
                            "with_gold": 0,
                            "validated": 0
                        }
                    
                    stats["by_paper"][paper_id]["total"] += 1
                    
                    # Track by type and complexity
                    q_type = question.get("metadata", {}).get("query_type", "unknown")
                    if q_type not in stats["by_type"]:
                        stats["by_type"][q_type] = {
                            "total": 0,
                            "with_gold": 0,
                            "validated": 0
                        }
                    
                    stats["by_type"][q_type]["total"] += 1
                    
                    complexity = question.get("metadata", {}).get("complexity", "unknown")
                    if complexity not in stats["by_complexity"]:
                        stats["by_complexity"][complexity] = {
                            "total": 0,
                            "with_gold": 0,
                            "validated": 0
                        }
                    
                    stats["by_complexity"][complexity]["total"] += 1
                    
                    # Check for gold answer
                    if question.get("gold_answer"):
                        stats["questions_with_gold"] += 1
                        stats["by_paper"][paper_id]["with_gold"] += 1
                        stats["by_type"][q_type]["with_gold"] += 1
                        stats["by_complexity"][complexity]["with_gold"] += 1
                        
                        # Check if validated
                        if question["gold_answer"].get("is_validated", False):
                            stats["validated_gold"] += 1
                            stats["by_paper"][paper_id]["validated"] += 1
                            stats["by_type"][q_type]["validated"] += 1
                            stats["by_complexity"][complexity]["validated"] += 1
            
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
        
        return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate gold standard answers for RAG evaluation questions")
    parser.add_argument("input", help="Question file or directory containing question files")
    parser.add_argument("--papers-dir", default="data/processed", 
                       help="Directory containing processed paper data")
    parser.add_argument("--output-dir", default="question_sets/gold_answers", 
                       help="Output directory for gold answers")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run in non-interactive mode with automatic answer generation")
    parser.add_argument("--stats", action="store_true",
                       help="Calculate and display gold answer coverage statistics")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    generator = GoldAnswerGenerator(
        papers_dir=args.papers_dir,
        output_dir=args.output_dir
    )
    
    input_path = Path(args.input)
    try:
        if args.stats:
            stats = generator.get_answer_coverage_stats(args.output_dir if input_path.is_dir() else Path(args.output_dir).parent)
            console.print("\n[bold]Gold Answer Coverage Statistics:[/bold]")
            console.print(f"Total questions: {stats['total_questions']}")
            console.print(f"Questions with gold answers: {stats['questions_with_gold']} ({stats['questions_with_gold']/stats['total_questions']*100:.1f if stats['total_questions'] > 0 else 0}%)")
            console.print(f"Validated gold answers: {stats['validated_gold']} ({stats['validated_gold']/stats['total_questions']*100:.1f if stats['total_questions'] > 0 else 0}%)")
        else:
            if input_path.is_file():
                generator.generate_answers_for_file(input_path, interactive=not args.non_interactive)
            elif input_path.is_dir():
                generator.generate_answers_for_directory(input_path, interactive=not args.non_interactive)
            else:
                print(f"Error: Input {input_path} is not a valid file or directory")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during answer generation: {e}", exc_info=True)
        console.print(f"[bold red]Error during answer generation: {e}[/bold red]")
        sys.exit(1)