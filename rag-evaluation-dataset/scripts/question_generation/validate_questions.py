#!/usr/bin/env python3
"""
Question validation and review tool for RAG evaluation dataset.
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
        logging.FileHandler("logs/question_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


class QuestionValidator:
    """Tool for validating and reviewing question-answer pairs."""
    
    QUALITY_CRITERIA = {
        "clarity": "Is the question clear and unambiguous?",
        "relevance": "Is the question relevant to the paper content?",
        "difficulty": "Is the difficulty level appropriate?",
        "annotation": "Are the metadata annotations accurate?",
        "answer": "Is the answer accurate and comprehensive?"
    }
    
    def __init__(self, 
                taxonomy_file: Optional[str] = None,
                output_dir: str = 'question_sets/validated_questions'):
        """
        Initialize the question validator.
        
        Args:
            taxonomy_file: Path to question taxonomy file
            output_dir: Directory to save validated questions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load taxonomy if provided
        self.taxonomy = self._load_taxonomy(taxonomy_file)
        
        # Track validation statistics
        self.stats = {
            "total_reviewed": 0,
            "approved": 0,
            "edited": 0,
            "rejected": 0,
            "by_type": {},
            "by_complexity": {},
            "by_criteria": {}
        }
        
        # Initialize stats for each question type
        for q_type in self.taxonomy["query_types"].keys():
            self.stats["by_type"][q_type] = {
                "total": 0,
                "approved": 0,
                "edited": 0,
                "rejected": 0
            }
        
        # Initialize stats for each complexity level
        for level in self.taxonomy["complexity_levels"].keys():
            self.stats["by_complexity"][level] = {
                "total": 0,
                "approved": 0,
                "edited": 0,
                "rejected": 0
            }
        
        # Initialize stats for each quality criterion
        for criterion in self.QUALITY_CRITERIA.keys():
            self.stats["by_criteria"][criterion] = {
                "passes": 0,
                "fails": 0
            }
    
    def _load_taxonomy(self, taxonomy_file: Optional[str]) -> Dict[str, Any]:
        """
        Load question taxonomy from file.
        
        Args:
            taxonomy_file: Path to taxonomy file
            
        Returns:
            Dictionary with taxonomy information
        """
        default_taxonomy = {
            "query_types": {
                "factoid": "Questions seeking specific facts",
                "definitional": "Questions seeking explanations or definitions",
                "procedural": "Questions about processes or methods",
                "comparative": "Questions requiring comparison",
                "causal": "Questions about causes or effects",
                "quantitative": "Questions requiring numerical analysis",
                "open_ended": "Questions requiring comprehensive analysis"
            },
            "complexity_levels": {
                "L1": "Basic - Single-hop retrieval",
                "L2": "Intermediate - Multi-hop retrieval (2-3 docs)",
                "L3": "Advanced - Complex multi-hop retrieval (4+ docs)"
            },
            "special_categories": {
                "ambiguous": "Questions with multiple valid interpretations",
                "temporal": "Questions whose answers depend on time periods",
                "impossible": "Questions that cannot be answered from the corpus",
                "multi_part": "Questions with multiple components"
            }
        }
        
        if taxonomy_file and os.path.exists(taxonomy_file):
            try:
                # Attempt to load YAML or JSON format
                if taxonomy_file.endswith('.yaml') or taxonomy_file.endswith('.yml'):
                    with open(taxonomy_file, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    with open(taxonomy_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading taxonomy file: {e}")
                return default_taxonomy
        
        return default_taxonomy
    
    def validate_question_file(self, question_file: Union[str, Path], 
                             interactive: bool = True) -> Dict[str, Any]:
        """
        Validate questions from a file.
        
        Args:
            question_file: Path to question file (JSON)
            interactive: Whether to run in interactive mode
            
        Returns:
            Validation statistics
        """
        question_file = Path(question_file)
        if not question_file.exists():
            raise FileNotFoundError(f"Question file not found: {question_file}")
        
        # Load questions
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        if not isinstance(questions, list):
            questions = [questions]  # Handle single question case
        
        logger.info(f"Validating {len(questions)} questions from {question_file}")
        console.print(f"\n[bold green]Validating {len(questions)} questions from [/bold green][cyan]{question_file}[/cyan]")
        
        # Track validated questions
        validated_questions = []
        
        # Process each question
        for i, question in enumerate(questions):
            console.print(f"\n[bold]Question {i+1}/{len(questions)}[/bold]")
            
            if interactive:
                validated_q = self._interactive_validate_question(question)
                if validated_q:  # None if rejected
                    validated_questions.append(validated_q)
            else:
                validated_q = self._automatic_validate_question(question)
                if validated_q:  # None if rejected
                    validated_questions.append(validated_q)
            
            # Update overall stats
            self.stats["total_reviewed"] += 1
        
        # Save validated questions
        if validated_questions:
            output_file = self.output_dir / question_file.name.replace('.json', '_validated.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated_questions, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(validated_questions)} validated questions to {output_file}")
            console.print(f"\n[bold green]Saved {len(validated_questions)} validated questions to [/bold green][cyan]{output_file}[/cyan]")
        
        # Print stats
        self._print_validation_stats()
        
        return self.stats
    
    def _interactive_validate_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Interactively validate a question.
        
        Args:
            question: Question dictionary
            
        Returns:
            Validated question or None if rejected
        """
        # Display question and metadata
        self._display_question(question)
        
        # Evaluate against quality criteria
        criteria_results = {}
        console.print("\n[bold yellow]Quality Criteria Evaluation:[/bold yellow]")
        
        for criterion, description in self.QUALITY_CRITERIA.items():
            console.print(f"[cyan]{criterion}[/cyan]: {description}")
            result = Confirm.ask("Does the question pass this criterion?", default=True)
            criteria_results[criterion] = result
            
            # Update criteria stats
            if result:
                self.stats["by_criteria"][criterion]["passes"] += 1
            else:
                self.stats["by_criteria"][criterion]["fails"] += 1
        
        # Determine if any edits are needed
        needs_edit = not all(criteria_results.values())
        
        if needs_edit:
            console.print("\n[bold yellow]This question needs improvement. Would you like to:[/bold yellow]")
            console.print("[1] Edit the question")
            console.print("[2] Edit the answer")
            console.print("[3] Edit the metadata")
            console.print("[4] Reject the question")
            console.print("[5] Approve anyway")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5"], default="1")
            
            if choice == "1":
                question["question"] = Prompt.ask("Enter revised question", default=question["question"])
                self.stats["edited"] += 1
                self._update_type_stats(question, "edited")
            elif choice == "2":
                console.print(f"\n[bold]Current answer:[/bold] {question['answer']}")
                question["answer"] = Prompt.ask("Enter revised answer", default=question["answer"])
                self.stats["edited"] += 1
                self._update_type_stats(question, "edited")
            elif choice == "3":
                self._edit_metadata(question)
                self.stats["edited"] += 1
                self._update_type_stats(question, "edited")
            elif choice == "4":
                console.print("[bold red]Question rejected[/bold red]")
                self.stats["rejected"] += 1
                self._update_type_stats(question, "rejected")
                return None
            elif choice == "5":
                console.print("[bold yellow]Question approved despite issues[/bold yellow]")
                self.stats["approved"] += 1
                self._update_type_stats(question, "approved")
        else:
            console.print("[bold green]Question approved[/bold green]")
            self.stats["approved"] += 1
            self._update_type_stats(question, "approved")
        
        # Add validation metadata
        question["metadata"]["validation"] = {
            "status": "approved" if not needs_edit or choice in ["1", "2", "3", "5"] else "rejected",
            "criteria_results": criteria_results,
            "edited": needs_edit and choice in ["1", "2", "3"]
        }
        
        return question
    
    def _automatic_validate_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Automatically validate a question using rule-based checks.
        
        Args:
            question: Question dictionary
            
        Returns:
            Validated question or None if rejected
        """
        # Initialize validation results
        criteria_results = {criterion: True for criterion in self.QUALITY_CRITERIA}
        issues = []
        
        # Check clarity
        if len(question["question"]) < 10:
            criteria_results["clarity"] = False
            issues.append("Question is too short")
        
        if "?" not in question["question"]:
            criteria_results["clarity"] = False
            issues.append("Question does not contain a question mark")
        
        if re.search(r'\[\w+\]', question["question"]):
            criteria_results["clarity"] = False
            issues.append("Question contains unfilled template slots")
        
        # Check answer
        if not question.get("answer"):
            criteria_results["answer"] = False
            issues.append("Missing answer")
        elif len(question["answer"]) < 20:
            criteria_results["answer"] = False
            issues.append("Answer is too short")
        
        # Check metadata
        if not question.get("metadata", {}).get("query_type"):
            criteria_results["annotation"] = False
            issues.append("Missing query type")
        
        if not question.get("metadata", {}).get("complexity"):
            criteria_results["annotation"] = False
            issues.append("Missing complexity level")
        
        # Update criteria stats
        for criterion, result in criteria_results.items():
            if result:
                self.stats["by_criteria"][criterion]["passes"] += 1
            else:
                self.stats["by_criteria"][criterion]["fails"] += 1
        
        # Determine validation status
        if all(criteria_results.values()):
            status = "approved"
            self.stats["approved"] += 1
            self._update_type_stats(question, "approved")
        elif sum(criteria_results.values()) >= 3:  # At least 3/5 criteria passed
            status = "needs_review"
            self.stats["edited"] += 1
            self._update_type_stats(question, "edited")
        else:
            status = "rejected"
            self.stats["rejected"] += 1
            self._update_type_stats(question, "rejected")
            return None
        
        # Add validation metadata
        question["metadata"]["validation"] = {
            "status": status,
            "criteria_results": criteria_results,
            "issues": issues,
            "edited": False
        }
        
        return question
    
    def _display_question(self, question: Dict[str, Any]) -> None:
        """
        Display a question, its answer, and metadata.
        
        Args:
            question: Question dictionary
        """
        # Create table for metadata
        metadata_table = Table(title="Question Metadata")
        metadata_table.add_column("Attribute", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        metadata = question.get("metadata", {})
        
        # Add basic metadata
        metadata_table.add_row("ID", question.get("question_id", "N/A"))
        metadata_table.add_row("Paper ID", question.get("paper_id", "N/A"))
        metadata_table.add_row("Query Type", metadata.get("query_type", "N/A"))
        metadata_table.add_row("Complexity", metadata.get("complexity", "N/A"))
        
        # Add special categories if any
        if metadata.get("special_categories"):
            metadata_table.add_row("Special Categories", ", ".join(metadata.get("special_categories", [])))
        
        # Display question
        console.print(Panel(question["question"], title="Question", border_style="blue"))
        
        # Display answer
        console.print(Panel(question.get("answer", "No answer provided"), title="Answer", border_style="green"))
        
        # Display contexts if available
        if question.get("contexts"):
            contexts_text = ""
            for ctx in question["contexts"]:
                contexts_text += f"[bold]{ctx.get('section', 'Context')}:[/bold]\n{ctx.get('text', '')}\n\n"
            console.print(Panel(contexts_text, title="Context Passages", border_style="yellow"))
        
        # Display metadata
        console.print(metadata_table)
    
    def _edit_metadata(self, question: Dict[str, Any]) -> None:
        """
        Edit question metadata interactively.
        
        Args:
            question: Question dictionary
        """
        metadata = question.get("metadata", {})
        
        # Query type
        console.print("\n[bold]Current query type:[/bold] " + metadata.get("query_type", "N/A"))
        query_types = list(self.taxonomy["query_types"].keys())
        q_type_idx = Prompt.ask(
            "Select query type",
            choices=[str(i) for i in range(len(query_types))],
            default=str(query_types.index(metadata.get("query_type", query_types[0])))
        )
        metadata["query_type"] = query_types[int(q_type_idx)]
        
        # Complexity
        console.print("\n[bold]Current complexity:[/bold] " + metadata.get("complexity", "N/A"))
        complexity_levels = list(self.taxonomy["complexity_levels"].keys())
        complexity_idx = Prompt.ask(
            "Select complexity level",
            choices=[str(i) for i in range(len(complexity_levels))],
            default=str(complexity_levels.index(metadata.get("complexity", complexity_levels[0])))
        )
        metadata["complexity"] = complexity_levels[int(complexity_idx)]
        
        # Special categories
        console.print("\n[bold]Current special categories:[/bold] " + 
                     (", ".join(metadata.get("special_categories", [])) if metadata.get("special_categories") else "None"))
        
        special_categories = list(self.taxonomy["special_categories"].keys())
        selected_categories = []
        
        for i, category in enumerate(special_categories):
            is_selected = category in metadata.get("special_categories", [])
            if Confirm.ask(f"Include '{category}'?", default=is_selected):
                selected_categories.append(category)
        
        metadata["special_categories"] = selected_categories
        
        # Update question metadata
        question["metadata"] = metadata
    
    def _update_type_stats(self, question: Dict[str, Any], status: str) -> None:
        """
        Update type-specific statistics.
        
        Args:
            question: Question dictionary
            status: Status (approved, edited, rejected)
        """
        metadata = question.get("metadata", {})
        q_type = metadata.get("query_type")
        complexity = metadata.get("complexity")
        
        if q_type in self.stats["by_type"]:
            self.stats["by_type"][q_type]["total"] += 1
            self.stats["by_type"][q_type][status] += 1
        
        if complexity in self.stats["by_complexity"]:
            self.stats["by_complexity"][complexity]["total"] += 1
            self.stats["by_complexity"][complexity][status] += 1
    
    def _print_validation_stats(self) -> None:
        """Print validation statistics."""
        console.print("\n[bold]Validation Statistics:[/bold]")
        
        # Create main stats table
        stats_table = Table(title="Overall Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", style="green")
        stats_table.add_column("Percentage", style="yellow")
        
        total = self.stats["total_reviewed"]
        if total > 0:
            stats_table.add_row("Total Reviewed", str(total), "100%")
            stats_table.add_row("Approved", str(self.stats["approved"]), f"{self.stats['approved']/total*100:.1f}%")
            stats_table.add_row("Edited", str(self.stats["edited"]), f"{self.stats['edited']/total*100:.1f}%")
            stats_table.add_row("Rejected", str(self.stats["rejected"]), f"{self.stats['rejected']/total*100:.1f}%")
        else:
            stats_table.add_row("Total Reviewed", "0", "0%")
            stats_table.add_row("Approved", "0", "0%")
            stats_table.add_row("Edited", "0", "0%")
            stats_table.add_row("Rejected", "0", "0%")
        
        console.print(stats_table)
        
        # Create type stats table
        type_table = Table(title="Statistics by Question Type")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Total", style="white")
        type_table.add_column("Approved", style="green")
        type_table.add_column("Edited", style="yellow")
        type_table.add_column("Rejected", style="red")
        
        for q_type, stats in self.stats["by_type"].items():
            if stats["total"] > 0:
                type_table.add_row(
                    q_type,
                    str(stats["total"]),
                    f"{stats['approved']} ({stats['approved']/stats['total']*100:.1f}%)",
                    f"{stats['edited']} ({stats['edited']/stats['total']*100:.1f}%)",
                    f"{stats['rejected']} ({stats['rejected']/stats['total']*100:.1f}%)"
                )
        
        console.print(type_table)
        
        # Create criteria stats table
        criteria_table = Table(title="Statistics by Quality Criteria")
        criteria_table.add_column("Criterion", style="cyan")
        criteria_table.add_column("Passes", style="green")
        criteria_table.add_column("Fails", style="red")
        criteria_table.add_column("Pass Rate", style="yellow")
        
        for criterion, stats in self.stats["by_criteria"].items():
            total = stats["passes"] + stats["fails"]
            if total > 0:
                pass_rate = stats["passes"] / total * 100
                criteria_table.add_row(
                    criterion,
                    str(stats["passes"]),
                    str(stats["fails"]),
                    f"{pass_rate:.1f}%"
                )
        
        console.print(criteria_table)
    
    def validate_directory(self, directory: Union[str, Path], 
                         interactive: bool = True) -> Dict[str, Any]:
        """
        Validate all question files in a directory.
        
        Args:
            directory: Directory containing question files
            interactive: Whether to run in interactive mode
            
        Returns:
            Validation statistics
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory}")
        
        # Find all question files
        question_files = list(directory.glob("*_questions.json"))
        if not question_files:
            logger.warning(f"No question files found in {directory}")
            return self.stats
        
        logger.info(f"Found {len(question_files)} question files in {directory}")
        console.print(f"\n[bold green]Found {len(question_files)} question files in [/bold green][cyan]{directory}[/cyan]")
        
        # Validate each file
        for file in question_files:
            try:
                self.validate_question_file(file, interactive=interactive)
            except Exception as e:
                logger.error(f"Error validating {file}: {e}")
                console.print(f"[bold red]Error validating {file}: {e}[/bold red]")
        
        return self.stats
    
    def save_validation_report(self, output_file: Optional[Union[str, Path]] = None) -> None:
        """
        Save validation statistics to a report file.
        
        Args:
            output_file: Path to output file (default: validation_report.json)
        """
        if output_file is None:
            output_file = self.output_dir / "validation_report.json"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved validation report to {output_file}")
        console.print(f"\n[bold green]Saved validation report to [/bold green][cyan]{output_file}[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate RAG evaluation questions")
    parser.add_argument("input", help="Question file or directory containing question files")
    parser.add_argument("--taxonomy", help="Path to question taxonomy file")
    parser.add_argument("--output-dir", default="question_sets/validated_questions", 
                       help="Output directory for validated questions")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run in non-interactive mode with automatic validation")
    parser.add_argument("--report", help="Path to save validation report")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    validator = QuestionValidator(
        taxonomy_file=args.taxonomy,
        output_dir=args.output_dir
    )
    
    input_path = Path(args.input)
    try:
        if input_path.is_file():
            validator.validate_question_file(input_path, interactive=not args.non_interactive)
        elif input_path.is_dir():
            validator.validate_directory(input_path, interactive=not args.non_interactive)
        else:
            print(f"Error: Input {input_path} is not a valid file or directory")
            sys.exit(1)
            
        if args.report:
            validator.save_validation_report(args.report)
        else:
            validator.save_validation_report()
            
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        console.print(f"[bold red]Error during validation: {e}[/bold red]")
        sys.exit(1)