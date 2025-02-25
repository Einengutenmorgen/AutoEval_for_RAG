#!/usr/bin/env python3
"""
Integration script to run the entire Phase 2 pipeline:
1. Process PDF academic papers
2. Generate questions
3. Validate questions
4. Generate gold standard answers
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Import core modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_processing.academic_pdf_processor import AcademicPaperProcessor
from question_generation.question_generator import QuestionGenerator
from question_generation.validate_questions import QuestionValidator
from question_generation.generate_answers import GoldAnswerGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/phase2_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


class Phase2Runner:
    """Run the entire Phase 2 pipeline."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the Phase 2 runner.
        
        Args:
            config_file: Path to configuration file
        """
        # Initialize directories
        self.base_dir = Path.cwd()
        self.pdf_dir = self.base_dir / "data" / "raw"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.questions_dir = self.base_dir / "question_sets" / "generated_questions"
        self.validated_dir = self.base_dir / "question_sets" / "validated_questions"
        self.gold_answers_dir = self.base_dir / "question_sets" / "gold_answers"
        self.taxonomy_dir = self.base_dir / "question_sets" / "taxonomy"
        self.templates_dir = self.taxonomy_dir / "templates"
        
        # Create directories if they don't exist
        for dir_path in [self.pdf_dir, self.processed_dir, self.questions_dir,
                        self.validated_dir, self.gold_answers_dir, 
                        self.taxonomy_dir, self.templates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.paper_processor = AcademicPaperProcessor(output_dir=str(self.processed_dir))
        self.question_generator = QuestionGenerator(
            taxonomy_file=str(self.taxonomy_dir / "question_taxonomy.json"),
            templates_dir=str(self.templates_dir),
            output_dir=str(self.questions_dir)
        )
        self.question_validator = QuestionValidator(
            taxonomy_file=str(self.taxonomy_dir / "question_taxonomy.json"),
            output_dir=str(self.validated_dir)
        )
        self.answer_generator = GoldAnswerGenerator(
            papers_dir=str(self.processed_dir),
            output_dir=str(self.gold_answers_dir)
        )
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "processing": {
                "process_all_pdfs": True,
                "overwrite_existing": False
            },
            "questions": {
                "questions_per_paper": 30,
                "seed_questions_per_type": 3,
                "generate_seed_only": False
            },
            "validation": {
                "interactive": True,
                "validate_all": True
            },
            "answers": {
                "interactive": True,
                "generate_for_all": True
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for section in default_config:
                    if section in config:
                        default_config[section].update(config[section])
                    
                logger.info(f"Loaded configuration from {config_file}")
                return default_config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return default_config
        
        return default_config
    
    def run_pipeline(self, start_step: int = 1, end_step: int = 4) -> None:
        """
        Run the Phase 2 pipeline.
        
        Args:
            start_step: Step to start from (1=Process PDFs, 2=Generate Questions, 
                                          3=Validate Questions, 4=Generate Answers)
            end_step: Step to end at
        """
        console.print(Panel("[bold green]RAG Evaluation Dataset - Phase 2 Pipeline[/bold green]"))
        
        # Ensure valid step range
        start_step = max(1, min(start_step, 4))
        end_step = max(start_step, min(end_step, 4))
        
        # Step 1: Process PDFs
        if start_step <= 1 and end_step >= 1:
            self._run_pdf_processing()
        
        # Step 2: Generate Questions
        if start_step <= 2 and end_step >= 2:
            self._run_question_generation()
        
        # Step 3: Validate Questions
        if start_step <= 3 and end_step >= 3:
            self._run_question_validation()
        
        # Step 4: Generate Gold Answers
        if start_step <= 4 and end_step >= 4:
            self._run_answer_generation()
        
        console.print(Panel("[bold green]Phase 2 Pipeline Complete![/bold green]"))
    
    def _run_pdf_processing(self) -> None:
        """Run the PDF processing step."""
        console.print("\n[bold blue]Step 1: Processing Academic PDFs[/bold blue]")
        
        # Find PDF files
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            console.print("[bold red]No PDF files found in data/raw directory[/bold red]")
            return
        
        console.print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for pdf_file in pdf_files:
                # Check if already processed
                paper_id = pdf_file.stem
                output_file = self.processed_dir / f"{paper_id}.json"
                
                if output_file.exists() and not self.config["processing"]["overwrite_existing"]:
                    console.print(f"[yellow]Skipping {pdf_file.name} (already processed)[/yellow]")
                    continue
                
                # Process the PDF
                task = progress.add_task(f"Processing {pdf_file.name}...", total=None)
                try:
                    self.paper_processor.process_pdf(pdf_file, paper_id=paper_id)
                    progress.update(task, description=f"[green]Processed {pdf_file.name}[/green]")
                except Exception as e:
                    logger.error(f"Error processing {pdf_file}: {e}")
                    progress.update(task, description=f"[red]Failed to process {pdf_file.name}: {str(e)}[/red]")
        
        console.print("[bold green]PDF processing complete[/bold green]")
    
    def _run_question_generation(self) -> None:
        """Run the question generation step."""
        console.print("\n[bold blue]Step 2: Generating Questions[/bold blue]")
        
        # Find processed paper files
        paper_files = list(self.processed_dir.glob("*.json"))
        if not paper_files:
            console.print("[bold red]No processed paper files found[/bold red]")
            return
        
        console.print(f"Found {len(paper_files)} processed papers")
        
        # Generate questions for each paper
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for paper_file in paper_files:
                paper_id = paper_file.stem
                
                # Determine output files
                questions_file = self.questions_dir / f"{paper_id}_questions.json"
                seed_file = self.questions_dir / f"{paper_id}_seed_questions.json"
                
                # Check if already generated
                if (questions_file.exists() or (self.config["questions"]["generate_seed_only"] and seed_file.exists())) and not self.config["processing"]["overwrite_existing"]:
                    console.print(f"[yellow]Skipping question generation for {paper_id} (already exists)[/yellow]")
                    continue
                
                # Generate questions
                task = progress.add_task(f"Generating questions for {paper_id}...", total=None)
                try:
                    # Load paper data
                    with open(paper_file, 'r', encoding='utf-8') as f:
                        paper_data = json.load(f)
                    
                    if self.config["questions"]["generate_seed_only"]:
                        # Generate seed questions
                        self.question_generator.generate_seed_questions(
                            paper_data,
                            num_per_type=self.config["questions"]["seed_questions_per_type"]
                        )
                    else:
                        # Generate full question set
                        self.question_generator.generate_questions_from_paper(
                            paper_data,
                            num_questions=self.config["questions"]["questions_per_paper"],
                            paper_id=paper_id
                        )
                    
                    progress.update(task, description=f"[green]Generated questions for {paper_id}[/green]")
                except Exception as e:
                    logger.error(f"Error generating questions for {paper_id}: {e}")
                    progress.update(task, description=f"[red]Failed to generate questions for {paper_id}: {str(e)}[/red]")
        
        console.print("[bold green]Question generation complete[/bold green]")
    
    def _run_question_validation(self) -> None:
        """Run the question validation step."""
        console.print("\n[bold blue]Step 3: Validating Questions[/bold blue]")
        
        # Find question files
        question_files = list(self.questions_dir.glob("*_questions.json"))
        if not question_files:
            console.print("[bold red]No question files found[/bold red]")
            return
        
        console.print(f"Found {len(question_files)} question files to validate")
        
        # Validate each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for question_file in question_files:
                # Check if already validated
                paper_id = question_file.stem.split("_")[0]
                output_file = self.validated_dir / f"{paper_id}_questions_validated.json"
                
                if output_file.exists() and not self.config["processing"]["overwrite_existing"]:
                    console.print(f"[yellow]Skipping validation for {question_file.name} (already validated)[/yellow]")
                    continue
                
                # Validate questions
                task = progress.add_task(f"Validating {question_file.name}...", total=None)
                try:
                    if self.config["validation"]["interactive"]:
                        # For interactive validation, we'll use subprocess to run the validator script
                        # This allows for better console interaction
                        progress.stop()
                        subprocess.run([
                            sys.executable,
                            "scripts/question_generation/validate_questions.py",
                            str(question_file),
                            "--output-dir", str(self.validated_dir),
                            "--taxonomy", str(self.taxonomy_dir / "question_taxonomy.json")
                        ])
                        progress.start()
                    else:
                        # Run non-interactive validation
                        self.question_validator.validate_question_file(
                            question_file,
                            interactive=False
                        )
                    
                    progress.update(task, description=f"[green]Validated {question_file.name}[/green]")
                except Exception as e:
                    logger.error(f"Error validating {question_file.name}: {e}")
                    progress.update(task, description=f"[red]Failed to validate {question_file.name}: {str(e)}[/red]")
        
        console.print("[bold green]Question validation complete[/bold green]")
    
    def _run_answer_generation(self) -> None:
        """Run the gold answer generation step."""
        console.print("\n[bold blue]Step 4: Generating Gold Standard Answers[/bold blue]")
        
        # Find validated question files or fall back to regular question files
        validated_files = list(self.validated_dir.glob("*_validated.json"))
        
        if validated_files:
            question_files = validated_files
            console.print(f"Found {len(question_files)} validated question files")
        else:
            question_files = list(self.questions_dir.glob("*_questions.json"))
            console.print(f"No validated question files found, using {len(question_files)} generated question files")
        
        if not question_files:
            console.print("[bold red]No question files found[/bold red]")
            return
        
        # Generate answers for each file
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for question_file in question_files:
                # Check if already processed
                paper_id = question_file.stem.split("_")[0]
                output_file = self.gold_answers_dir / f"{paper_id}_questions_with_gold.json"
                
                if output_file.exists() and not self.config["processing"]["overwrite_existing"]:
                    console.print(f"[yellow]Skipping answer generation for {question_file.name} (already processed)[/yellow]")
                    continue
                
                # Generate answers
                task = progress.add_task(f"Generating answers for {question_file.name}...", total=None)
                try:
                    if self.config["answers"]["interactive"]:
                        # For interactive answer generation, use subprocess
                        progress.stop()
                        subprocess.run([
                            sys.executable,
                            "scripts/question_generation/generate_answers.py",
                            str(question_file),
                            "--papers-dir", str(self.processed_dir),
                            "--output-dir", str(self.gold_answers_dir)
                        ])
                        progress.start()
                    else:
                        # Run non-interactive answer generation
                        self.answer_generator.generate_answers_for_file(
                            question_file,
                            interactive=False
                        )
                    
                    progress.update(task, description=f"[green]Generated answers for {question_file.name}[/green]")
                except Exception as e:
                    logger.error(f"Error generating answers for {question_file.name}: {e}")
                    progress.update(task, description=f"[red]Failed to generate answers for {question_file.name}: {str(e)}[/red]")
        
        console.print("[bold green]Gold answer generation complete[/bold green]")
    
    def generate_phase2_report(self, output_file: Optional[str] = None) -> None:
        """
        Generate a summary report for Phase 2.
        
        Args:
            output_file: Path to output file (default: phase2_report.json)
        """
        if output_file is None:
            output_file = self.base_dir / "reports" / "phase2_report.json"
        else:
            output_file = Path(output_file)
        
        # Create reports directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect statistics
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "papers": {
                "total": len(list(self.pdf_dir.glob("*.pdf"))),
                "processed": len(list(self.processed_dir.glob("*.json")))
            },
            "questions": {
                "total_generated": sum(1 for _ in self._count_questions(self.questions_dir)),
                "total_validated": sum(1 for _ in self._count_questions(self.validated_dir)),
                "by_type": self._count_questions_by_type(),
                "by_complexity": self._count_questions_by_complexity()
            },
            "answers": {
                "total_with_gold": sum(1 for _ in self._count_questions_with_gold()),
                "validated_gold": sum(1 for q in self._count_questions_with_gold() if q.get("gold_answer", {}).get("is_validated", False))
            }
        }
        
        # Save the report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved Phase 2 report to {output_file}")
        console.print(f"\n[bold green]Saved Phase 2 report to [/bold green][cyan]{output_file}[/cyan]")
        
        # Display summary
        console.print("\n[bold]Phase 2 Summary:[/bold]")
        console.print(f"Papers: {report['papers']['processed']}/{report['papers']['total']} processed")
        console.print(f"Questions: {report['questions']['total_generated']} generated, {report['questions']['total_validated']} validated")
        console.print(f"Gold Answers: {report['answers']['total_with_gold']} total, {report['answers']['validated_gold']} validated")
    
    def _count_questions(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Count and yield all questions in a directory.
        
        Args:
            directory: Directory containing question files
            
        Yields:
            Question dictionaries
        """
        if not directory.exists():
            return
        
        for file in directory.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                
                if not isinstance(questions, list):
                    questions = [questions]
                
                for question in questions:
                    yield question
            except Exception:
                continue
    
    def _count_questions_by_type(self) -> Dict[str, int]:
        """
        Count questions by type.
        
        Returns:
            Dictionary of counts by type
        """
        counts = {}
        
        for question in self._count_questions(self.questions_dir):
            q_type = question.get("metadata", {}).get("query_type", "unknown")
            counts[q_type] = counts.get(q_type, 0) + 1
        
        return counts
    
    def _count_questions_by_complexity(self) -> Dict[str, int]:
        """
        Count questions by complexity.
        
        Returns:
            Dictionary of counts by complexity
        """
        counts = {}
        
        for question in self._count_questions(self.questions_dir):
            complexity = question.get("metadata", {}).get("complexity", "unknown")
            counts[complexity] = counts.get(complexity, 0) + 1
        
        return counts
    
    def _count_questions_with_gold(self) -> List[Dict[str, Any]]:
        """
        Count and yield questions with gold answers.
        
        Yields:
            Question dictionaries with gold answers
        """
        if not self.gold_answers_dir.exists():
            return
        
        for file in self.gold_answers_dir.glob("*_with_gold.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    questions = json.load(f)
                
                if not isinstance(questions, list):
                    questions = [questions]
                
                for question in questions:
                    if question.get("gold_answer"):
                        yield question
            except Exception:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 2 pipeline for RAG evaluation dataset creation")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--start-step", type=int, choices=[1, 2, 3, 4], default=1,
                       help="Step to start from (1=Process PDFs, 2=Generate Questions, 3=Validate Questions, 4=Generate Answers)")
    parser.add_argument("--end-step", type=int, choices=[1, 2, 3, 4], default=4,
                       help="Step to end at")
    parser.add_argument("--report", action="store_true",
                       help="Generate Phase 2 summary report")
    parser.add_argument("--report-file", help="Path to save the report")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    runner = Phase2Runner(config_file=args.config)
    
    try:
        if args.report:
            runner.generate_phase2_report(args.report_file)
        else:
            runner.run_pipeline(start_step=args.start_step, end_step=args.end_step)
    except Exception as e:
        logger.error(f"Error running Phase 2 pipeline: {e}", exc_info=True)
        console.print(f"[bold red]Error running Phase 2 pipeline: {e}[/bold red]")
        sys.exit(1)