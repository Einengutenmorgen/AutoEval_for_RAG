#!/usr/bin/env python3
"""
Phase 3 runner for RAG evaluation framework.
Orchestrates the evaluation of RAG systems against the question dataset.
"""

import os
import sys
import json
import yaml
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import subprocess
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
import concurrent.futures
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import evaluation components
from evaluation.evaluation_framework import RAGEvaluationFramework
from evaluation.bias_analyzer import RAGBiasAnalyzer
from evaluation.rag_system_connector import RAGConnectorFactory


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/phase3_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


class Phase3Runner:
    """Orchestrator for Phase 3 RAG evaluation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Phase 3 runner.
        
        Args:
            config_path: Path to configuration file
        """
        # Set default paths
        self.base_dir = Path.cwd()
        self.config_dir = self.base_dir / "config"
        self.questions_dir = self.base_dir / "question_sets" / "gold_answers"
        self.results_dir = self.base_dir / "evaluation" / "results"
        self.reports_dir = self.base_dir / "reports"
        self.bias_reports_dir = self.reports_dir / "bias_analysis"
        
        # Create directories if they don't exist
        for dir_path in [self.results_dir, self.reports_dir, self.bias_reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize evaluation framework
        self.framework = RAGEvaluationFramework(
            config_path=self.config.get("evaluation_config")
        )
        
        # Initialize bias analyzer
        self.bias_analyzer = RAGBiasAnalyzer(
            questions_dir=str(self.questions_dir),
            results_dir=str(self.results_dir),
            reports_dir=str(self.bias_reports_dir)
        )
        
        # Load target systems
        self.connectors = self._load_connectors()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load runner configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "evaluation_config": "config/evaluation_config.yaml",
            "connector_config": "config/connectors_config.yaml",
            "questions_file": None,  # Use all questions if None
            "targets": [],  # Empty means evaluate all systems
            "steps": {
                "evaluate": True,
                "analyze_bias": True,
                "generate_reports": True,
                "run_visualization": False
            },
            "parallelism": 4,
            "output_formats": ["json", "csv", "html"]
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith(('.yaml', '.yml')):
                        custom_config = yaml.safe_load(f)
                    else:
                        custom_config = json.load(f)
                
                # Update default config with custom values
                for key, value in custom_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        return default_config
    
    def _load_connectors(self) -> Dict[str, Any]:
        """
        Load RAG system connectors.
        
        Returns:
            Dictionary of connectors by system name
        """
        connectors = {}
        
        connector_config = self.config.get("connector_config")
        if not connector_config or not os.path.exists(connector_config):
            logger.warning(f"Connector configuration not found: {connector_config}")
            return connectors
        
        try:
            # Load connectors from configuration
            loaded_connectors = RAGConnectorFactory.load_from_config(connector_config)
            
            for connector in loaded_connectors:
                system_name = connector.name
                connectors[system_name] = connector
                logger.info(f"Loaded connector for system: {system_name}")
        except Exception as e:
            logger.error(f"Error loading connectors: {e}")
        
        return connectors
    
    def run_pipeline(self) -> None:
        """Run the Phase 3 evaluation pipeline."""
        console.print(Panel("[bold green]RAG Evaluation Framework - Phase 3[/bold green]"))
        
        # Check if we have any connectors
        if not self.connectors:
            console.print("[bold red]Error:[/bold red] No RAG system connectors found. Please check your connector configuration.")
            return
        
        # Determine which systems to evaluate
        target_systems = self.config.get("targets", [])
        if not target_systems:
            target_systems = list(self.connectors.keys())
        else:
            # Filter to only include available connectors
            target_systems = [name for name in target_systems if name in self.connectors]
        
        if not target_systems:
            console.print("[bold red]Error:[/bold red] No target systems found for evaluation.")
            return
        
        console.print(f"Target systems for evaluation: [cyan]{', '.join(target_systems)}[/cyan]")
        
        # Get pipeline steps
        steps = self.config.get("steps", {})
        
        # Step 1: Evaluate systems
        if steps.get("evaluate", True):
            self._run_evaluation(target_systems)
        
        # Step 2: Analyze bias
        if steps.get("analyze_bias", True):
            self._run_bias_analysis()
        
        # Step 3: Generate reports
        if steps.get("generate_reports", True):
            self._generate_reports(target_systems)
        
        # Step 4: Run visualization dashboard
        if steps.get("run_visualization", False):
            self._run_visualization_dashboard()
        
        console.print(Panel("[bold green]Phase 3 Pipeline Complete![/bold green]"))
    
    def _run_evaluation(self, target_systems: List[str]) -> None:
        """
        Run the evaluation for target systems.
        
        Args:
            target_systems: List of system names to evaluate
        """
        console.print("\n[bold blue]Step 1: Evaluating RAG Systems[/bold blue]")
        
        questions_file = self.config.get("questions_file")
        
        # Evaluate each system
        for system_name in target_systems:
            if system_name not in self.connectors:
                console.print(f"[yellow]Warning:[/yellow] System '{system_name}' not found in connectors. Skipping.")
                continue
            
            console.print(f"\nEvaluating system: [cyan]{system_name}[/cyan]")
            
            try:
                # Get system information
                system_info = self.connectors[system_name].get_system_info()
                
                # Create a temporary system configuration for the evaluation framework
                system_config = {
                    "name": system_name,
                    "description": system_info.get("description", ""),
                    "api_endpoint": "connector",  # Special marker for direct connector use
                    "connector": self.connectors[system_name]
                }
                
                # Register the system in the evaluation framework
                self.framework.config["target_systems"] = [system_config]
                
                # Run evaluation
                output_file = self.results_dir / f"{system_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Evaluating {system_name}...", total=None)
                    
                    # Override query method to use connector directly
                    original_query_method = self.framework._query_system
                    
                    def connector_query_override(framework, sys_config, question):
                        if "connector" in sys_config:
                            connector = sys_config["connector"]
                            metadata = question.get("metadata", {})
                            start_time = time.time()
                            
                            try:
                                # Call the connector directly
                                response = connector.query(question["question"], metadata)
                                # Add the question to the response
                                response["question"] = question
                                end_time = time.time()
                                response["latency"] = end_time - start_time
                                return response
                            except Exception as e:
                                logger.error(f"Error querying connector: {e}")
                                return {
                                    "question": question,
                                    "error": str(e),
                                    "success": False,
                                    "latency": time.time() - start_time
                                }
                        else:
                            # Fall back to original method
                            return original_query_method(framework, sys_config, question)
                    
                    # Replace method temporarily
                    self.framework._query_system = lambda sys_config, question: connector_query_override(self.framework, sys_config, question)
                    
                    try:
                        # Run evaluation
                        results = self.framework.evaluate_system(
                            system_name,
                            questions_file=questions_file,
                            output_file=str(output_file)
                        )
                        
                        # Update progress
                        progress.update(task, description=f"[green]Evaluation complete for {system_name}[/green]")
                    finally:
                        # Restore original method
                        self.framework._query_system = original_query_method
                
                # Display summary
                if results and "summary" in results:
                    summary = results["summary"]
                    
                    console.print(f"Total questions: [bold]{summary.get('total_questions', 0)}[/bold]")
                    console.print(f"Success rate: [bold]{summary.get('success_rate', 0)*100:.1f}%[/bold]")
                    console.print(f"Average latency: [bold]{summary.get('average_latency', 0):.2f} s[/bold]")
                    
                    if "rag_effectiveness_score" in summary:
                        console.print(f"RAG effectiveness score: [bold]{summary['rag_effectiveness_score']:.2f}[/bold]")
                
                console.print(f"Results saved to: [cyan]{output_file}[/cyan]")
                
            except Exception as e:
                console.print(f"[bold red]Error evaluating {system_name}: {e}[/bold red]")
                logger.error(f"Error evaluating {system_name}: {e}", exc_info=True)
    
    def _run_bias_analysis(self) -> None:
        """Run bias analysis on the evaluation results."""
        console.print("\n[bold blue]Step 2: Analyzing Bias in Dataset and Results[/bold blue]")
        
        # Load questions
        self.bias_analyzer.load_questions()
        
        # Load evaluation results
        self.bias_analyzer.load_evaluation_results()
        
        # Check if we have questions and results
        if not self.bias_analyzer.questions:
            console.print("[yellow]Warning:[/yellow] No questions loaded for bias analysis.")
            return
        
        console.print(f"Loaded {len(self.bias_analyzer.questions)} questions for analysis.")
        
        if not self.bias_analyzer.results:
            console.print("[yellow]Warning:[/yellow] No evaluation results loaded for bias analysis.")
        else:
            console.print(f"Loaded results for {len(self.bias_analyzer.results)} systems.")
        
        # Run dataset bias analysis
        console.print("\nAnalyzing dataset bias...")
        
        dataset_output_file = self.bias_reports_dir / "dataset_bias_analysis.json"
        self.bias_analyzer.analyze_dataset_bias(str(dataset_output_file))
        
        console.print(f"Dataset bias analysis saved to: [cyan]{dataset_output_file}[/cyan]")
        
        # Run performance bias analysis if we have results
        if self.bias_analyzer.results:
            console.print("\nAnalyzing performance bias...")
            
            performance_output_file = self.bias_reports_dir / "performance_bias_analysis.json"
            self.bias_analyzer.analyze_performance_bias(str(performance_output_file))
            
            console.print(f"Performance bias analysis saved to: [cyan]{performance_output_file}[/cyan]")
        
        # Generate adversarial examples based on bias analysis
        console.print("\nGenerating adversarial examples...")
        
        adversarial_output_file = self.questions_dir / "adversarial_examples.json"
        adversarial_examples = self.bias_analyzer.generate_adversarial_examples(str(adversarial_output_file))
        
        console.print(f"Generated {len(adversarial_examples)} adversarial examples saved to: [cyan]{adversarial_output_file}[/cyan]")
    
    def _generate_reports(self, target_systems: List[str]) -> None:
        """
        Generate evaluation reports.
        
        Args:
            target_systems: List of system names that were evaluated
        """
        console.print("\n[bold blue]Step 3: Generating Evaluation Reports[/bold blue]")
        
        # Get output formats
        output_formats = self.config.get("output_formats", ["json", "csv", "html"])
        
        # Generate reports for each system
        for system_name in target_systems:
            # Find the latest evaluation results for this system
            result_files = list(self.results_dir.glob(f"{system_name}_results_*.json"))
            if not result_files:
                console.print(f"[yellow]Warning:[/yellow] No evaluation results found for system '{system_name}'. Skipping report generation.")
                continue
            
            # Use the latest results file
            latest_file = max(result_files, key=os.path.getmtime)
            evaluation_id = latest_file.stem
            
            console.print(f"\nGenerating reports for system: [cyan]{system_name}[/cyan]")
            
            try:
                # Load results into framework
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                # Ensure framework has the results loaded
                self.framework.results = {evaluation_id: results}
                
                # Generate report
                report_base = self.reports_dir / f"{system_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Generating reports...", total=None)
                    
                    report_file = self.framework.generate_report(
                        evaluation_id,
                        formats=output_formats,
                        output_file=str(report_base)
                    )
                    
                    progress.update(task, description=f"[green]Reports generated[/green]")
                
                console.print(f"Reports saved to: [cyan]{report_base}.*[/cyan]")
                
            except Exception as e:
                console.print(f"[bold red]Error generating reports for {system_name}: {e}[/bold red]")
                logger.error(f"Error generating reports for {system_name}: {e}", exc_info=True)
        
        # Generate system comparison report if we have multiple systems
        if len(target_systems) > 1:
            console.print("\nGenerating system comparison report...")
            
            try:
                # Find all evaluation IDs
                evaluation_ids = []
                
                for system_name in target_systems:
                    result_files = list(self.results_dir.glob(f"{system_name}_results_*.json"))
                    if result_files:
                        latest_file = max(result_files, key=os.path.getmtime)
                        evaluation_id = latest_file.stem
                        
                        # Load results into framework if not already loaded
                        if evaluation_id not in self.framework.results:
                            with open(latest_file, 'r') as f:
                                self.framework.results[evaluation_id] = json.load(f)
                        
                        evaluation_ids.append(evaluation_id)
                
                if len(evaluation_ids) > 1:
                    # Generate comparison report
                    comparison_file = self.reports_dir / f"systems_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    self.framework.compare_systems(
                        evaluation_ids,
                        output_file=str(comparison_file)
                    )
                    
                    console.print(f"System comparison report saved to: [cyan]{comparison_file}[/cyan]")
                else:
                    console.print("[yellow]Warning:[/yellow] Not enough systems with results for comparison.")
                
            except Exception as e:
                console.print(f"[bold red]Error generating system comparison report: {e}[/bold red]")
                logger.error(f"Error generating system comparison report: {e}", exc_info=True)
    
    def _run_visualization_dashboard(self) -> None:
        """Run the visualization dashboard."""
        console.print("\n[bold blue]Step 4: Running Visualization Dashboard[/bold blue]")
        
        try:
            # Check if streamlit is available
            try:
                import streamlit
                streamlit_available = True
            except ImportError:
                streamlit_available = False
                console.print("[yellow]Warning:[/yellow] Streamlit not found. Please install with: pip install streamlit")
                return
            
            if streamlit_available:
                # Get dashboard script path
                dashboard_script = Path(__file__).parent / "evaluation" / "visualization_dashboard.py"
                
                if not dashboard_script.exists():
                    console.print(f"[bold red]Error:[/bold red] Dashboard script not found: {dashboard_script}")
                    return
                
                # Run the dashboard
                console.print("Starting visualization dashboard...")
                console.print("Press Ctrl+C to stop the dashboard\n")
                
                # Run the dashboard using subprocess
                cmd = ["streamlit", "run", str(dashboard_script), "--server.port=8501"]
                process = subprocess.Popen(cmd)
                
                # Wait for the dashboard to start
                time.sleep(2)
                
                # Open in browser
                console.print("Dashboard running at: [link=http://localhost:8501]http://localhost:8501[/link]")
                
                # Wait for user to stop the dashboard
                try:
                    process.wait()
                except KeyboardInterrupt:
                    console.print("\nStopping dashboard...")
                    process.terminate()
                
        except Exception as e:
            console.print(f"[bold red]Error running visualization dashboard: {e}[/bold red]")
            logger.error(f"Error running visualization dashboard: {e}", exc_info=True)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework - Phase 3 Runner")
    parser.add_argument("--config", help="Path to Phase 3 configuration file")
    parser.add_argument("--questions", help="Path to specific questions file")
    parser.add_argument("--systems", nargs="+", help="Specific systems to evaluate")
    parser.add_argument("--step", choices=["evaluate", "analyze", "report", "visualize", "all"], 
                      default="all", help="Run specific step only")
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize runner
        runner = Phase3Runner(config_path=args.config)
        
        # Update configuration from command-line arguments
        if args.questions:
            runner.config["questions_file"] = args.questions
        
        if args.systems:
            runner.config["targets"] = args.systems
        
        if args.step != "all":
            # Reset all steps to False
            for step in runner.config["steps"]:
                runner.config["steps"][step] = False
            
            # Enable only the requested step
            if args.step == "evaluate":
                runner.config["steps"]["evaluate"] = True
            elif args.step == "analyze":
                runner.config["steps"]["analyze_bias"] = True
            elif args.step == "report":
                runner.config["steps"]["generate_reports"] = True
            elif args.step == "visualize":
                runner.config["steps"]["run_visualization"] = True
        
        # Run the pipeline
        runner.run_pipeline()
        
    except Exception as e:
        logger.error(f"Error running Phase 3 pipeline: {e}", exc_info=True)
        console.print(f"[bold red]Error running Phase 3 pipeline: {e}[/bold red]")
        sys.exit(1)