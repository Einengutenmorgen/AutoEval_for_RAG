#!/usr/bin/env python3
"""
Interactive visualization dashboard for RAG evaluation results.
Creates a Streamlit dashboard to explore evaluation metrics and performance.
"""

import os
import sys
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGVisualizationDashboard:
    """Dashboard for visualizing RAG evaluation results."""
    
    def __init__(self, results_dir: str = "evaluation/results"):
        """
        Initialize the visualization dashboard.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        self.comparison_data = {}
        
        # Load results
        self._load_results()
    
    def _load_results(self) -> None:
        """Load all evaluation results from the results directory."""
        if not self.results_dir.exists():
            logger.warning(f"Results directory {self.results_dir} does not exist")
            return
        
        # Find result files
        result_files = list(self.results_dir.glob("*_results_*.json"))
        if not result_files:
            logger.warning(f"No result files found in {self.results_dir}")
            return
        
        # Load each file
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Use system name as key
                if "system" in data and "name" in data["system"]:
                    system_name = data["system"]["name"]
                    self.results[system_name] = data
                    logger.info(f"Loaded results for system: {system_name}")
                else:
                    logger.warning(f"Could not determine system name for {file_path}")
            except Exception as e:
                logger.error(f"Error loading results from {file_path}: {e}")
        
        logger.info(f"Loaded results for {len(self.results)} systems")
    
    def _prepare_comparison_data(self) -> None:
        """Prepare data for system comparison."""
        if not self.results:
            return
        
        # Prepare metrics comparison
        metrics_data = []
        
        for system_name, results in self.results.items():
            if "metrics" in results:
                metrics = results["metrics"]
                
                # Handle retrieval metrics
                if "retrieval" in metrics:
                    retrieval = metrics["retrieval"]
                    
                    # Process precision@k
                    if "precision_at_k" in retrieval:
                        for k, value in retrieval["precision_at_k"].items():
                            metrics_data.append({
                                "System": system_name,
                                "Category": "Retrieval",
                                "Metric": f"Precision@{k}",
                                "Value": value
                            })
                    
                    # Process recall@k
                    if "recall_at_k" in retrieval:
                        for k, value in retrieval["recall_at_k"].items():
                            metrics_data.append({
                                "System": system_name,
                                "Category": "Retrieval",
                                "Metric": f"Recall@{k}",
                                "Value": value
                            })
                    
                    # Process MRR
                    if "mrr" in retrieval:
                        metrics_data.append({
                            "System": system_name,
                            "Category": "Retrieval",
                            "Metric": "MRR",
                            "Value": retrieval["mrr"]
                        })
                
                # Handle answer metrics
                if "answer" in metrics:
                    answer = metrics["answer"]
                    
                    # Process ROUGE scores
                    if "rouge" in answer:
                        rouge = answer["rouge"]
                        for metric, value in rouge.items():
                            metrics_data.append({
                                "System": system_name,
                                "Category": "Answer",
                                "Metric": f"ROUGE-{metric}",
                                "Value": value
                            })
                    
                    # Process BLEU
                    if "bleu" in answer:
                        metrics_data.append({
                            "System": system_name,
                            "Category": "Answer",
                            "Metric": "BLEU",
                            "Value": answer["bleu"]
                        })
                    
                    # Process BERTScore
                    if "bertscore" in answer:
                        bertscore = answer["bertscore"]
                        for metric, value in bertscore.items():
                            metrics_data.append({
                                "System": system_name,
                                "Category": "Answer",
                                "Metric": f"BERTScore-{metric}",
                                "Value": value
                            })
                
                # Handle robustness metrics
                if "robustness" in metrics:
                    robustness = metrics["robustness"]
                    
                    if "impossible_detection" in robustness:
                        metrics_data.append({
                            "System": system_name,
                            "Category": "Robustness",
                            "Metric": "Impossible Detection",
                            "Value": robustness["impossible_detection"]["accuracy"]
                        })
        
        # Create DataFrame
        self.comparison_data["metrics"] = pd.DataFrame(metrics_data)
        
        # Prepare performance by query type comparison
        query_type_data = []
        
        for system_name, results in self.results.items():
            if "responses" in results:
                responses = results["responses"]
                
                # Group by query type
                query_type_scores = {}
                query_type_counts = {}
                
                for response in responses:
                    if not response.get("success", False):
                        continue
                    
                    # Get query type
                    query_type = response.get("question", {}).get("metadata", {}).get("query_type", "unknown")
                    
                    # Get score - either use a pre-calculated score or estimate from ROUGE if available
                    score = None
                    
                    if "score" in response:
                        score = response["score"]
                    elif "metrics" in response and "rouge" in response["metrics"]:
                        rouge_metrics = response["metrics"]["rouge"]
                        if "rougeL_fmeasure" in rouge_metrics:
                            score = rouge_metrics["rougeL_fmeasure"]
                    
                    # If no score available, use a default
                    if score is None:
                        score = 0.5
                    
                    # Add to query type scores
                    if query_type not in query_type_scores:
                        query_type_scores[query_type] = []
                        query_type_counts[query_type] = 0
                    
                    query_type_scores[query_type].append(score)
                    query_type_counts[query_type] += 1
                
                # Calculate average scores
                for query_type, scores in query_type_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        query_type_data.append({
                            "System": system_name,
                            "Query Type": query_type,
                            "Average Score": avg_score,
                            "Count": query_type_counts[query_type]
                        })
        
        # Create DataFrame
        self.comparison_data["query_type"] = pd.DataFrame(query_type_data)
        
        # Prepare performance by complexity comparison
        complexity_data = []
        
        for system_name, results in self.results.items():
            if "responses" in results:
                responses = results["responses"]
                
                # Group by complexity
                complexity_scores = {}
                complexity_counts = {}
                
                for response in responses:
                    if not response.get("success", False):
                        continue
                    
                    # Get complexity
                    complexity = response.get("question", {}).get("metadata", {}).get("complexity", "unknown")
                    
                    # Get score
                    score = None
                    
                    if "score" in response:
                        score = response["score"]
                    elif "metrics" in response and "rouge" in response["metrics"]:
                        rouge_metrics = response["metrics"]["rouge"]
                        if "rougeL_fmeasure" in rouge_metrics:
                            score = rouge_metrics["rougeL_fmeasure"]
                    
                    # If no score available, use a default
                    if score is None:
                        score = 0.5
                    
                    # Add to complexity scores
                    if complexity not in complexity_scores:
                        complexity_scores[complexity] = []
                        complexity_counts[complexity] = 0
                    
                    complexity_scores[complexity].append(score)
                    complexity_counts[complexity] += 1
                
                # Calculate average scores
                for complexity, scores in complexity_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        complexity_data.append({
                            "System": system_name,
                            "Complexity": complexity,
                            "Average Score": avg_score,
                            "Count": complexity_counts[complexity]
                        })
        
        # Create DataFrame
        self.comparison_data["complexity"] = pd.DataFrame(complexity_data)
        
        # Prepare latency comparison
        latency_data = []
        
        for system_name, results in self.results.items():
            if "responses" in results:
                responses = results["responses"]
                
                # Collect latencies
                latencies = [r.get("latency", 0) for r in responses if r.get("success", False)]
                
                if latencies:
                    latency_data.append({
                        "System": system_name,
                        "Min Latency": min(latencies),
                        "Max Latency": max(latencies),
                        "Avg Latency": sum(latencies) / len(latencies),
                        "Median Latency": sorted(latencies)[len(latencies) // 2],
                        "Count": len(latencies)
                    })
        
        # Create DataFrame
        self.comparison_data["latency"] = pd.DataFrame(latency_data)
    
    def run_dashboard(self) -> None:
        """Run the Streamlit dashboard."""
        # Set page configuration
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        # Title and description
        st.title("RAG Evaluation Dashboard")
        st.write("""
        This dashboard visualizes the results of evaluating Retrieval-Augmented Generation (RAG) systems.
        Explore metrics, performance breakdowns, and system comparisons.
        """)
        
        # Check if we have results
        if not self.results:
            st.warning("No evaluation results found. Please run evaluations first.")
            return
        
        # Prepare comparison data if not already done
        if not self.comparison_data:
            self._prepare_comparison_data()
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select a page",
            ["Overview", "System Details", "Metrics Comparison", "Query Type Analysis", 
             "Complexity Analysis", "Latency Analysis", "Example Responses"]
        )
        
        # System selection for details page
        if page == "System Details":
            selected_system = st.sidebar.selectbox(
                "Select a system",
                list(self.results.keys())
            )
        
        # Render the selected page
        if page == "Overview":
            self._render_overview_page()
        elif page == "System Details":
            self._render_system_details_page(selected_system)
        elif page == "Metrics Comparison":
            self._render_metrics_comparison_page()
        elif page == "Query Type Analysis":
            self._render_query_type_analysis_page()
        elif page == "Complexity Analysis":
            self._render_complexity_analysis_page()
        elif page == "Latency Analysis":
            self._render_latency_analysis_page()
        elif page == "Example Responses":
            self._render_example_responses_page()
    
    def _render_overview_page(self) -> None:
        """Render the overview page."""
        st.header("Overview")
        
        # Display general statistics
        st.subheader("Systems Evaluated")
        
        # Create a table of system info
        system_info = []
        for system_name, results in self.results.items():
            info = {
                "System": system_name,
                "Description": results.get("system", {}).get("description", ""),
                "Questions": len(results.get("responses", [])),
                "Success Rate": sum(1 for r in results.get("responses", []) if r.get("success", False)) / 
                              len(results.get("responses", [])) if results.get("responses", []) else 0,
                "Avg. Latency": sum(r.get("latency", 0) for r in results.get("responses", []) if r.get("success", False)) / 
                              sum(1 for r in results.get("responses", []) if r.get("success", False)) 
                              if sum(1 for r in results.get("responses", []) if r.get("success", False)) > 0 else 0,
                "Evaluation Date": results.get("timestamp", "Unknown")
            }
            system_info.append(info)
        
        # Convert to DataFrame and display
        if system_info:
            df_systems = pd.DataFrame(system_info)
            
            # Format success rate as percentage
            df_systems["Success Rate"] = df_systems["Success Rate"].map("{:.1%}".format)
            
            # Format latency in seconds
            df_systems["Avg. Latency"] = df_systems["Avg. Latency"].map("{:.2f} s".format)
            
            # Display as a table
            st.table(df_systems)
        
        # Display overall performance comparison
        st.subheader("Performance Comparison")
        
        if "metrics" in self.comparison_data:
            # Get key metrics for comparison
            key_metrics = [
                ("Retrieval", "Precision@5"),
                ("Retrieval", "MRR"),
                ("Answer", "ROUGE-rougeL_fmeasure"),
                ("Answer", "BERTScore-f1")
            ]
            
            # Filter for key metrics
            df_metrics = self.comparison_data["metrics"]
            df_key_metrics = pd.DataFrame()
            
            for category, metric in key_metrics:
                metric_data = df_metrics[(df_metrics["Category"] == category) & 
                                       (df_metrics["Metric"] == metric)]
                if not metric_data.empty:
                    df_key_metrics = pd.concat([df_key_metrics, metric_data])
            
            if not df_key_metrics.empty:
                # Create bar chart
                fig = px.bar(
                    df_key_metrics,
                    x="System",
                    y="Value",
                    color="Metric",
                    barmode="group",
                    title="Key Metrics Comparison",
                    labels={"Value": "Score", "System": "System"},
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display RAG effectiveness scores if available
        effectiveness_data = []
        for system_name, results in self.results.items():
            if "summary" in results and "rag_effectiveness_score" in results["summary"]:
                effectiveness_data.append({
                    "System": system_name,
                    "RAG Effectiveness Score": results["summary"]["rag_effectiveness_score"]
                })
        
        if effectiveness_data:
            st.subheader("RAG Effectiveness Scores")
            df_effectiveness = pd.DataFrame(effectiveness_data)
            
            # Create horizontal bar chart
            fig = px.bar(
                df_effectiveness,
                x="RAG Effectiveness Score",
                y="System",
                orientation="h",
                title="Overall RAG Effectiveness",
                labels={"RAG Effectiveness Score": "Score", "System": "System"},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_system_details_page(self, system_name: str) -> None:
        """
        Render the system details page.
        
        Args:
            system_name: Name of the selected system
        """
        st.header(f"System Details: {system_name}")
        
        if system_name not in self.results:
            st.warning(f"No results found for system: {system_name}")
            return
        
        results = self.results[system_name]
        
        # System information
        st.subheader("System Information")
        system_info = results.get("system", {})
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Description:** {system_info.get('description', 'N/A')}")
            st.write(f"**API Endpoint:** {system_info.get('api_endpoint', 'N/A')}")
        
        with col2:
            st.write(f"**Request Format:** {system_info.get('request_format', 'N/A')}")
            st.write(f"**Authentication Type:** {system_info.get('authentication', {}).get('type', 'N/A')}")
        
        # Overall statistics
        st.subheader("Overall Statistics")
        
        responses = results.get("responses", [])
        successful_responses = [r for r in responses if r.get("success", False)]
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", len(responses))
        
        with col2:
            success_rate = len(successful_responses) / len(responses) if responses else 0
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        with col3:
            avg_latency = sum(r.get("latency", 0) for r in successful_responses) / len(successful_responses) if successful_responses else 0
            st.metric("Avg. Latency", f"{avg_latency:.2f} s")
        
        with col4:
            if "summary" in results and "rag_effectiveness_score" in results["summary"]:
                effectiveness = results["summary"]["rag_effectiveness_score"]
                st.metric("RAG Effectiveness", f"{effectiveness:.2f}")
            else:
                st.metric("RAG Effectiveness", "N/A")
        
        # Metrics details
        st.subheader("Metrics")
        
        if "metrics" in results:
            metrics = results["metrics"]
            
            # Create tabs for different metric categories
            tabs = st.tabs(["Retrieval", "Answer", "Robustness"])
            
            # Retrieval metrics
            with tabs[0]:
                if "retrieval" in metrics:
                    retrieval = metrics["retrieval"]
                    
                    # Create DataFrame for retrieval metrics
                    retrieval_data = []
                    
                    # Process precision@k
                    if "precision_at_k" in retrieval:
                        for k, value in retrieval["precision_at_k"].items():
                            retrieval_data.append({
                                "Metric": f"Precision@{k}",
                                "Value": value
                            })
                    
                    # Process recall@k
                    if "recall_at_k" in retrieval:
                        for k, value in retrieval["recall_at_k"].items():
                            retrieval_data.append({
                                "Metric": f"Recall@{k}",
                                "Value": value
                            })
                    
                    # Process MRR
                    if "mrr" in retrieval:
                        retrieval_data.append({
                            "Metric": "MRR",
                            "Value": retrieval["mrr"]
                        })
                    
                    # Display table
                    if retrieval_data:
                        df_retrieval = pd.DataFrame(retrieval_data)
                        
                        # Format values
                        df_retrieval["Value"] = df_retrieval["Value"].map("{:.4f}".format)
                        
                        st.table(df_retrieval)
                        
                        # Create bar chart
                        fig = px.bar(
                            df_retrieval,
                            x="Metric",
                            y="Value",
                            title="Retrieval Metrics",
                            labels={"Value": "Score", "Metric": "Metric"},
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No retrieval metrics available")
            
            # Answer metrics
            with tabs[1]:
                if "answer" in metrics:
                    answer = metrics["answer"]
                    
                    # Create DataFrame for answer metrics
                    answer_data = []
                    
                    # Process ROUGE scores
                    if "rouge" in answer:
                        rouge = answer["rouge"]
                        for metric, value in rouge.items():
                            answer_data.append({
                                "Metric": f"ROUGE-{metric}",
                                "Value": value
                            })
                    
                    # Process BLEU
                    if "bleu" in answer:
                        answer_data.append({
                            "Metric": "BLEU",
                            "Value": answer["bleu"]
                        })
                    
                    # Process BERTScore
                    if "bertscore" in answer:
                        bertscore = answer["bertscore"]
                        for metric, value in bertscore.items():
                            answer_data.append({
                                "Metric": f"BERTScore-{metric}",
                                "Value": value
                            })
                    
                    # Display table
                    if answer_data:
                        df_answer = pd.DataFrame(answer_data)
                        
                        # Format values
                        df_answer["Value"] = df_answer["Value"].map("{:.4f}".format)
                        
                        st.table(df_answer)
                        
                        # Create bar chart
                        fig = px.bar(
                            df_answer,
                            x="Metric",
                            y="Value",
                            title="Answer Quality Metrics",
                            labels={"Value": "Score", "Metric": "Metric"},
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No answer metrics available")
            
            # Robustness metrics
            with tabs[2]:
                if "robustness" in metrics:
                    robustness = metrics["robustness"]
                    
                    if "impossible_detection" in robustness:
                        impossible = robustness["impossible_detection"]
                        
                        # Display metrics
                        st.write(f"**Impossible Question Detection Accuracy:** {impossible['accuracy']:.4f}")
                        st.write(f"**Impossible Questions:** {impossible['impossible_count']}")
                        st.write(f"**Possible Questions:** {impossible['possible_count']}")
                        
                        # Create pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=['Correct Detection', 'Incorrect Detection'],
                            values=[impossible['accuracy'] * impossible['impossible_count'], 
                                   (1 - impossible['accuracy']) * impossible['impossible_count']],
                            hole=.3
                        )])
                        fig.update_layout(title_text="Impossible Question Detection")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No impossible question detection metrics available")
                else:
                    st.info("No robustness metrics available")
        else:
            st.info("No metrics available for this system")
        
        # Best and worst examples
        st.subheader("Best and Worst Examples")
        
        if "summary" in results:
            summary = results["summary"]
            
            if "best_examples" in summary and "worst_examples" in summary:
                # Create tabs for best and worst examples
                tabs = st.tabs(["Best Examples", "Worst Examples"])
                
                # Best examples
                with tabs[0]:
                    for i, example in enumerate(summary["best_examples"]):
                        index = example.get("index", 0)
                        question = example.get("question", "")
                        score = example.get("score", 0)
                        
                        st.write(f"**Example {i+1}** (Score: {score:.4f})")
                        st.write(f"Question: {question}")
                        
                        if index < len(responses):
                            response = responses[index]
                            answer = response.get("answer", "")
                            gold_answer = response.get("question", {}).get("gold_answer", {}).get("text", "")
                            
                            # Display answer and gold answer
                            st.write(f"**System Answer:** {answer}")
                            st.write(f"**Gold Answer:** {gold_answer}")
                            st.markdown("---")
                
                # Worst examples
                with tabs[1]:
                    for i, example in enumerate(summary["worst_examples"]):
                        index = example.get("index", 0)
                        question = example.get("question", "")
                        score = example.get("score", 0)
                        
                        st.write(f"**Example {i+1}** (Score: {score:.4f})")
                        st.write(f"Question: {question}")
                        
                        if index < len(responses):
                            response = responses[index]
                            answer = response.get("answer", "")
                            gold_answer = response.get("question", {}).get("gold_answer", {}).get("text", "")
                            
                            # Display answer and gold answer
                            st.write(f"**System Answer:** {answer}")
                            st.write(f"**Gold Answer:** {gold_answer}")
                            st.markdown("---")
            else:
                st.info("No example information available")
        else:
            st.info("No summary available for this system")
    
    def _render_metrics_comparison_page(self) -> None:
        """Render the metrics comparison page."""
        st.header("Metrics Comparison")
        
        if "metrics" not in self.comparison_data:
            st.warning("No metrics comparison data available")
            return
        
        df_metrics = self.comparison_data["metrics"]
        
        # Filter options
        st.subheader("Filter Options")
        
        # Get unique categories and metrics
        categories = sorted(df_metrics["Category"].unique())
        
        # Allow selecting categories
        selected_categories = st.multiselect(
            "Select categories",
            categories,
            default=categories
        )
        
        # Filter metrics by selected categories
        filtered_metrics = df_metrics[df_metrics["Category"].isin(selected_categories)]
        
        # Get unique metrics for selected categories
        metrics_by_category = {}
        for category in selected_categories:
            metrics_by_category[category] = sorted(filtered_metrics[filtered_metrics["Category"] == category]["Metric"].unique())
        
        # Allow selecting metrics for each category
        selected_metrics = []
        for category in selected_categories:
            if metrics_by_category[category]:
                selected = st.multiselect(
                    f"Select {category} metrics",
                    metrics_by_category[category],
                    default=metrics_by_category[category]
                )
                for metric in selected:
                    selected_metrics.append((category, metric))
        
        # Final filtering
        final_filter = pd.DataFrame()
        for category, metric in selected_metrics:
            metric_data = df_metrics[(df_metrics["Category"] == category) & 
                                   (df_metrics["Metric"] == metric)]
            final_filter = pd.concat([final_filter, metric_data])
        
        # Display visualization options
        st.subheader("Visualization")
        
        viz_type = st.radio(
            "Select visualization type",
            ["Bar Chart", "Radar Chart", "Heatmap"]
        )
        
        if not final_filter.empty:
            if viz_type == "Bar Chart":
                # Create grouped bar chart
                fig = px.bar(
                    final_filter,
                    x="Metric",
                    y="Value",
                    color="System",
                    barmode="group",
                    title="Metrics Comparison",
                    labels={"Value": "Score", "Metric": "Metric"},
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Radar Chart":
                # Create radar chart
                # Need to pivot the data
                pivot = final_filter.pivot(index="System", columns="Metric", values="Value")
                
                # Create radar chart
                fig = go.Figure()
                
                for system in pivot.index:
                    fig.add_trace(go.Scatterpolar(
                        r=pivot.loc[system].values,
                        theta=pivot.columns,
                        fill='toself',
                        name=system
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Metrics Comparison (Radar Chart)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Heatmap":
                # Create heatmap
                # Need to pivot the data
                pivot = final_filter.pivot(index="System", columns="Metric", values="Value")
                
                # Create heatmap
                fig = px.imshow(
                    pivot,
                    labels=dict(x="Metric", y="System", color="Score"),
                    title="Metrics Comparison (Heatmap)",
                    height=500,
                    color_continuous_scale="viridis"
                )
                
                # Add text annotations
                for i in range(len(pivot.index)):
                    for j in range(len(pivot.columns)):
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=f"{pivot.iloc[i, j]:.3f}",
                            showarrow=False,
                            font=dict(color="white" if pivot.iloc[i, j] < 0.5 else "black")
                        )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters")
        
        # Display raw data
        st.subheader("Raw Data")
        
        if not final_filter.empty:
            # Format values
            final_filter["Value"] = final_filter["Value"].map("{:.4f}".format)
            
            # Display as a table
            st.dataframe(final_filter)
            
            # Download option
            csv = final_filter.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "metrics_comparison.csv",
                "text/csv",
                key='download-metrics-csv'
            )
        else:
            st.warning("No data available for the selected filters")
    
    def _render_query_type_analysis_page(self) -> None:
        """Render the query type analysis page."""
        st.header("Query Type Analysis")
        
        if "query_type" not in self.comparison_data:
            st.warning("No query type analysis data available")
            return
        
        df_query_type = self.comparison_data["query_type"]
        
        # Overview
        st.subheader("Performance by Query Type")
        
        # Get unique systems and query types
        systems = sorted(df_query_type["System"].unique())
        query_types = sorted(df_query_type["Query Type"].unique())
        
        # Allow selecting systems
        selected_systems = st.multiselect(
            "Select systems",
            systems,
            default=systems
        )
        
        # Filter by selected systems
        filtered_data = df_query_type[df_query_type["System"].isin(selected_systems)]
        
        if not filtered_data.empty:
            # Create bar chart
            fig = px.bar(
                filtered_data,
                x="Query Type",
                y="Average Score",
                color="System",
                barmode="group",
                title="Average Score by Query Type",
                labels={"Average Score": "Score", "Query Type": "Query Type"},
                height=500
            )
            
            # Add count as text
            for system in selected_systems:
                system_data = filtered_data[filtered_data["System"] == system]
                for i, row in system_data.iterrows():
                    fig.add_annotation(
                        x=row["Query Type"],
                        y=row["Average Score"],
                        text=f"n={row['Count']}",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10)
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create heatmap
            pivot = filtered_data.pivot(index="System", columns="Query Type", values="Average Score")
            
            fig = px.imshow(
                pivot,
                labels=dict(x="Query Type", y="System", color="Score"),
                title="Performance Heatmap by Query Type",
                height=400,
                color_continuous_scale="viridis"
            )
            
            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=f"{pivot.iloc[i, j]:.3f}",
                        showarrow=False,
                        font=dict(color="white" if pivot.iloc[i, j] < 0.5 else "black")
                    )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected systems")
        
        # Performance variation
        st.subheader("Performance Variation Analysis")
        
        if not filtered_data.empty and len(selected_systems) > 0:
            # Calculate performance variation across query types for each system
            variation_data = []
            
            for system in selected_systems:
                system_data = filtered_data[filtered_data["System"] == system]
                
                if len(system_data) > 1:
                    scores = system_data["Average Score"].values
                    variation_data.append({
                        "System": system,
                        "Min Score": min(scores),
                        "Max Score": max(scores),
                        "Variance": np.var(scores),
                        "Range": max(scores) - min(scores),
                        "Best Query Type": system_data.loc[system_data["Average Score"].idxmax()]["Query Type"],
                        "Worst Query Type": system_data.loc[system_data["Average Score"].idxmin()]["Query Type"]
                    })
            
            if variation_data:
                df_variation = pd.DataFrame(variation_data)
                
                # Create range plot
                fig = go.Figure()
                
                for i, row in df_variation.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row["System"], row["System"]],
                        y=[row["Min Score"], row["Max Score"]],
                        mode="lines",
                        line=dict(width=4),
                        name=row["System"]
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[row["System"]],
                        y=[row["Min Score"]],
                        mode="markers",
                        marker=dict(size=10, color="red"),
                        name=f"{row['System']} Min",
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[row["System"]],
                        y=[row["Max Score"]],
                        mode="markers",
                        marker=dict(size=10, color="green"),
                        name=f"{row['System']} Max",
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Performance Range by System",
                    xaxis_title="System",
                    yaxis_title="Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.subheader("Variation Statistics")
                
                # Format values
                df_variation["Min Score"] = df_variation["Min Score"].map("{:.4f}".format)
                df_variation["Max Score"] = df_variation["Max Score"].map("{:.4f}".format)
                df_variation["Variance"] = df_variation["Variance"].map("{:.4f}".format)
                df_variation["Range"] = df_variation["Range"].map("{:.4f}".format)
                
                st.table(df_variation)
            else:
                st.warning("Insufficient data for variation analysis")
        else:
            st.warning("No data available for the selected systems")
    
    def _render_complexity_analysis_page(self) -> None:
        """Render the complexity analysis page."""
        st.header("Complexity Analysis")
        
        if "complexity" not in self.comparison_data:
            st.warning("No complexity analysis data available")
            return
        
        df_complexity = self.comparison_data["complexity"]
        
        # Overview
        st.subheader("Performance by Complexity Level")
        
        # Get unique systems and complexity levels
        systems = sorted(df_complexity["System"].unique())
        complexity_levels = sorted(df_complexity["Complexity"].unique())
        
        # Allow selecting systems
        selected_systems = st.multiselect(
            "Select systems",
            systems,
            default=systems
        )
        
        # Filter by selected systems
        filtered_data = df_complexity[df_complexity["System"].isin(selected_systems)]
        
        if not filtered_data.empty:
            # Create bar chart
            fig = px.bar(
                filtered_data,
                x="Complexity",
                y="Average Score",
                color="System",
                barmode="group",
                title="Average Score by Complexity Level",
                labels={"Average Score": "Score", "Complexity": "Complexity Level"},
                height=500
            )
            
            # Add count as text
            for system in selected_systems:
                system_data = filtered_data[filtered_data["System"] == system]
                for i, row in system_data.iterrows():
                    fig.add_annotation(
                        x=row["Complexity"],
                        y=row["Average Score"],
                        text=f"n={row['Count']}",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10)
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create line chart to show trends
            fig = px.line(
                filtered_data,
                x="Complexity",
                y="Average Score",
                color="System",
                markers=True,
                title="Performance Trend by Complexity Level",
                labels={"Average Score": "Score", "Complexity": "Complexity Level"},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected systems")
        
        # Performance degradation analysis
        st.subheader("Performance Degradation Analysis")
        
        if not filtered_data.empty and len(selected_systems) > 0:
            # Calculate performance degradation with increasing complexity
            degradation_data = []
            
            for system in selected_systems:
                system_data = filtered_data[filtered_data["System"] == system]
                
                # Sort by complexity level (assuming L1, L2, L3)
                if "L1" in system_data["Complexity"].values and "L3" in system_data["Complexity"].values:
                    l1_score = system_data[system_data["Complexity"] == "L1"]["Average Score"].values[0]
                    l3_score = system_data[system_data["Complexity"] == "L3"]["Average Score"].values[0]
                    
                    degradation = l1_score - l3_score
                    degradation_percent = (degradation / l1_score) * 100 if l1_score > 0 else 0
                    
                    degradation_data.append({
                        "System": system,
                        "L1 Score": l1_score,
                        "L3 Score": l3_score,
                        "Absolute Degradation": degradation,
                        "Relative Degradation (%)": degradation_percent
                    })
            
            if degradation_data:
                df_degradation = pd.DataFrame(degradation_data)
                
                # Sort by degradation
                df_degradation = df_degradation.sort_values("Relative Degradation (%)", ascending=False)
                
                # Create bar chart for degradation
                fig = px.bar(
                    df_degradation,
                    x="System",
                    y="Relative Degradation (%)",
                    title="Performance Degradation (L1 to L3)",
                    labels={"Relative Degradation (%)": "Degradation (%)", "System": "System"},
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.subheader("Degradation Statistics")
                
                # Format values
                df_degradation["L1 Score"] = df_degradation["L1 Score"].map("{:.4f}".format)
                df_degradation["L3 Score"] = df_degradation["L3 Score"].map("{:.4f}".format)
                df_degradation["Absolute Degradation"] = df_degradation["Absolute Degradation"].map("{:.4f}".format)
                df_degradation["Relative Degradation (%)"] = df_degradation["Relative Degradation (%)"].map("{:.2f}%".format)
                
                st.table(df_degradation)
            else:
                st.warning("Insufficient data for degradation analysis")
        else:
            st.warning("No data available for the selected systems")
    
    def _render_latency_analysis_page(self) -> None:
        """Render the latency analysis page."""
        st.header("Latency Analysis")
        
        if "latency" not in self.comparison_data:
            st.warning("No latency analysis data available")
            return
        
        df_latency = self.comparison_data["latency"]
        
        # Overview
        st.subheader("Latency Comparison")
        
        if not df_latency.empty:
            # Create bar chart for average latency
            fig = px.bar(
                df_latency,
                x="System",
                y="Avg Latency",
                title="Average Response Latency",
                labels={"Avg Latency": "Seconds", "System": "System"},
                height=400
            )
            
            # Add count as text
            for i, row in df_latency.iterrows():
                fig.add_annotation(
                    x=row["System"],
                    y=row["Avg Latency"],
                    text=f"n={row['Count']}",
                    showarrow=False,
                    yshift=10,
                    font=dict(size=10)
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create range visualization
            fig = go.Figure()
            
            for i, row in df_latency.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["System"], row["System"]],
                    y=[row["Min Latency"], row["Max Latency"]],
                    mode="lines",
                    line=dict(width=4),
                    name=row["System"]
                ))
                
                fig.add_trace(go.Scatter(
                    x=[row["System"]],
                    y=[row["Min Latency"]],
                    mode="markers",
                    marker=dict(size=10, color="green"),
                    name=f"{row['System']} Min",
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[row["System"]],
                    y=[row["Median Latency"]],
                    mode="markers",
                    marker=dict(size=10, color="blue"),
                    name=f"{row['System']} Median",
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=[row["System"]],
                    y=[row["Max Latency"]],
                    mode="markers",
                    marker=dict(size=10, color="red"),
                    name=f"{row['System']} Max",
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Latency Range by System",
                xaxis_title="System",
                yaxis_title="Seconds",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed statistics
            st.subheader("Latency Statistics")
            
            # Format values
            df_display = df_latency.copy()
            df_display["Min Latency"] = df_display["Min Latency"].map("{:.2f} s".format)
            df_display["Max Latency"] = df_display["Max Latency"].map("{:.2f} s".format)
            df_display["Avg Latency"] = df_display["Avg Latency"].map("{:.2f} s".format)
            df_display["Median Latency"] = df_display["Median Latency"].map("{:.2f} s".format)
            
            st.table(df_display)
        else:
            st.warning("No latency data available")
    
    def _render_example_responses_page(self) -> None:
        """Render the example responses page."""
        st.header("Example Responses")
        
        # Get systems with responses
        systems_with_responses = []
        for system_name, results in self.results.items():
            if "responses" in results and results["responses"]:
                systems_with_responses.append(system_name)
        
        if not systems_with_responses:
            st.warning("No systems with responses found")
            return
        
        # System selection
        selected_system = st.selectbox(
            "Select a system",
            systems_with_responses
        )
        
        # Get responses for selected system
        responses = self.results[selected_system].get("responses", [])
        successful_responses = [r for r in responses if r.get("success", False)]
        
        if not successful_responses:
            st.warning(f"No successful responses found for system: {selected_system}")
            return
        
        # Filtering options
        st.subheader("Filter Options")
        
        # Get all query types
        query_types = set()
        for response in successful_responses:
            query_type = response.get("question", {}).get("metadata", {}).get("query_type", "unknown")
            query_types.add(query_type)
        
        # Allow selecting query types
        selected_query_types = st.multiselect(
            "Select query types",
            sorted(query_types),
            default=sorted(query_types)
        )
        
        # Filter by query type
        filtered_responses = [r for r in successful_responses 
                            if r.get("question", {}).get("metadata", {}).get("query_type", "unknown") 
                            in selected_query_types]
        
        # Allow searching by question text
        search_query = st.text_input("Search in questions")
        
        if search_query:
            filtered_responses = [r for r in filtered_responses 
                               if search_query.lower() in r.get("question", {}).get("question", "").lower()]
        
        # Display example count
        st.write(f"Showing {len(filtered_responses)} of {len(successful_responses)} examples")
        
        # Display examples
        if filtered_responses:
            # Allow sorting
            sort_option = st.radio(
                "Sort by",
                ["Default", "Latency (Fastest First)", "Latency (Slowest First)"]
            )
            
            if sort_option == "Latency (Fastest First)":
                filtered_responses = sorted(filtered_responses, key=lambda r: r.get("latency", 0))
            elif sort_option == "Latency (Slowest First)":
                filtered_responses = sorted(filtered_responses, key=lambda r: r.get("latency", 0), reverse=True)
            
            # Limit examples to display
            max_examples = st.slider("Maximum examples to display", 1, 50, 10)
            display_responses = filtered_responses[:max_examples]
            
            # Display examples
            for i, response in enumerate(display_responses):
                with st.expander(f"Example {i+1}: {response.get('question', {}).get('question', '')[:100]}..."):
                    # Question details
                    st.write("**Question:**")
                    st.write(response.get("question", {}).get("question", ""))
                    
                    # Query type and complexity
                    metadata = response.get("question", {}).get("metadata", {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Query Type:** {metadata.get('query_type', 'unknown')}")
                    
                    with col2:
                        st.write(f"**Complexity:** {metadata.get('complexity', 'unknown')}")
                    
                    with col3:
                        st.write(f"**Latency:** {response.get('latency', 0):.2f} s")
                    
                    # Answer
                    st.write("**System Answer:**")
                    st.write(response.get("answer", ""))
                    
                    # Gold answer
                    st.write("**Gold Answer:**")
                    st.write(response.get("question", {}).get("gold_answer", {}).get("text", ""))
                    
                    # Show retrieved contexts if available
                    if "contexts" in response:
                        st.write("**Retrieved Contexts:**")
                        for j, ctx in enumerate(response["contexts"]):
                            st.write(f"*Context {j+1}:*")
                            st.write(ctx.get("text", ""))
        else:
            st.info("No examples match your filters")


if __name__ == "__main__":
    # Create and run the dashboard
    dashboard = RAGVisualizationDashboard()
    dashboard.run_dashboard()