#!/usr/bin/env python3
"""
Question generator for academic papers.
Generates various types of questions based on processed paper content.
"""

import os
import re
import json
import logging
import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import spacy
import pandas as pd
from collections import defaultdict
import string
import nltk
from nltk.tokenize import sent_tokenize
import hashlib

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logger.warning("Downloading spaCy model en_core_web_lg...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


class QuestionGenerator:
    """Generator for creating diverse questions from academic papers."""
    
    def __init__(self, 
                 taxonomy_file: Optional[str] = None,
                 distribution_file: Optional[str] = None,
                 templates_dir: Optional[str] = None,
                 output_dir: str = 'question_sets/generated_questions'):
        """
        Initialize the question generator.
        
        Args:
            taxonomy_file: Path to question taxonomy file
            distribution_file: Path to question distribution config
            templates_dir: Path to question templates directory
            output_dir: Directory to save generated questions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load taxonomy and distribution if provided
        self.taxonomy = self._load_taxonomy(taxonomy_file)
        self.distribution = self._load_distribution(distribution_file)
        
        # Load templates
        self.templates_dir = Path(templates_dir) if templates_dir else Path('question_sets/taxonomy/templates')
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.templates = self._load_templates()
        
        # Track generated questions to avoid duplicates
        self.generated_questions = set()
    
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
    
    def _load_distribution(self, distribution_file: Optional[str]) -> Dict[str, Any]:
        """
        Load question distribution configuration.
        
        Args:
            distribution_file: Path to distribution file
            
        Returns:
            Dictionary with distribution information
        """
        default_distribution = {
            "query_types": {
                "factoid": 0.2,
                "definitional": 0.15,
                "procedural": 0.15,
                "comparative": 0.15,
                "causal": 0.1,
                "quantitative": 0.1,
                "open_ended": 0.15
            },
            "complexity_levels": {
                "L1": 0.3,
                "L2": 0.5,
                "L3": 0.2
            },
            "special_categories": {
                "ambiguous": 0.1,
                "temporal": 0.15,
                "impossible": 0.05,
                "multi_part": 0.15
            }
        }
        
        if distribution_file and os.path.exists(distribution_file):
            try:
                if distribution_file.endswith(('.yaml', '.yml')):
                    with open(distribution_file, 'r') as f:
                        return yaml.safe_load(f)
                else:
                    with open(distribution_file, 'r') as f:
                        return json.load(f)
            except Exception as e:
                logger.error(f"Error loading distribution file: {e}")
                return default_distribution
        
        return default_distribution
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """
        Load question templates from template directory.
        
        Returns:
            Dictionary of templates by question type
        """
        templates = {}
        
        # Check if template directory exists
        if not self.templates_dir.exists():
            logger.warning(f"Template directory {self.templates_dir} not found. Using default templates.")
            return self._get_default_templates()
        
        # Load templates from files
        for template_file in self.templates_dir.glob('*.txt'):
            query_type = template_file.stem
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template_lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if template_lines:
                templates[query_type] = template_lines
        
        # If no templates found, use defaults
        if not templates:
            logger.warning("No template files found. Using default templates.")
            return self._get_default_templates()
        
        return templates
    
    def _get_default_templates(self) -> Dict[str, List[str]]:
        """
        Get default question templates.
        
        Returns:
            Dictionary of default templates by question type
        """
        return {
            "factoid": [
                "What is {entity}?",
                "Who developed {concept}?",
                "When was {concept} first introduced?",
                "Where is {entity} typically used?",
                "How many {entities} are mentioned in the paper?",
                "Which {category} is used in {context}?",
            ],
            "definitional": [
                "What does {concept} mean?",
                "How is {concept} defined in the context of {domain}?",
                "What are the key characteristics of {concept}?",
                "What is meant by {term} in this paper?",
                "How would you explain {concept} to someone new to the field?",
                "What constitutes a {concept}?",
            ],
            "procedural": [
                "How does {process} work?",
                "What steps are involved in {process}?",
                "How is {concept} implemented?",
                "What is the procedure for {action}?",
                "How can one {action}?",
                "What methodology is used for {process}?",
                "How do the authors perform {process}?",
            ],
            "comparative": [
                "What is the difference between {concept1} and {concept2}?",
                "How does {method1} compare to {method2}?",
                "What are the advantages of {concept} over {alternative}?",
                "How do the results of {experiment1} differ from {experiment2}?",
                "What are the trade-offs between {approach1} and {approach2}?",
                "Which performs better, {method1} or {method2}, and why?",
            ],
            "causal": [
                "Why does {phenomenon} occur?",
                "What causes {effect}?",
                "What are the factors that lead to {outcome}?",
                "How does {factor} affect {outcome}?",
                "What is the relationship between {variable1} and {variable2}?",
                "Why is {concept} important for {application}?",
                "What are the implications of {finding}?",
            ],
            "quantitative": [
                "What is the {metric} of {method} reported in the paper?",
                "How much {improvement} was achieved by {method}?",
                "What percentage of {category} showed {characteristic}?",
                "What is the statistical significance of {finding}?",
                "How does the {metric} vary with changes in {parameter}?",
                "What was the sample size used in {experiment}?",
            ],
            "open_ended": [
                "What are the main contributions of this paper?",
                "What future research directions are suggested?",
                "What are the limitations of the approach presented?",
                "How might the findings be applied to {domain}?",
                "What ethical considerations are relevant to this research?",
                "How does this work advance the field of {domain}?",
                "What alternative approaches could address the same problem?",
            ],
        }
    
    def generate_questions_from_paper(self, paper_data: Dict[str, Any], 
                                      num_questions: int = 20,
                                      paper_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate questions from a processed academic paper.
        
        Args:
            paper_data: Processed paper data
            num_questions: Number of questions to generate
            paper_id: Paper identifier
            
        Returns:
            List of generated question dictionaries
        """
        if paper_id is None and 'paper_id' in paper_data:
            paper_id = paper_data['paper_id']
        elif paper_id is None:
            paper_id = hashlib.md5(json.dumps(paper_data["metadata"]).encode()).hexdigest()[:8]
        
        logger.info(f"Generating questions for paper: {paper_id}")
        
        # Extract key information for question generation
        paper_info = self._extract_question_components(paper_data)
        
        # Determine how many questions of each type to generate
        type_counts = self._calculate_type_counts(num_questions)
        
        # Generate questions
        questions = []
        for q_type, count in type_counts.items():
            for _ in range(count):
                question = self._generate_question_of_type(q_type, paper_info, paper_data)
                if question:
                    questions.append(question)
        
        # Ensure we have the requested number of questions (or as close as possible)
        while len(questions) < num_questions:
            # Pick a random type weighted by distribution
            q_type = random.choices(
                list(self.distribution["query_types"].keys()),
                weights=list(self.distribution["query_types"].values())
            )[0]
            
            question = self._generate_question_of_type(q_type, paper_info, paper_data)
            if question:
                questions.append(question)
        
        # Trim to exact count if we generated extras
        questions = questions[:num_questions]
        
        # Save questions to file
        output_file = self.output_dir / f"{paper_id}_questions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(questions)} questions for paper {paper_id}")
        return questions
    
    def _extract_question_components(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract components from paper that can be used for question generation.
        
        Args:
            paper_data: Processed paper data
            
        Returns:
            Dictionary with extracted components
        """
        paper_info = {
            "title": paper_data["metadata"].get("title", ""),
            "authors": paper_data["metadata"].get("authors", []),
            "concepts": [c["text"] for c in paper_data.get("concepts", [])],
            "entities": [],
            "methods": [],
            "results": [],
            "metrics": [],
            "processes": [],
            "key_sentences": [],
            "sections": {},
            "figures": paper_data.get("figures", []),
            "tables": paper_data.get("tables", []),
            "references": paper_data.get("references", [])
        }
        
        # Collect entities by type
        for concept in paper_data.get("concepts", []):
            if concept.get("type") == "entity":
                paper_info["entities"].append(concept["text"])
        
        # Extract key sections
        for section in paper_data.get("sections", []):
            heading = section["heading"].lower()
            paper_info["sections"][heading] = section["content"]
            
            # Extract key sentences from each section
            sentences = sent_tokenize(section["content"])
            
            # Process content for different section types
            if any(term in heading for term in ["method", "approach", "implementation"]):
                # Extract method names and processes
                doc = nlp(section["content"])
                for chunk in doc.noun_chunks:
                    if any(term in chunk.text.lower() for term in ["method", "approach", "algorithm", "technique"]):
                        paper_info["methods"].append(chunk.text)
                
                # Extract processes (verb phrases)
                for sent in sentences:
                    if any(term in sent.lower() for term in ["we propose", "we present", "we introduce", "we develop"]):
                        paper_info["processes"].append(sent)
            
            elif any(term in heading for term in ["result", "evaluation", "experiment"]):
                # Extract numerical results and metrics
                for sent in sentences:
                    if re.search(r'\d+\.?\d*%|\d+\.?\d*\s*(?:accuracy|precision|recall|f1|score)', sent.lower()):
                        paper_info["results"].append(sent)
                    
                    # Find metrics
                    metric_match = re.search(r'(accuracy|precision|recall|f1(?:-score)?|BLEU|ROUGE|MAP|MRR|NDCG)', sent, re.IGNORECASE)
                    if metric_match:
                        paper_info["metrics"].append(metric_match.group(1))
            
            # Extract key sentences (ones that look important)
            important_patterns = [
                r'we (?:show|demonstrate|find|conclude|argue|propose)',
                r'results (?:show|demonstrate|indicate|suggest)',
                r'this paper (?:presents|introduces|proposes|describes)',
                r'our (?:main|key|important) contribution',
                r'significantly (?:better|worse|higher|lower)',
                r'novel',
                r'state[- ]of[- ]the[- ]art',
                r'outperforms',
                r'advantage',
                r'limitation'
            ]
            
            for sent in sentences:
                if any(re.search(pattern, sent, re.IGNORECASE) for pattern in important_patterns):
                    paper_info["key_sentences"].append(sent)
        
        # Ensure we have unique lists
        for key in ["concepts", "entities", "methods", "metrics", "processes", "key_sentences"]:
            paper_info[key] = list(set(paper_info[key]))
        
        return paper_info
    
    def _calculate_type_counts(self, total: int) -> Dict[str, int]:
        """
        Calculate how many questions of each type to generate.
        
        Args:
            total: Total number of questions to generate
            
        Returns:
            Dictionary with count for each question type
        """
        type_counts = {}
        remaining = total
        
        # Allocate based on distribution
        for q_type, proportion in self.distribution["query_types"].items():
            count = round(total * proportion)
            type_counts[q_type] = count
            remaining -= count
        
        # Adjust for rounding errors
        if remaining > 0:
            # Distribute remaining among types
            for q_type in sorted(type_counts.keys()):
                type_counts[q_type] += 1
                remaining -= 1
                if remaining == 0:
                    break
        elif remaining < 0:
            # Remove from types with highest counts
            for q_type in sorted(type_counts.keys(), key=lambda k: type_counts[k], reverse=True):
                type_counts[q_type] -= 1
                remaining += 1
                if remaining == 0:
                    break
        
        return type_counts
    
    def _generate_question_of_type(self, q_type: str, paper_info: Dict[str, Any], 
                                  paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a question of the specified type.
        
        Args:
            q_type: Question type
            paper_info: Extracted paper information
            paper_data: Full paper data
            
        Returns:
            Generated question dictionary or None if generation failed
        """
        # Get templates for this question type
        templates = self.templates.get(q_type, [])
        if not templates:
            logger.warning(f"No templates found for question type: {q_type}")
            return None
        
        # Try different templates until we get a valid question (max 5 attempts)
        for _ in range(5):
            template = random.choice(templates)
            
            # Fill in the template
            question_text, template_slots = self._fill_template(template, paper_info, paper_data)
            
            # Check if the question is valid and not a duplicate
            if question_text and self._is_valid_question(question_text):
                # Choose a complexity level
                complexity = self._assign_complexity_level(q_type, template_slots)
                
                # Generate answer
                answer, answer_contexts = self._generate_answer(question_text, q_type, template_slots, paper_data)
                
                # Create question object
                question_obj = {
                    "question_id": self._generate_question_id(question_text),
                    "question": question_text,
                    "answer": answer,
                    "paper_id": paper_data.get("paper_id", ""),
                    "metadata": {
                        "query_type": q_type,
                        "complexity": complexity,
                        "special_categories": self._assign_special_categories(question_text, q_type),
                        "template": template,
                        "template_slots": template_slots
                    },
                    "contexts": answer_contexts
                }
                
                self.generated_questions.add(question_text)
                return question_obj
        
        logger.warning(f"Failed to generate valid question of type {q_type} after multiple attempts")
        return None
    
    def _fill_template(self, template: str, paper_info: Dict[str, Any], 
                      paper_data: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        """
        Fill in a question template with paper-specific information.
        
        Args:
            template: Question template string
            paper_info: Extracted paper information
            paper_data: Full paper data
            
        Returns:
            Tuple of (filled question text, dictionary of filled slots)
        """
        # Extract slots from template
        slots = re.findall(r'\{([^}]+)\}', template)
        filled_slots = {}
        
        # Try to fill all slots
        for slot in slots:
            # Handle special slot types
            if slot == "concept" and paper_info["concepts"]:
                value = random.choice(paper_info["concepts"])
            elif slot == "entity" and paper_info["entities"]:
                value = random.choice(paper_info["entities"])
            elif slot == "method" and paper_info["methods"]:
                value = random.choice(paper_info["methods"])
            elif slot == "process" and paper_info["processes"]:
                # Extract just the key part from the sentence
                sent = random.choice(paper_info["processes"])
                # Try to extract the process name
                value = self._extract_process_from_sentence(sent)
            elif slot == "metric" and paper_info["metrics"]:
                value = random.choice(paper_info["metrics"])
            elif slot == "domain":
                # Try to infer domain from title or abstract
                domain = self._infer_domain(paper_data)
                value = domain if domain else "this field"
            elif slot.startswith("concept") and slot.endswith(("1", "2")) and len(paper_info["concepts"]) >= 2:
                # For comparative questions with two concepts
                concepts = random.sample(paper_info["concepts"], 2)
                if slot.endswith("1"):
                    value = concepts[0]
                else:
                    value = concepts[1]
            elif slot.startswith("method") and slot.endswith(("1", "2")) and len(paper_info["methods"]) >= 2:
                # For comparative questions with two methods
                methods = random.sample(paper_info["methods"], 2)
                if slot.endswith("1"):
                    value = methods[0]
                else:
                    value = methods[1]
            elif slot == "finding" and paper_info["key_sentences"]:
                sent = random.choice(paper_info["key_sentences"])
                # Extract the key finding
                value = self._extract_finding_from_sentence(sent)
            elif slot == "experiment" and "experiment" in paper_info["sections"]:
                value = "the experiment described in the paper"
            elif slot.startswith("experiment") and slot.endswith(("1", "2")):
                value = f"experiment {slot[-1]}"
            elif slot == "improvement" and paper_info["results"]:
                # Look for improvement mentioned in results
                improvements = [s for s in paper_info["results"] if "improv" in s.lower()]
                if improvements:
                    value = self._extract_improvement_from_sentence(random.choice(improvements))
                else:
                    value = "improvement"
            elif slot == "category":
                # Generic categories from the paper domain
                categories = ["approach", "method", "technique", "model", "algorithm"]
                value = random.choice(categories)
            elif slot == "context":
                contexts = ["this research", "this study", "this paper", "the described framework"]
                value = random.choice(contexts)
            elif slot == "action" and paper_info["processes"]:
                # Extract action verbs from processes
                sent = random.choice(paper_info["processes"])
                doc = nlp(sent)
                verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
                if verbs:
                    value = random.choice(verbs) + " " + self._extract_object_for_verb(sent, verbs[0])
                else:
                    value = "perform this process"
            else:
                # Default fallback values for slots
                fallbacks = {
                    "concept": "this concept",
                    "entity": "this component",
                    "method": "this method",
                    "process": "this process",
                    "metric": "performance",
                    "approach1": "the first approach",
                    "approach2": "the second approach",
                    "variable1": "the independent variable",
                    "variable2": "the dependent variable",
                    "phenomenon": "this phenomenon",
                    "effect": "this effect",
                    "outcome": "the observed outcome",
                    "factor": "this factor",
                    "application": "practical applications",
                    "parameter": "the parameter",
                    "term": "this term",
                }
                value = fallbacks.get(slot, f"[{slot}]")
            
            filled_slots[slot] = value
        
        # Apply the slot values to the template
        question_text = template
        for slot, value in filled_slots.items():
            question_text = question_text.replace(f"{{{slot}}}", value)
        
        return question_text, filled_slots
    
    def _extract_process_from_sentence(self, sentence: str) -> str:
        """
        Extract a process description from a sentence.
        
        Args:
            sentence: Sentence containing process description
            
        Returns:
            Extracted process text
        """
        # Look for common patterns where processes are mentioned
        patterns = [
            r'we propose ([\w\s]+)',
            r'we present ([\w\s]+)',
            r'we introduce ([\w\s]+)',
            r'we develop ([\w\s]+)',
            r'using ([\w\s]+) to',
            r'the process of ([\w\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: Use dependency parsing
        doc = nlp(sentence)
        for token in doc:
            if token.dep_ == "dobj" and token.head.pos_ == "VERB":
                verb = token.head.text
                obj = token.text
                
                # Get the full noun phrase
                noun_phrase = ""
                for chunk in doc.noun_chunks:
                    if obj in chunk.text:
                        noun_phrase = chunk.text
                        break
                
                if noun_phrase:
                    return f"{verb} {noun_phrase}"
                else:
                    return f"{verb} {obj}"
        
        # Further fallback
        return "the described process"
    
    def _extract_finding_from_sentence(self, sentence: str) -> str:
        """
        Extract a key finding from a sentence.
        
        Args:
            sentence: Sentence containing a finding
            
        Returns:
            Extracted finding text
        """
        # Look for common patterns where findings are mentioned
        patterns = [
            r'we (?:find|show|demonstrate) that ([\w\s,]+)',
            r'results (?:show|demonstrate|indicate|suggest) that ([\w\s,]+)',
            r'our (?:results|findings) ([\w\s,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback
        return sentence
    
    def _extract_improvement_from_sentence(self, sentence: str) -> str:
        """
        Extract improvement description from a sentence.
        
        Args:
            sentence: Sentence containing improvement description
            
        Returns:
            Extracted improvement text
        """
        # Look for numerical improvements
        num_pattern = r'(\d+\.?\d*%|\d+\.?\d*) (?:improvement|increase|decrease|reduction)'
        match = re.search(num_pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0)
        
        # Look for qualitative improvements
        qual_pattern = r'(significant|substantial|marginal|modest) (?:improvement|increase|decrease|reduction)'
        match = re.search(qual_pattern, sentence, re.IGNORECASE)
        if match:
            return match.group(0)
        
        # Fallback
        return "improvement"
    
    def _extract_object_for_verb(self, sentence: str, verb: str) -> str:
        """
        Extract the object associated with a verb in a sentence.
        
        Args:
            sentence: Sentence containing the verb
            verb: Verb to find object for
            
        Returns:
            Object associated with the verb
        """
        doc = nlp(sentence)
        
        # Find the verb token
        verb_token = None
        for token in doc:
            if token.lemma_ == verb:
                verb_token = token
                break
        
        if not verb_token:
            return ""
        
        # Look for direct object
        for token in doc:
            if token.head == verb_token and token.dep_ == "dobj":
                # Return the full noun phrase if possible
                for chunk in doc.noun_chunks:
                    if token.text in chunk.text:
                        return chunk.text
                return token.text
        
        # Fallback: look for prepositional object
        for token in doc:
            if token.head == verb_token and token.dep_ == "pobj":
                # Return the full noun phrase if possible
                for chunk in doc.noun_chunks:
                    if token.text in chunk.text:
                        return chunk.text
                return token.text
        
        return ""
    
    def _infer_domain(self, paper_data: Dict[str, Any]) -> str:
        """
        Infer the domain of the paper.
        
        Args:
            paper_data: Processed paper data
            
        Returns:
            Inferred domain
        """
        # Common academic domains
        domains = [
            "machine learning", "artificial intelligence", "natural language processing",
            "computer vision", "data mining", "information retrieval", "robotics",
            "computer graphics", "human-computer interaction", "cybersecurity",
            "bioinformatics", "computational biology", "neuroscience", "psychology",
            "cognitive science", "physics", "chemistry", "mathematics", "statistics",
            "economics", "finance", "political science", "sociology", "medicine",
            "materials science", "engineering", "education", "linguistics"
        ]
        
        # Look for domain mentions in title and abstract
        text = paper_data["metadata"].get("title", "") + " " + paper_data["metadata"].get("abstract", "")
        
        for domain in domains:
            if domain.lower() in text.lower():
                return domain
        
        # Look in the introduction if available
        for section in paper_data.get("sections", []):
            if any(term in section["heading"].lower() for term in ["introduction", "background"]):
                for domain in domains:
                    if domain.lower() in section["content"].lower():
                        return domain
        
        # Fallback: look at most frequent concepts
        concepts = paper_data.get("concepts", [])
        if concepts:
            # Sort by count and check top concepts
            sorted_concepts = sorted(concepts, key=lambda x: x.get("count", 0), reverse=True)
            for concept in sorted_concepts[:5]:
                concept_text = concept.get("text", "").lower()
                for domain in domains:
                    if domain.lower() in concept_text:
                        return domain
        
        # Couldn't determine domain
        return ""
    
    def _is_valid_question(self, question_text: str) -> bool:
        """
        Check if a question is valid and not a duplicate.
        
        Args:
            question_text: Question text
            
        Returns:
            True if question is valid
        """
        # Check if it's a duplicate
        if question_text in self.generated_questions:
            return False
        
        # Check if it contains unfilled slots
        if re.search(r'\[\w+\]', question_text):
            return False
        
        # Check if it's too short
        if len(question_text) < 10:
            return False
        
        # Check if it ends with a question mark
        if not question_text.endswith('?'):
            return False
        
        return True
    
    def _assign_complexity_level(self, q_type: str, template_slots: Dict[str, str]) -> str:
        """
        Assign a complexity level to the question.
        
        Args:
            q_type: Question type
            template_slots: Filled template slots
            
        Returns:
            Complexity level (L1, L2, or L3)
        """
        # Assign based on question type and complexity
        if q_type in ["factoid", "definitional"]:
            return random.choices(
                ["L1", "L2", "L3"],
                weights=[0.6, 0.3, 0.1]
            )[0]
        elif q_type in ["procedural", "quantitative"]:
            return random.choices(
                ["L1", "L2", "L3"],
                weights=[0.3, 0.5, 0.2]
            )[0]
        elif q_type in ["comparative", "causal"]:
            return random.choices(
                ["L1", "L2", "L3"],
                weights=[0.2, 0.5, 0.3]
            )[0]
        elif q_type == "open_ended":
            return random.choices(
                ["L1", "L2", "L3"],
                weights=[0.1, 0.4, 0.5]
            )[0]
        
        # Default
        return random.choices(
            ["L1", "L2", "L3"],
            weights=[0.3, 0.5, 0.2]
        )[0]
    
    def _assign_special_categories(self, question_text: str, q_type: str) -> List[str]:
        """
        Assign special categories to the question.
        
        Args:
            question_text: Question text
            q_type: Question type
            
        Returns:
            List of special categories
        """
        categories = []
        
        # Check for multi-part questions
        if "and" in question_text and ("what" in question_text.lower() or "how" in question_text.lower()):
            if question_text.lower().count("what") > 1 or question_text.lower().count("how") > 1:
                categories.append("multi_part")
        
        # Check for temporal questions
        temporal_indicators = ["when", "before", "after", "during", "year", "time", "period", "history", "evolution"]
        if any(indicator in question_text.lower() for indicator in temporal_indicators):
            categories.append("temporal")
        
        # Check for ambiguous questions (more subjective)
        ambiguous_indicators = ["could", "might", "may", "possible", "potential", "future", "consider"]
        if any(indicator in question_text.lower() for indicator in ambiguous_indicators):
            if random.random() < 0.7:  # 70% chance to mark as ambiguous
                categories.append("ambiguous")
        
        # Occasionally mark as impossible (5% chance)
        if random.random() < 0.05:
            categories.append("impossible")
        
        return categories
    
    def _generate_answer(self, question_text: str, q_type: str, template_slots: Dict[str, str], 
                        paper_data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate an answer for the question.
        
        Args:
            question_text: Question text
            q_type: Question type
            template_slots: Filled template slots
            paper_data: Full paper data
            
        Returns:
            Tuple of (answer text, list of context passages)
        """
        # Placeholder - in a real implementation this would use more sophisticated
        # techniques to generate proper answers from the paper content
        answer = "This is a placeholder answer. In a full implementation, this would extract relevant information from the paper."
        
        # Find sections that might contain the answer
        contexts = []
        
        # Look for context based on question type and template slots
        if q_type == "factoid":
            # For factoid questions, look for mentions of entities in the question
            for slot, value in template_slots.items():
                for section in paper_data.get("sections", []):
                    if value in section["content"]:
                        contexts.append({
                            "section": section["heading"],
                            "text": self._extract_context_around_term(section["content"], value)
                        })
                        break
        
        elif q_type == "definitional":
            # For definitional questions, look for definitions
            for slot, value in template_slots.items():
                if slot in ["concept", "term"]:
                    definition = self._find_definition(value, paper_data)
                    if definition:
                        contexts.append({
                            "section": "Definition",
                            "text": definition
                        })
        
        elif q_type == "comparative":
            # For comparative questions, look for both concepts
            concept1 = template_slots.get("concept1", "") or template_slots.get("method1", "")
            concept2 = template_slots.get("concept2", "") or template_slots.get("method2", "")
            
            if concept1 and concept2:
                comparison = self._find_comparison(concept1, concept2, paper_data)
                if comparison:
                    contexts.append({
                        "section": "Comparison",
                        "text": comparison
                    })
        
        # If we didn't find specific contexts, add some generic ones
        if not contexts:
            # Add abstract
            if paper_data["metadata"].get("abstract"):
                contexts.append({
                    "section": "Abstract",
                    "text": paper_data["metadata"]["abstract"]
                })
            
            # Add a relevant section based on question type
            relevant_section = None
            if q_type in ["factoid", "definitional"]:
                relevant_section = self._find_relevant_section(["introduction", "background"], paper_data)
            elif q_type == "procedural":
                relevant_section = self._find_relevant_section(["method", "approach", "implementation"], paper_data)
            elif q_type in ["comparative", "causal"]:
                relevant_section = self._find_relevant_section(["discussion", "analysis"], paper_data)
            elif q_type == "quantitative":
                relevant_section = self._find_relevant_section(["result", "evaluation", "experiment"], paper_data)
            elif q_type == "open_ended":
                relevant_section = self._find_relevant_section(["conclusion", "discussion", "future work"], paper_data)
            
            if relevant_section:
                contexts.append({
                    "section": relevant_section["heading"],
                    "text": self._shorten_context(relevant_section["content"])
                })
        
        # For special category "impossible" questions, note that in the answer
        if "impossible" in self._assign_special_categories(question_text, q_type):
            answer = "Based on the information provided in the paper, it is not possible to answer this question definitively."
        
        return answer, contexts
    
    def _extract_context_around_term(self, text: str, term: str, context_size: int = 200) -> str:
        """
        Extract text around a term mention.
        
        Args:
            text: Source text
            term: Term to find
            context_size: Number of characters around the term
            
        Returns:
            Context text
        """
        term_index = text.find(term)
        if term_index == -1:
            # Try case-insensitive search
            term_index = text.lower().find(term.lower())
            if term_index == -1:
                return ""
        
        start = max(0, term_index - context_size)
        end = min(len(text), term_index + len(term) + context_size)
        
        # Adjust to complete sentences if possible
        while start > 0 and text[start] not in ".!?":
            start -= 1
        if start > 0:
            start += 1  # Move past the punctuation
        
        while end < len(text) and text[end] not in ".!?":
            end += 1
        if end < len(text):
            end += 1  # Include the punctuation
        
        return text[start:end].strip()
    
    def _find_definition(self, term: str, paper_data: Dict[str, Any]) -> str:
        """
        Find a definition of a term in the paper.
        
        Args:
            term: Term to find definition for
            paper_data: Processed paper data
            
        Returns:
            Definition text or empty string if not found
        """
        # Look for common definition patterns
        definition_patterns = [
            f"{term} (?:is|are) defined as (.*?)\\.",
            f"{term} refers to (.*?)\\.",
            f"{term} is (.*?)\\.",
            f"{term} are (.*?)\\.",
            f"define (?:the|a|an)? {term} as (.*?)\\.",
            f"call (?:this|it|them) (?:the|a|an)? {term} (.*?)\\."
        ]
        
        for section in paper_data.get("sections", []):
            for pattern in definition_patterns:
                match = re.search(pattern, section["content"], re.IGNORECASE)
                if match:
                    # Return the sentence containing the definition
                    sentences = sent_tokenize(section["content"])
                    for sent in sentences:
                        if match.group(0) in sent:
                            return sent
                    
                    # If sentence not found, return the matched definition
                    return match.group(0)
        
        # Look around first mention of the term
        for section in paper_data.get("sections", []):
            if term.lower() in section["content"].lower():
                # Find the sentence containing the first mention
                term_index = section["content"].lower().find(term.lower())
                if term_index >= 0:
                    start = max(0, term_index - 100)
                    end = min(len(section["content"]), term_index + 200)
                    context = section["content"][start:end]
                    
                    # Find the sentence containing the term
                    sentences = sent_tokenize(context)
                    for sent in sentences:
                        if term.lower() in sent.lower():
                            return sent
        
        return ""
    
    def _find_comparison(self, term1: str, term2: str, paper_data: Dict[str, Any]) -> str:
        """
        Find a comparison between two terms.
        
        Args:
            term1: First term
            term2: Second term
            paper_data: Processed paper data
            
        Returns:
            Comparison text or empty string if not found
        """
        # Look for sentences that mention both terms
        for section in paper_data.get("sections", []):
            if term1.lower() in section["content"].lower() and term2.lower() in section["content"].lower():
                sentences = sent_tokenize(section["content"])
                
                # First priority: sentences containing both terms
                for sent in sentences:
                    if term1.lower() in sent.lower() and term2.lower() in sent.lower():
                        return sent
                
                # Second priority: consecutive sentences mentioning the terms
                for i in range(len(sentences) - 1):
                    if ((term1.lower() in sentences[i].lower() and term2.lower() in sentences[i+1].lower()) or
                        (term2.lower() in sentences[i].lower() and term1.lower() in sentences[i+1].lower())):
                        return f"{sentences[i]} {sentences[i+1]}"
                
                # Third priority: paragraph containing both terms
                term1_idx = section["content"].lower().find(term1.lower())
                term2_idx = section["content"].lower().find(term2.lower())
                
                start_idx = min(term1_idx, term2_idx)
                end_idx = max(term1_idx + len(term1), term2_idx + len(term2))
                
                # Expand to paragraph boundaries
                start = section["content"].rfind('\n', 0, start_idx)
                if start == -1:
                    start = 0
                else:
                    start += 1  # Skip the newline
                
                end = section["content"].find('\n', end_idx)
                if end == -1:
                    end = len(section["content"])
                
                if end - start <= 500:
                    return section["content"][start:end].strip()
                else:
                    # Just return the sentences around both terms
                    context1 = self._extract_context_around_term(section["content"], term1, 100)
                    context2 = self._extract_context_around_term(section["content"], term2, 100)
                    return f"{context1} [...] {context2}"
        
        return ""
    
    def _find_relevant_section(self, section_keywords: List[str], paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a relevant section based on keywords.
        
        Args:
            section_keywords: List of keywords to look for in section headings
            paper_data: Processed paper data
            
        Returns:
            Section dictionary or None if not found
        """
        for section in paper_data.get("sections", []):
            heading = section["heading"].lower()
            if any(keyword in heading for keyword in section_keywords):
                return section
        
        return None
    
    def _shorten_context(self, text: str, max_length: int = 500) -> str:
        """
        Shorten context to a reasonable length.
        
        Args:
            text: Text to shorten
            max_length: Maximum length
            
        Returns:
            Shortened text
        """
        if len(text) <= max_length:
            return text
        
        # Try to find a good breaking point
        sentences = sent_tokenize(text)
        
        result = ""
        for sent in sentences:
            if len(result) + len(sent) <= max_length:
                result += sent + " "
            else:
                break
        
        return result.strip()
    
    def _generate_question_id(self, question_text: str) -> str:
        """
        Generate a unique ID for a question.
        
        Args:
            question_text: Question text
            
        Returns:
            Question ID
        """
        # Create a hash of the question text
        question_hash = hashlib.md5(question_text.encode()).hexdigest()[:12]
        return f"q_{question_hash}"
    
    def process_paper_file(self, paper_file: Union[str, Path], 
                          num_questions: int = 20) -> List[Dict[str, Any]]:
        """
        Process a paper file and generate questions.
        
        Args:
            paper_file: Path to processed paper JSON file
            num_questions: Number of questions to generate
            
        Returns:
            List of generated question dictionaries
        """
        paper_file = Path(paper_file)
        if not paper_file.exists():
            raise FileNotFoundError(f"Paper file not found: {paper_file}")
        
        # Load paper data
        with open(paper_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        return self.generate_questions_from_paper(paper_data, num_questions, paper_id=paper_file.stem)
    
    def save_question_templates(self) -> None:
        """Save the current templates to files."""
        for q_type, templates in self.templates.items():
            output_file = self.templates_dir / f"{q_type}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for template in templates:
                    f.write(f"{template}\n")
        
        logger.info(f"Saved {len(self.templates)} template files to {self.templates_dir}")

    def generate_seed_questions(self, paper_data: Dict[str, Any], 
                              num_per_type: int = 3) -> List[Dict[str, Any]]:
        """
        Generate seed questions that should be manually reviewed and refined.
        
        Args:
            paper_data: Processed paper data
            num_per_type: Number of questions per type
            
        Returns:
            List of generated seed question dictionaries
        """
        seed_questions = []
        paper_id = paper_data.get("paper_id", "unknown")
        
        # Extract key information
        paper_info = self._extract_question_components(paper_data)
        
        # Generate questions for each type
        for q_type in self.taxonomy["query_types"].keys():
            for _ in range(num_per_type):
                question = self._generate_question_of_type(q_type, paper_info, paper_data)
                if question:
                    # Mark as seed question
                    question["metadata"]["is_seed"] = True
                    seed_questions.append(question)
        
        # Save seed questions
        output_dir = Path("question_sets/seed_questions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{paper_id}_seed_questions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(seed_questions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(seed_questions)} seed questions for paper {paper_id}")
        return seed_questions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate questions from academic papers")
    parser.add_argument("input", help="Path to processed paper JSON file or directory")
    parser.add_argument("--taxonomy", help="Path to question taxonomy file")
    parser.add_argument("--distribution", help="Path to question distribution config")
    parser.add_argument("--templates", help="Path to question templates directory")
    parser.add_argument("--output-dir", default="question_sets/generated_questions", 
                       help="Output directory for generated questions")
    parser.add_argument("--num-questions", type=int, default=20,
                       help="Number of questions to generate per paper")
    parser.add_argument("--seed", action="store_true",
                       help="Generate seed questions for manual review")
    parser.add_argument("--num-per-type", type=int, default=3,
                       help="Number of seed questions per type")
    
    args = parser.parse_args()
    
    generator = QuestionGenerator(
        taxonomy_file=args.taxonomy,
        distribution_file=args.distribution,
        templates_dir=args.templates,
        output_dir=args.output_dir
    )
    
    input_path = Path(args.input)
    if input_path.is_file():
        if args.seed:
            # Load paper data
            with open(input_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            generator.generate_seed_questions(paper_data, args.num_per_type)
        else:
            generator.process_paper_file(input_path, args.num_questions)
    elif input_path.is_dir():
        # Process all JSON files in directory
        for paper_file in input_path.glob("*.json"):
            try:
                if args.seed:
                    # Load paper data
                    with open(paper_file, 'r', encoding='utf-8') as f:
                        paper_data = json.load(f)
                    generator.generate_seed_questions(paper_data, args.num_per_type)
                else:
                    generator.process_paper_file(paper_file, args.num_questions)
            except Exception as e:
                logger.error(f"Error processing {paper_file}: {str(e)}")
    else:
        print(f"Error: Input {input_path} is not a valid file or directory")