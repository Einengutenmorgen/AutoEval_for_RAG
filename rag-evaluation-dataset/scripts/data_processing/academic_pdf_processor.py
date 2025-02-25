#!/usr/bin/env python3
"""
Specialized PDF processor for academic papers.
Extracts structured content from academic PDFs including sections,
references, figures, tables, and metadata.
"""
import sys
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import pdfplumber
import spacy
from pdfminer.high_level import extract_text
import pandas as pd
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load spaCy model for NER and concept extraction
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logger.warning("Downloading spaCy model en_core_web_lg...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


class AcademicPaperProcessor:
    """Processor for extracting structured content from academic papers in PDF format."""
    
    # Common section patterns in academic papers
    SECTION_PATTERNS = [
        r'^(Abstract)[\s:]*$',
        r'^(\d+\.?\s+[A-Z][a-zA-Z\s]+)[\s:]*$',  # Numbered sections like "1. Introduction"
        r'^([A-Z][a-zA-Z\s]+)[\s:]*$',  # Capitalized sections like "INTRODUCTION"
        r'^(Introduction|Background|Related Work|Methods?|Methodology|Experiments?|Results|Discussion|Conclusion|References)[\s:]*$'
    ]
    
    # Citation patterns
    CITATION_PATTERNS = [
        r'\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]',  # [1] or [1, 2, 3]
        r'\(\s*([A-Za-z]+(?:\s+et\s+al\.)?(?:,\s+\d{4})?(?:\s*;\s*[A-Za-z]+(?:\s+et\s+al\.)?(?:,\s+\d{4})?)*)\s*\)'  # (Author et al., 2020) or (Author, 2020; Author2, 2021)
    ]
    
    def __init__(self, output_dir: str = 'data/processed'):
        """
        Initialize the academic paper processor.
        
        Args:
            output_dir: Directory to save processed output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_pdf(self, pdf_path: Union[str, Path], paper_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single academic paper PDF.
        
        Args:
            pdf_path: Path to the PDF file
            paper_id: Optional identifier for the paper (derived from filename if None)
            
        Returns:
            Dictionary containing structured paper content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if paper_id is None:
            paper_id = pdf_path.stem
        
        logger.info(f"Processing academic paper: {pdf_path} (ID: {paper_id})")
        
        # Extract structure and content
        try:
            paper_structure = self._extract_paper_structure(pdf_path)
            metadata = self._extract_metadata(pdf_path, paper_structure)
            sections = self._process_sections(paper_structure)
            figures = self._extract_figures(pdf_path)
            tables = self._extract_tables(pdf_path)
            references = self._extract_references(paper_structure)
            citations = self._extract_citations(paper_structure)
            concepts = self._extract_key_concepts(sections)
            
            # Compile paper data
            paper_data = {
                "paper_id": paper_id,
                "metadata": metadata,
                "sections": sections,
                "figures": figures,
                "tables": tables,
                "references": references,
                "citations": citations,
                "concepts": concepts,
                "source_file": str(pdf_path)
            }
            
            # Save processed data
            output_file = self.output_dir / f"{paper_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully processed paper {paper_id}")
            return paper_data
            
        except Exception as e:
            logger.error(f"Error processing paper {pdf_path}: {str(e)}", exc_info=True)
            raise
    
    def _extract_paper_structure(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract the overall structure of the paper.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with paper structure including pages, text, and layout
        """
        structure = {
            "pages": [],
            "full_text": ""
        }
        
        # Extract text from each page with layout information
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    # Get layout information
                    words = page.extract_words()
                    lines = self._group_words_into_lines(words)
                    
                    page_data = {
                        "page_number": i + 1,
                        "text": text,
                        "lines": lines
                    }
                    structure["pages"].append(page_data)
                    structure["full_text"] += text + "\n\n"
        
        return structure
    
    def _group_words_into_lines(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group extracted words into lines based on y-position.
        
        Args:
            words: List of word dictionaries from pdfplumber
            
        Returns:
            List of line dictionaries
        """
        if not words:
            return []
        
        # Sort words by top y-coordinate and then by x-coordinate
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        lines = []
        current_line = {
            "text": sorted_words[0]['text'],
            "words": [sorted_words[0]],
            "top": sorted_words[0]['top'],
            "bottom": sorted_words[0]['bottom'],
            "x0": sorted_words[0]['x0'],
            "x1": sorted_words[0]['x1']
        }
        
        # Group words into lines with a tolerance for y-position
        y_tolerance = 3  # pixels
        
        for word in sorted_words[1:]:
            if abs(word['top'] - current_line['top']) <= y_tolerance:
                # Same line
                current_line['text'] += ' ' + word['text']
                current_line['words'].append(word)
                current_line['x1'] = word['x1']
                current_line['bottom'] = max(current_line['bottom'], word['bottom'])
            else:
                # New line
                lines.append(current_line)
                current_line = {
                    "text": word['text'],
                    "words": [word],
                    "top": word['top'],
                    "bottom": word['bottom'],
                    "x0": word['x0'],
                    "x1": word['x1']
                }
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _extract_metadata(self, pdf_path: Path, paper_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from the paper.
        
        Args:
            pdf_path: Path to the PDF file
            paper_structure: Extracted paper structure
            
        Returns:
            Dictionary with metadata fields
        """
        metadata = {
            "title": "",
            "authors": [],
            "publication_date": None,
            "journal": "",
            "abstract": "",
            "keywords": []
        }
        
        # Try to extract title from the first page
        if paper_structure["pages"]:
            first_page = paper_structure["pages"][0]
            lines = first_page["lines"]
            
            # Title is typically the first non-empty line that's centered
            if lines:
                # Find potential title lines (first few big, centered lines)
                potential_titles = []
                for line in lines[:10]:  # Check first 10 lines
                    line_text = line["text"].strip()
                    if line_text and len(line_text) > 10:
                        potential_titles.append(line_text)
                        if len(potential_titles) >= 3:  # Usually enough to get the title
                            break
                
                if potential_titles:
                    # Use the longest line as the title (heuristic)
                    metadata["title"] = max(potential_titles, key=len)
            
            # Extract abstract
            abstract_text = self._find_abstract(paper_structure)
            if abstract_text:
                metadata["abstract"] = abstract_text
            
            # Try to extract authors (usually after title, before abstract)
            # This is a simplistic approach - you may need more sophisticated methods
            author_line = None
            for line in lines[1:15]:  # Look in first 15 lines
                line_text = line["text"].strip()
                if "abstract" in line_text.lower():
                    break
                
                # Author lines often contain commas, affiliations, or email markers
                if (line_text and "," in line_text) or ("@" in line_text) or ("*" in line_text):
                    author_line = line_text
                    break
            
            if author_line:
                # Simple processing - split by commas or 'and'
                author_line = re.sub(r'\s*,\s*|\s+and\s+|\s*&\s*', ',', author_line)
                # Remove email addresses
                author_line = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '', author_line)
                # Remove affiliations (often in parentheses or superscript numbers)
                author_line = re.sub(r'\(\d+\)|\(\w+\)|\d+|[*†‡§]', '', author_line)
                
                potential_authors = [a.strip() for a in author_line.split(',') if a.strip()]
                metadata["authors"] = [a for a in potential_authors if len(a) > 2]
            
            # Keywords often found after abstract
            keywords_match = re.search(r'Keywords?[:\s]+([\s\S]+?)(?:\n\n|\.[^\w])', paper_structure["full_text"], re.IGNORECASE)
            if keywords_match:
                keywords_text = keywords_match.group(1).strip()
                # Split by commas or semicolons
                keywords = [k.strip() for k in re.split(r'[,;]', keywords_text) if k.strip()]
                metadata["keywords"] = keywords
        
        return metadata
    
    def _find_abstract(self, paper_structure: Dict[str, Any]) -> str:
        """
        Extract the abstract from the paper.
        
        Args:
            paper_structure: Extracted paper structure
            
        Returns:
            Abstract text or empty string if not found
        """
        full_text = paper_structure["full_text"]
        
        # Find the abstract section
        abstract_pattern = r'Abstract[:\s]*\n*([\s\S]+?)(?:\n\n|\n[A-Z][a-z]+\s+\d+\.|\n\d+\.)'
        abstract_match = re.search(abstract_pattern, full_text, re.IGNORECASE)
        
        if abstract_match:
            return abstract_match.group(1).strip()
        
        # Alternative: try to find within first page between "Abstract" and next heading
        if paper_structure["pages"]:
            first_page_text = paper_structure["pages"][0]["text"]
            abstract_idx = first_page_text.lower().find("abstract")
            
            if abstract_idx != -1:
                # Find the next line that looks like a section heading
                lines = first_page_text[abstract_idx:].split('\n')
                abstract_text = ""
                capture = False
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if i == 0 and "abstract" in line.lower():
                        capture = True
                        # Skip the "Abstract" line itself
                        continue
                    
                    # Stop at next section heading or empty line followed by potential heading
                    if capture and (any(re.match(pattern, line) for pattern in self.SECTION_PATTERNS) or 
                                  (not line and i+1 < len(lines) and 
                                   any(re.match(pattern, lines[i+1].strip()) for pattern in self.SECTION_PATTERNS))):
                        break
                    
                    if capture:
                        abstract_text += line + " "
                
                if abstract_text:
                    return abstract_text.strip()
        
        return ""
    
    def _process_sections(self, paper_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process and extract sections from the paper.
        
        Args:
            paper_structure: Extracted paper structure
            
        Returns:
            List of section dictionaries with headings and content
        """
        full_text = paper_structure["full_text"]
        sections = []
        
        # Combine all section patterns into one regex
        section_pattern = '|'.join(f'({pattern})' for pattern in self.SECTION_PATTERNS)
        
        # Find all potential section headings
        matches = list(re.finditer(section_pattern, full_text, re.MULTILINE))
        
        if not matches:
            # If no sections found, treat the whole paper as one section
            return [{
                "heading": "Main Content",
                "level": 1,
                "content": full_text.strip(),
                "start_offset": 0,
                "end_offset": len(full_text)
            }]
        
        # Process each section
        for i, match in enumerate(matches):
            heading = match.group(0).strip()
            level = 1  # Default level
            
            # Try to determine section level from numbered headings
            if re.match(r'^\d+\.', heading):
                level = 1
            elif re.match(r'^\d+\.\d+\.', heading):
                level = 2
            elif re.match(r'^\d+\.\d+\.\d+\.', heading):
                level = 3
            
            start_offset = match.end()
            
            # Determine end of section
            if i < len(matches) - 1:
                end_offset = matches[i + 1].start()
                content = full_text[start_offset:end_offset].strip()
            else:
                end_offset = len(full_text)
                content = full_text[start_offset:].strip()
            
            sections.append({
                "heading": heading,
                "level": level,
                "content": content,
                "start_offset": start_offset,
                "end_offset": end_offset
            })
        
        return sections
    
    def _extract_figures(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract figures and captions from the paper.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of figure dictionaries
        """
        figures = []
        
        # This is complex to do well - here we'll extract potential figure captions
        # A more robust solution would require image extraction and OCR
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n\n"
        
        # Find figure captions using regex patterns common in academic papers
        fig_patterns = [
            r'(Fig(?:ure)?\.?\s+\d+[\.:]?\s+[^\n.]+\.)',
            r'(Figure\s+\d+[\.:]?\s+[^\n.]+\.)'
        ]
        
        for pattern in fig_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                caption = match.group(0).strip()
                # Extract figure number
                fig_num_match = re.search(r'Fig(?:ure)?\.?\s+(\d+)', caption)
                fig_num = fig_num_match.group(1) if fig_num_match else "unknown"
                
                figures.append({
                    "figure_id": f"fig_{fig_num}",
                    "caption": caption,
                    "image_path": None,  # Would need image extraction
                    "reference_count": full_text.lower().count(f"fig. {fig_num}") + 
                                     full_text.lower().count(f"figure {fig_num}") - 1  # Subtract the caption itself
                })
        
        return figures
    
    def _extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract tables and captions from the paper.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of table dictionaries
        """
        tables = []
        
        # Similar to figures, we'll extract table captions
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n\n"
        
        # Find table captions
        table_patterns = [
            r'(Table\s+\d+[\.:]?\s+[^\n.]+\.)',
            r'(Tab\.\s+\d+[\.:]?\s+[^\n.]+\.)'
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, full_text)
            for match in matches:
                caption = match.group(0).strip()
                # Extract table number
                table_num_match = re.search(r'Ta(?:b(?:le)?)?\.?\s+(\d+)', caption)
                table_num = table_num_match.group(1) if table_num_match else "unknown"
                
                tables.append({
                    "table_id": f"table_{table_num}",
                    "caption": caption,
                    "data": None,  # Would need more sophisticated table extraction
                    "reference_count": full_text.lower().count(f"table {table_num}") + 
                                     full_text.lower().count(f"tab. {table_num}") - 1  # Subtract the caption itself
                })
        
        # Attempt to extract actual tables (simplified)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    for j, table_data in enumerate(page_tables):
                        # Convert to proper data structure
                        if table_data and len(table_data) > 1:  # At least header + one row
                            # Try to match with caption
                            table_id = f"extracted_table_p{i+1}_{j+1}"
                            matched = False
                            
                            # Convert table data to list of dictionaries
                            headers = [h.strip() if h else f"Column_{k}" for k, h in enumerate(table_data[0])]
                            rows = []
                            for row in table_data[1:]:
                                row_dict = {}
                                for k, cell in enumerate(row):
                                    if k < len(headers):
                                        row_dict[headers[k]] = cell.strip() if cell else ""
                                if any(row_dict.values()):  # Only add non-empty rows
                                    rows.append(row_dict)
                            
                            # Check if this table matches any caption
                            for t in tables:
                                if t["table_id"].endswith(f"_{j+1}") and t["data"] is None:
                                    t["data"] = rows
                                    matched = True
                                    break
                            
                            # If no match, add as a new table
                            if not matched and rows:
                                tables.append({
                                    "table_id": table_id,
                                    "caption": f"Extracted Table {len(tables) + 1}",
                                    "data": rows,
                                    "reference_count": 0
                                })
        except Exception as e:
            logger.warning(f"Error extracting tables from {pdf_path}: {str(e)}")
        
        return tables
    
    def _extract_references(self, paper_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract the references/bibliography from the paper.
        
        Args:
            paper_structure: Extracted paper structure
            
        Returns:
            List of reference dictionaries
        """
        references = []
        full_text = paper_structure["full_text"]
        
        # Find the references section
        ref_section_match = None
        for pattern in [
            r'References\s*\n([\s\S]+)(?:\n\s*(?:Appendix|Supplementary|Acknowledgements)|\Z)',
            r'Bibliography\s*\n([\s\S]+)(?:\n\s*(?:Appendix|Supplementary|Acknowledgements)|\Z)',
            r'Cited\s+Works\s*\n([\s\S]+)(?:\n\s*(?:Appendix|Supplementary|Acknowledgements)|\Z)',
        ]:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                ref_section_match = match
                break
        
        if not ref_section_match:
            # Try to find the section from the extracted sections
            for section in self._process_sections(paper_structure):
                if re.match(r'^References|^Bibliography|^Cited\s+Works', section["heading"], re.IGNORECASE):
                    ref_section_match = re.search(r'([\s\S]+)', section["content"])
                    break
        
        if ref_section_match:
            ref_text = ref_section_match.group(1).strip()
            
            # Different papers use different reference formats
            # We'll try a couple of common patterns
            
            # Pattern 1: Numbered references [1], [2], etc.
            numbered_refs = re.findall(r'^\s*\[(\d+)\]\s+(.+?)(?=\n\s*\[\d+\]|\Z)', ref_text, re.MULTILINE | re.DOTALL)
            if numbered_refs:
                for ref_num, ref_content in numbered_refs:
                    references.append({
                        "id": ref_num,
                        "text": ref_content.strip(),
                        "parsed": self._parse_reference(ref_content.strip())
                    })
                return references
            
            # Pattern 2: References with author names at start of line
            author_refs = re.findall(r'^\s*([A-Za-z\-]+,\s+[A-Za-z\.\s]+(?:,|and|&).+?)(?=\n[A-Za-z\-]+,|\Z)', 
                                    ref_text, re.MULTILINE | re.DOTALL)
            if author_refs:
                for i, ref_content in enumerate(author_refs):
                    references.append({
                        "id": str(i + 1),
                        "text": ref_content.strip(),
                        "parsed": self._parse_reference(ref_content.strip())
                    })
                return references
            
            # If specific patterns fail, split by lines and try to heuristically identify references
            lines = ref_text.split('\n')
            current_ref = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new reference
                if (re.match(r'^\[\d+\]', line) or 
                    re.match(r'^[A-Za-z\-]+,\s+[A-Za-z\.\s]+,', line) or
                    re.match(r'^\d+\.', line)):
                    
                    if current_ref:
                        references.append({
                            "id": str(len(references) + 1),
                            "text": current_ref.strip(),
                            "parsed": self._parse_reference(current_ref.strip())
                        })
                    
                    current_ref = line
                else:
                    current_ref += " " + line
            
            # Add the last reference
            if current_ref:
                references.append({
                    "id": str(len(references) + 1),
                    "text": current_ref.strip(),
                    "parsed": self._parse_reference(current_ref.strip())
                })
        
        return references
    
    def _parse_reference(self, ref_text: str) -> Dict[str, Any]:
        """
        Attempt to parse a reference into structured fields.
        
        Args:
            ref_text: Raw reference text
            
        Returns:
            Dictionary with parsed reference fields
        """
        parsed = {
            "authors": [],
            "year": None,
            "title": "",
            "journal": "",
            "volume": "",
            "issue": "",
            "pages": "",
            "doi": "",
            "url": ""
        }
        
        # Remove any leading numbers or brackets
        ref_text = re.sub(r'^\s*\[\d+\]|\d+\.\s+', '', ref_text).strip()
        
        # Extract DOI if present
        doi_match = re.search(r'doi:?\s*(\S+)', ref_text, re.IGNORECASE)
        if doi_match:
            parsed["doi"] = doi_match.group(1).rstrip('.')
        
        # Extract URL if present
        url_match = re.search(r'https?://\S+', ref_text)
        if url_match:
            parsed["url"] = url_match.group(0).rstrip('.')
        
        # Try to extract year
        year_match = re.search(r'\((\d{4})\)|,\s*(\d{4})\s*[,\.]', ref_text)
        if year_match:
            parsed["year"] = year_match.group(1) or year_match.group(2)
        
        # Try to extract authors (simplified)
        # This is challenging due to varied formats
        author_match = re.match(r'^([^\.]+)\.\s+', ref_text)
        if author_match:
            authors_text = author_match.group(1)
            # Split by "and" or "&" or ","
            authors = re.split(r',\s+(?:and|&)|\s+(?:and|&)\s+|,\s+', authors_text)
            parsed["authors"] = [a.strip() for a in authors if a.strip()]
        
        # Try to extract title - often between year and journal
        if parsed["year"]:
            after_year = ref_text.split(parsed["year"], 1)[1] if parsed["year"] in ref_text else ref_text
            # Title often ends with a period followed by journal name
            title_match = re.match(r'[,\.\s]*([^\.]+)\.\s+', after_year)
            if title_match:
                parsed["title"] = title_match.group(1).strip()
                
                # Journal and other details often follow the title
                journal_part = after_year[title_match.end():].strip()
                
                # Try to extract journal name
                journal_match = re.match(r'([^,]+)', journal_part)
                if journal_match:
                    parsed["journal"] = journal_match.group(1).strip()
                
                # Try to extract volume, issue, pages
                vol_issue_match = re.search(r'(\d+)[^\d]*(\d+)[^\d]*(\d+(?:-\d+)?)', journal_part)
                if vol_issue_match:
                    parsed["volume"] = vol_issue_match.group(1)
                    parsed["issue"] = vol_issue_match.group(2)
                    parsed["pages"] = vol_issue_match.group(3)
        
        return parsed
    
    def _extract_citations(self, paper_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract citations from the paper text.
        
        Args:
            paper_structure: Extracted paper structure
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        full_text = paper_structure["full_text"]
        
        # Find all citations using the defined patterns
        for pattern in self.CITATION_PATTERNS:
            citation_matches = re.finditer(pattern, full_text)
            for match in citation_matches:
                citation_text = match.group(0)
                citation_content = match.group(1)
                
                # For numbered citations [1, 2, 3]
                if re.match(r'^\d+(?:\s*,\s*\d+)*$', citation_content):
                    ref_ids = [id.strip() for id in citation_content.split(',')]
                    citations.append({
                        "text": citation_text,
                        "ref_ids": ref_ids,
                        "position": match.start(),
                        "context": full_text[max(0, match.start()-50):min(len(full_text), match.end()+50)]
                    })
                # For author-year citations (Author et al., 2020)
                else:
                    # Split multiple citations if present
                    citation_parts = re.split(r'\s*;\s*', citation_content)
                    ref_ids = []
                    
                    for part in citation_parts:
                        part = part.strip()
                        if part:
                            ref_ids.append(part)
                    
                    citations.append({
                        "text": citation_text,
                        "ref_ids": ref_ids,
                        "position": match.start(),
                        "context": full_text[max(0, match.start()-50):min(len(full_text), match.end()+50)]
                    })
        
        return citations
    
    def _extract_key_concepts(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract key concepts and entities from the paper.
        
        Args:
            sections: Processed paper sections
            
        Returns:
            List of concept dictionaries
        """
        concepts = []
        entity_counts = defaultdict(int)
        entity_examples = defaultdict(list)
        
        # Process each section for concepts
        for section in sections:
            # Skip references and acknowledgments sections
            if re.match(r'^References|^Bibliography|^Acknowledgements', section["heading"], re.IGNORECASE):
                continue
            
            # Process content with spaCy
            doc = nlp(section["content"])
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "LAW", "GPE", "LOC"]:
                    entity_counts[ent.text] += 1
                    
                    # Keep a few examples of the entity in context
                    if len(entity_examples[ent.text]) < 3:
                        start_char = max(0, ent.start_char - 50)
                        end_char = min(len(section["content"]), ent.end_char + 50)
                        context = section["content"][start_char:end_char]
                        entity_examples[ent.text].append({
                            "section": section["heading"],
                            "context": context.strip()
                        })
        
        # Find noun chunks that appear multiple times (potential domain concepts)
        text = " ".join(section["content"] for section in sections)
        doc = nlp(text)
        
        # Extract noun chunks and filter for technical concepts
        chunk_counts = defaultdict(int)
        for chunk in doc.noun_chunks:
            # Only keep chunks that look like technical terms
            if len(chunk.text.split()) <= 4 and len(chunk.text) > 3:
                # Normalize the chunk
                normalized = chunk.text.lower().strip()
                if normalized and not normalized.isdigit():
                    chunk_counts[normalized] += 1
        
        # Add entities as concepts
        for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:  # Only include entities mentioned multiple times
                concepts.append({
                    "text": entity,
                    "type": "entity",
                    "count": count,
                    "examples": entity_examples[entity]
                })
        
        # Add frequent technical terms
        for chunk, count in sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 3 and len(chunk.split()) > 1:  # Only multi-word terms mentioned 3+ times
                # Avoid duplicating entities
                if not any(concept["text"].lower() == chunk.lower() for concept in concepts):
                    concepts.append({
                        "text": chunk,
                        "type": "technical_term",
                        "count": count,
                        "examples": []  # Could add examples similar to entities
                    })
        
        # Limit to top concepts
        return concepts[:50]
    
    def process_directory(self, directory_path: Union[str, Path], recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            recursive: Whether to recursively process subdirectories
            
        Returns:
            List of paper data dictionaries
        """
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")
        
        logger.info(f"Processing academic papers in directory: {directory_path}")
        
        # Find all PDF files
        if recursive:
            pdf_files = list(directory_path.glob('**/*.pdf'))
        else:
            pdf_files = list(directory_path.glob('*.pdf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each file
        processed_papers = []
        for pdf_file in pdf_files:
            try:
                paper_data = self.process_pdf(pdf_file)
                processed_papers.append(paper_data)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        
        logger.info(f"Successfully processed {len(processed_papers)} papers")
        return processed_papers


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process academic papers in PDF format")
    parser.add_argument("input", help="PDF file or directory to process")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for processed files")
    parser.add_argument("--recursive", action="store_true", help="Recursively process subdirectories")
    parser.add_argument("--paper-id", help="Optional paper ID (for single file processing)")
    
    args = parser.parse_args()
    
    processor = AcademicPaperProcessor(output_dir=args.output_dir)
    
    input_path = Path(args.input)
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        processor.process_pdf(input_path, paper_id=args.paper_id)
    elif input_path.is_dir():
        processor.process_directory(input_path, recursive=args.recursive)
    else:
        print(f"Error: Input must be a PDF file or directory containing PDFs")
        sys.exit(1)