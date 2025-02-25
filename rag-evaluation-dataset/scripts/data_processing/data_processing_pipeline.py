import os
import yaml
import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib
import nltk
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGDataProcessor:
    """
    Data processing pipeline for RAG evaluation dataset creation
    """
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Initialize the data processor with configuration
        
        Args:
            config_path: Path to the data configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.raw_data_dir = Path(self.config["paths"]["raw_data_dir"])
        self.processed_data_dir = Path(self.config["paths"]["processed_data_dir"])
        self.splits_dir = Path(self.config["paths"]["splits_dir"])
        self.metadata_dir = Path(self.config["paths"]["metadata_dir"])
        
        # Create directories if they don't exist
        for dir_path in [self.raw_data_dir, self.processed_data_dir, 
                         self.splits_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load NLTK resources if needed
        if self.config["preprocessing"]["use_nltk"]:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def ingest_documents(self, source_path: Optional[str] = None) -> None:
        """
        Ingest documents from source path into raw data directory
        
        Args:
            source_path: Path to source documents (optional, uses config if None)
        """
        if source_path is None:
            source_path = self.config["data_sources"]["main_source_path"]
        
        source_path = Path(source_path)
        if not source_path.exists():
            logger.error(f"Source path {source_path} does not exist")
            raise FileNotFoundError(f"Source path {source_path} does not exist")
        
        # Handle different data source types
        if self.config["data_sources"]["type"] == "files":
            self._ingest_files(source_path)
        elif self.config["data_sources"]["type"] == "database":
            self._ingest_from_database(source_path)
        elif self.config["data_sources"]["type"] == "api":
            self._ingest_from_api(source_path)
        else:
            logger.error(f"Unknown data source type: {self.config['data_sources']['type']}")
            raise ValueError(f"Unknown data source type: {self.config['data_sources']['type']}")
    
    def _ingest_files(self, source_path: Path) -> None:
        """
        Ingest documents from file system
        
        Args:
            source_path: Path to source documents
        """
        logger.info(f"Ingesting files from {source_path}")
        file_extensions = self.config["data_sources"]["file_extensions"]
        
        # Get all files with the specified extensions
        files = []
        for ext in file_extensions:
            files.extend(list(source_path.glob(f"**/*.{ext}")))
        
        logger.info(f"Found {len(files)} files with extensions {file_extensions}")
        
        # Copy files to raw data directory with appropriate naming
        for i, file_path in enumerate(tqdm(files, desc="Ingesting files")):
            try:
                # Generate a unique ID for the document
                doc_id = self._generate_document_id(file_path)
                
                # Determine target format (use original extension or convert)
                target_ext = self.config["preprocessing"].get("target_format", file_path.suffix[1:])
                target_path = self.raw_data_dir / f"{doc_id}.{target_ext}"
                
                # Copy or convert the file
                if target_ext == file_path.suffix[1:]:
                    # Simple copy if format is the same
                    with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                        dst.write(src.read())
                else:
                    # TODO: Implement conversion for different formats
                    logger.warning(f"Conversion from {file_path.suffix} to {target_ext} not implemented")
                    continue
                
                # Store metadata
                self._store_document_metadata(doc_id, {
                    "original_path": str(file_path),
                    "original_filename": file_path.name,
                    "original_extension": file_path.suffix[1:],
                    "size_bytes": os.path.getsize(file_path),
                    "ingest_timestamp": pd.Timestamp.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        logger.info(f"Ingestion complete. {len(files)} files processed.")
    
    def _ingest_from_database(self, connection_string: Path) -> None:
        """
        Ingest documents from a database
        
        Args:
            connection_string: Database connection information
        """
        # TODO: Implement database ingestion
        logger.info("Database ingestion not yet implemented")
        pass
    
    def _ingest_from_api(self, api_config: Path) -> None:
        """
        Ingest documents from an API
        
        Args:
            api_config: API configuration
        """
        # TODO: Implement API ingestion
        logger.info("API ingestion not yet implemented")
        pass
    
    def _generate_document_id(self, file_path: Path) -> str:
        """
        Generate a unique document ID based on content hash and path
        
        Args:
            file_path: Path to the document
            
        Returns:
            Unique document ID
        """
        # Generate a hash based on file path and modification time
        # For larger files, we could hash just the first few KB
        path_str = str(file_path.absolute())
        mod_time = str(os.path.getmtime(file_path))
        hash_input = f"{path_str}:{mod_time}"
        
        # Create hash
        hash_obj = hashlib.md5(hash_input.encode())
        return hash_obj.hexdigest()[:12]  # Use first 12 characters of hash
    
    def _store_document_metadata(self, doc_id: str, metadata: Dict) -> None:
        """
        Store document metadata
        
        Args:
            doc_id: Document ID
            metadata: Metadata dictionary
        """
        metadata_path = self.metadata_dir / f"{doc_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def preprocess_documents(self) -> None:
        """
        Preprocess all documents in the raw data directory
        """
        logger.info("Starting document preprocessing")
        
        # Get all documents in raw data directory
        raw_docs = list(self.raw_data_dir.glob("*.*"))
        logger.info(f"Found {len(raw_docs)} documents for preprocessing")
        
        for doc_path in tqdm(raw_docs, desc="Preprocessing documents"):
            try:
                doc_id = doc_path.stem
                # Load metadata
                metadata_path = self.metadata_dir / f"{doc_id}.json"
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Process based on file type
                extension = doc_path.suffix[1:]
                
                if extension in ["txt", "md", "rst"]:
                    self._process_text_document(doc_path, doc_id)
                elif extension in ["pdf"]:
                    self._process_pdf_document(doc_path, doc_id)
                elif extension in ["html", "htm"]:
                    self._process_html_document(doc_path, doc_id)
                elif extension in ["doc", "docx"]:
                    self._process_word_document(doc_path, doc_id)
                elif extension in ["csv", "xls", "xlsx"]:
                    self._process_structured_document(doc_path, doc_id)
                elif extension in ["json", "xml", "yaml"]:
                    self._process_data_document(doc_path, doc_id)
                else:
                    logger.warning(f"Unsupported file type: {extension} for {doc_path}")
                    continue
                
                # Update metadata with preprocessing info
                metadata["preprocessing"] = {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "processor_version": "0.1.0"
                }
                
                # Save updated metadata
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error preprocessing document {doc_path}: {e}")
        
        logger.info("Document preprocessing complete")
    
    def _process_text_document(self, doc_path: Path, doc_id: str) -> None:
        """
        Process plain text documents
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
        """
        try:
            # Read the text file
            with open(doc_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Apply text cleaning
            if self.config["preprocessing"]["clean_text"]:
                text = self._clean_text(text)
            
            # Segment into chunks if required
            if self.config["preprocessing"]["chunk_documents"]:
                chunks = self._chunk_text(text)
                # Save each chunk separately
                for i, chunk in enumerate(chunks):
                    chunk_path = self.processed_data_dir / f"{doc_id}_chunk_{i:03d}.txt"
                    with open(chunk_path, 'w', encoding='utf-8') as f:
                        f.write(chunk)
            else:
                # Save the entire processed document
                processed_path = self.processed_data_dir / f"{doc_id}.txt"
                with open(processed_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
        except Exception as e:
            logger.error(f"Error processing text document {doc_path}: {e}")
            raise
    
    def _process_pdf_document(self, doc_path: Path, doc_id: str) -> None:
        """
        Process PDF documents
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
        """
        # TODO: Implement PDF processing (requires external library)
        logger.info(f"PDF processing not fully implemented for {doc_path}")
        
        # Placeholder for PDF extraction logic
        # You would use a library like PyPDF2, pdfminer.six, or pdf2text
        
        # Example pseudocode:
        # text = extract_text_from_pdf(doc_path)
        # processed_path = self.processed_data_dir / f"{doc_id}.txt"
        # with open(processed_path, 'w', encoding='utf-8') as f:
        #     f.write(text)
    
    def _process_html_document(self, doc_path: Path, doc_id: str) -> None:
        """
        Process HTML documents
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
        """
        # TODO: Implement HTML processing
        logger.info(f"HTML processing not fully implemented for {doc_path}")
        
        # Placeholder for HTML extraction logic
        # You would use a library like BeautifulSoup
        
        # Example pseudocode:
        # from bs4 import BeautifulSoup
        # with open(doc_path, 'r', encoding='utf-8') as f:
        #     soup = BeautifulSoup(f, 'html.parser')
        # text = soup.get_text()
        # processed_path = self.processed_data_dir / f"{doc_id}.txt"
        # with open(processed_path, 'w', encoding='utf-8') as f:
        #     f.write(text)
    
    def _process_word_document(self, doc_path: Path, doc_id: str) -> None:
        """
        Process Word documents
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
        """
        # TODO: Implement Word document processing
        logger.info(f"Word document processing not fully implemented for {doc_path}")
        
        # Placeholder for Word document extraction logic
        # You would use a library like python-docx for .docx files
        
        # Example pseudocode:
        # from docx import Document
        # doc = Document(doc_path)
        # text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        # processed_path = self.processed_data_dir / f"{doc_id}.txt"
        # with open(processed_path, 'w', encoding='utf-8') as f:
        #     f.write(text)
    
    def _process_structured_document(self, doc_path: Path, doc_id: str) -> None:
        """
        Process structured documents like CSV, Excel
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
        """
        # TODO: Implement structured document processing
        logger.info(f"Structured document processing not fully implemented for {doc_path}")
        
        # Placeholder for structured document extraction logic
        # You would use libraries like pandas
        
        # Example pseudocode:
        # if doc_path.suffix.lower() == '.csv':
        #     df = pd.read_csv(doc_path)
        # else:  # Excel
        #     df = pd.read_excel(doc_path)
        # 
        # # Convert to text format
        # text = df.to_string()
        # processed_path = self.processed_data_dir / f"{doc_id}.txt"
        # with open(processed_path, 'w', encoding='utf-8') as f:
        #     f.write(text)
        #
        # # Optionally also save as JSON for structured access
        # json_path = self.processed_data_dir / f"{doc_id}.json"
        # df.to_json(json_path, orient='records', lines=True)
    
    def _process_data_document(self, doc_path: Path, doc_id: str) -> None:
        """
        Process data documents like JSON, XML, YAML
        
        Args:
            doc_path: Path to the document
            doc_id: Document ID
        """
        # TODO: Implement data document processing
        logger.info(f"Data document processing not fully implemented for {doc_path}")
        
        # Placeholder for data document extraction logic
        # Would depend on the specific format
        
        # Example pseudocode for JSON:
        # with open(doc_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        # 
        # # Convert to text if needed
        # text = json.dumps(data, indent=2)
        # processed_path = self.processed_data_dir / f"{doc_id}.txt"
        # with open(processed_path, 'w', encoding='utf-8') as f:
        #     f.write(text)
        #
        # # Also keep the structured data
        # processed_json = self.processed_data_dir / f"{doc_id}.json"
        # with open(processed_json, 'w', encoding='utf-8') as f:
        #     json.dump(data, f, indent=2)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace, special characters, etc.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters if configured
        if self.config["preprocessing"].get("remove_special_chars", False):
            text = re.sub(r'[^\w\s.,?!-]', '', text)
        
        # Additional cleaning as specified in config
        # ...
        
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of configured size
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunk_size = self.config["preprocessing"].get("chunk_size", 1000)
        overlap = self.config["preprocessing"].get("chunk_overlap", 100)
        
        if self.config["preprocessing"].get("chunk_by", "chars") == "chars":
            # Chunk by characters
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
        else:
            # Chunk by sentences
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence_size = len(sentence)
                if current_size + sentence_size > chunk_size and current_chunk:
                    # Start a new chunk if adding this sentence would exceed chunk size
                    chunks.append(' '.join(current_chunk))
                    
                    # Keep the last few sentences for overlap
                    overlap_tokens = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        if overlap_size + len(sent) <= overlap:
                            overlap_tokens.insert(0, sent)
                            overlap_size += len(sent)
                        else:
                            break
                    
                    current_chunk = overlap_tokens
                    current_size = overlap_size
                
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_data_splits(self) -> None:
        """
        Create train/validation/test splits from processed documents
        """
        logger.info("Creating data splits")
        
        # Get all processed documents
        processed_docs = list(self.processed_data_dir.glob("*.*"))
        logger.info(f"Found {len(processed_docs)} processed documents for splitting")
        
        # Extract document IDs (removing chunk suffixes if any)
        doc_ids = set()
        for doc_path in processed_docs:
            # Extract base document ID without chunk suffix
            doc_id = re.sub(r'_chunk_\d+$', '', doc_path.stem)
            doc_ids.add(doc_id)
        
        doc_ids = list(doc_ids)
        logger.info(f"Found {len(doc_ids)} unique documents for splitting")
        
        # Get split ratios from config
        train_ratio = self.config["data_splits"].get("train_ratio", 0.7)
        val_ratio = self.config["data_splits"].get("val_ratio", 0.15)
        test_ratio = self.config["data_splits"].get("test_ratio", 0.15)
        
        # Verify ratios sum to 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            logger.warning(f"Split ratios {train_ratio}, {val_ratio}, {test_ratio} do not sum to 1")
            # Normalize ratios
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
        
        # Create stratified splits if metadata available
        if self.config["data_splits"].get("stratify_by") and all(os.path.exists(self.metadata_dir / f"{doc_id}.json") for doc_id in doc_ids):
            # Load metadata to extract stratification variable
            stratify_by = self.config["data_splits"]["stratify_by"]
            strata = []
            
            for doc_id in doc_ids:
                with open(self.metadata_dir / f"{doc_id}.json", 'r') as f:
                    metadata = json.load(f)
                
                # Extract stratification variable, default to "unknown" if not found
                strata.append(metadata.get(stratify_by, "unknown"))
            
            # Create stratified train/test split first
            train_ids, test_val_ids, _, test_val_strata = train_test_split(
                doc_ids, strata, 
                train_size=train_ratio,
                stratify=strata,
                random_state=self.config["data_splits"].get("random_seed", 42)
            )
            
            # Then split test_val into test and validation
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_ids, test_ids = train_test_split(
                test_val_ids,
                train_size=val_ratio_adjusted,
                stratify=test_val_strata,
                random_state=self.config["data_splits"].get("random_seed", 42)
            )
            
        else:
            # Create random splits
            train_ids, test_val_ids = train_test_split(
                doc_ids, 
                train_size=train_ratio,
                random_state=self.config["data_splits"].get("random_seed", 42)
            )
            
            # Then split test_val into test and validation
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_ids, test_ids = train_test_split(
                test_val_ids,
                train_size=val_ratio_adjusted,
                random_state=self.config["data_splits"].get("random_seed", 42)
            )
        
        logger.info(f"Split sizes - Train: {len(train_ids)}, Validation: {len(val_ids)}, Test: {len(test_ids)}")
        
        # Create split files
        self._write_split_file("train.txt", train_ids)
        self._write_split_file("val.txt", val_ids)
        self._write_split_file("test.txt", test_ids)
        
        # Optionally copy files to split directories
        if self.config["data_splits"].get("create_split_dirs", False):
            self._copy_to_split_dirs(train_ids, val_ids, test_ids)
        
        logger.info("Data splits created successfully")
    
    def _write_split_file(self, filename: str, doc_ids: List[str]) -> None:
        """
        Write document IDs to a split file
        
        Args:
            filename: Name of the split file
            doc_ids: List of document IDs in the split
        """
        split_path = self.splits_dir / filename
        with open(split_path, 'w', encoding='utf-8') as f:
            for doc_id in doc_ids:
                f.write(f"{doc_id}\n")
    
    def _copy_to_split_dirs(self, train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> None:
        """
        Copy documents to train/val/test directories
        
        Args:
            train_ids: List of training document IDs
            val_ids: List of validation document IDs
            test_ids: List of test document IDs
        """
        # Create split directories
        train_dir = self.splits_dir / "train"
        val_dir = self.splits_dir / "val"
        test_dir = self.splits_dir / "test"
        
        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Copy files to respective directories
        self._copy_files_for_ids(train_ids, train_dir)
        self._copy_files_for_ids(val_ids, val_dir)
        self._copy_files_for_ids(test_ids, test_dir)
    
    def _copy_files_for_ids(self, doc_ids: List[str], target_dir: Path) -> None:
        """
        Copy all files for given document IDs to target directory
        
        Args:
            doc_ids: List of document IDs
            target_dir: Target directory
        """
        for doc_id in doc_ids:
            # Find all files for this document (including chunks)
            files = list(self.processed_data_dir.glob(f"{doc_id}*.*"))
            
            # Copy each file
            for file_path in files:
                target_path = target_dir / file_path.name
                with open(file_path, 'rb') as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())


if __name__ == "__main__":
    # Example usage
    processor = RAGDataProcessor()
    
    # Ingest documents
    processor.ingest_documents()
    
    # Preprocess documents
    processor.preprocess_documents()
    
    # Create data splits
    processor.create_data_splits()