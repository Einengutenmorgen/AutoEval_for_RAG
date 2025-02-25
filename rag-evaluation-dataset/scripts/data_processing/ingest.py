#!/usr/bin/env python3
"""
Document ingestion script for RAG Evaluation Dataset.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_processing import RAGDataProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ingest documents for RAG evaluation dataset')
    parser.add_argument('--config', type=str, default='config/data_config.yaml',
                        help='Path to data configuration file')
    parser.add_argument('--source', type=str, default=None,
                        help='Override source path from config')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run without actually ingesting documents')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level')
    return parser.parse_args()

def main():
    """Main entry point for document ingestion."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/ingest.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting document ingestion process")
    
    try:
        # Initialize the data processor
        processor = RAGDataProcessor(config_path=args.config)
        
        if args.dry_run:
            logger.info("DRY RUN: Would ingest documents from %s", 
                      args.source or processor.config["data_sources"]["main_source_path"])
            return
        
        # Ingest documents
        processor.ingest_documents(source_path=args.source)
        
        logger.info("Document ingestion completed successfully")
        
    except Exception as e:
        logger.error("Error during document ingestion: %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()