#!/usr/bin/env python3
"""
RAG system connector for evaluation framework.
Provides interfaces for connecting to various RAG implementations.
"""

import os
import sys
import json
import requests
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml
import hashlib
from abc import ABC, abstractmethod
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rag_connector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGConnector(ABC):
    """Abstract base class for RAG system connectors."""
    
    @abstractmethod
    def query(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question text
            metadata: Optional metadata about the question
            
        Returns:
            Response dictionary
        """
        pass
    
    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system.
        
        Returns:
            System information dictionary
        """
        pass


class RESTAPIConnector(RAGConnector):
    """Connector for RAG systems with REST API endpoints."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the REST API connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get("name", "rest_api_rag")
        self.description = config.get("description", "REST API RAG System")
        self.api_endpoint = config.get("api_endpoint", "http://localhost:8000/query")
        self.request_format = config.get("request_format", "json")
        self.authentication = config.get("authentication", {"type": "none"})
        self.timeout = config.get("timeout", 30)
        self.retry_count = config.get("retry", {}).get("attempts", 3)
        self.retry_backoff = config.get("retry", {}).get("backoff_factor", 2)
        
        # Set up auth token environment variable handling
        self._setup_auth()
    
    def _setup_auth(self) -> None:
        """Set up authentication from environment variables if needed."""
        if self.authentication["type"] == "bearer":
            token = self.authentication.get("token", "")
            
            # Replace environment variable if present
            if token.startswith("${") and token.endswith("}"):
                env_var = token[2:-1]
                self.authentication["token"] = os.environ.get(env_var, "")
                
                if not self.authentication["token"]:
                    logger.warning(f"Environment variable {env_var} not found for bearer token")
        
        elif self.authentication["type"] == "api_key":
            key = self.authentication.get("key", "")
            
            # Replace environment variable if present
            if key.startswith("${") and key.endswith("}"):
                env_var = key[2:-1]
                self.authentication["key"] = os.environ.get(env_var, "")
                
                if not self.authentication["key"]:
                    logger.warning(f"Environment variable {env_var} not found for API key")
    
    def query(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question text
            metadata: Optional metadata about the question
            
        Returns:
            Response dictionary
        """
        # Prepare the request
        headers = {"Content-Type": "application/json"}
        
        # Add authentication
        if self.authentication["type"] == "bearer":
            headers["Authorization"] = f"Bearer {self.authentication['token']}"
        elif self.authentication["type"] == "api_key":
            header_name = self.authentication.get("header_name", "X-API-Key")
            headers[header_name] = self.authentication["key"]
        
        # Prepare data
        if self.request_format == "json":
            data = {
                "question": question
            }
            # Add metadata if provided
            if metadata:
                data["metadata"] = metadata
        else:
            # Default to simple text format
            data = {"query": question}
        
        # Record start time
        start_time = time.time()
        
        # Send request with retries
        response_data = None
        success = False
        error_message = ""
        
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()  # Raise error for bad status codes
                response_data = response.json()
                success = True
                break
            
            except requests.RequestException as e:
                error_message = str(e)
                logger.warning(f"Request failed (attempt {attempt+1}/{self.retry_count}): {error_message}")
                
                if attempt < self.retry_count - 1:
                    # Wait before retrying (exponential backoff)
                    sleep_time = self.retry_backoff ** attempt
                    time.sleep(sleep_time)
        
        # Record end time
        end_time = time.time()
        latency = end_time - start_time
        
        # Prepare response object
        result = {
            "success": success,
            "latency": latency,
            "timestamp": time.time()
        }
        
        if success:
            result["response"] = response_data
            
            # Extract answer if available
            if "answer" in response_data:
                result["answer"] = response_data["answer"]
            elif "text" in response_data:
                result["answer"] = response_data["text"]
            elif "result" in response_data:
                result["answer"] = response_data["result"]
            else:
                # Fallback: use entire response
                result["answer"] = str(response_data)
            
            # Extract retrieved contexts if available
            if "contexts" in response_data:
                result["contexts"] = response_data["contexts"]
            elif "context" in response_data:
                result["contexts"] = [response_data["context"]]
        else:
            result["error"] = error_message
        
        return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system.
        
        Returns:
            System information dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "api_endpoint": self.api_endpoint,
            "request_format": self.request_format,
            "authentication": {
                "type": self.authentication["type"],
                # Don't include actual auth tokens for security
                **({"header_name": self.authentication.get("header_name")} 
                   if self.authentication["type"] == "api_key" else {})
            }
        }


class LangChainConnector(RAGConnector):
    """Connector for LangChain RAG implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LangChain connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get("name", "langchain_rag")
        self.description = config.get("description", "LangChain RAG System")
        
        # Check if LangChain is available
        try:
            from langchain.chat_models import ChatOpenAI
            from langchain.chains import RetrievalQA
            from langchain.vectorstores import FAISS
            from langchain.embeddings import OpenAIEmbeddings
            self.langchain_available = True
        except ImportError:
            logger.warning("LangChain not available. Please install with: pip install langchain openai faiss-cpu")
            self.langchain_available = False
        
        self.qa_chain = None
        
        # Initialize the RAG chain if LangChain is available
        if self.langchain_available:
            self._initialize_rag_chain()
    
    def _initialize_rag_chain(self) -> None:
        """Initialize the LangChain RAG chain."""
        try:
            # Import required components
            from langchain.chat_models import ChatOpenAI
            from langchain.chains import RetrievalQA
            from langchain.vectorstores import FAISS
            from langchain.embeddings import OpenAIEmbeddings
            
            # Get vector store path
            vector_store_path = self.config.get("vector_store_path")
            
            if not vector_store_path or not os.path.exists(vector_store_path):
                logger.error(f"Vector store path not found: {vector_store_path}")
                return
            
            # Load vector store
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
            
            vector_store = FAISS.load_local(
                vector_store_path,
                embeddings
            )
            
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.get("top_k", 5)}
            )
            
            # Create LLM
            llm = ChatOpenAI(
                model_name=self.config.get("model_name", "gpt-3.5-turbo"),
                temperature=self.config.get("temperature", 0),
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            logger.info("LangChain RAG chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LangChain RAG chain: {e}")
            self.qa_chain = None
    
    def query(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question text
            metadata: Optional metadata about the question
            
        Returns:
            Response dictionary
        """
        if not self.langchain_available or self.qa_chain is None:
            return {
                "success": False,
                "error": "LangChain RAG chain not initialized",
                "latency": 0,
                "timestamp": time.time()
            }
        
        # Record start time
        start_time = time.time()
        
        try:
            # Query the chain
            result = self.qa_chain({"query": question})
            
            # Extract answer
            answer = result.get("result", "")
            
            # Extract source documents as contexts
            source_docs = result.get("source_documents", [])
            contexts = []
            
            for i, doc in enumerate(source_docs):
                contexts.append({
                    "id": f"doc_{i+1}",
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Record end time
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "success": True,
                "answer": answer,
                "contexts": contexts,
                "latency": latency,
                "timestamp": time.time()
            }
            
        except Exception as e:
            # Record end time for failed query
            end_time = time.time()
            latency = end_time - start_time
            
            logger.error(f"Error querying LangChain RAG chain: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "timestamp": time.time()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system.
        
        Returns:
            System information dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": "langchain",
            "model": self.config.get("model_name", "gpt-3.5-turbo"),
            "top_k": self.config.get("top_k", 5),
            "vector_store": os.path.basename(self.config.get("vector_store_path", ""))
        }


class HuggingFaceConnector(RAGConnector):
    """Connector for Hugging Face RAG implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Hugging Face connector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = config.get("name", "huggingface_rag")
        self.description = config.get("description", "Hugging Face RAG System")
        
        # Check if Hugging Face Transformers is available
        try:
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            self.hf_available = True
        except ImportError:
            logger.warning("Hugging Face Transformers not available. Please install with: pip install transformers")
            self.hf_available = False
        
        self.tokenizer = None
        self.model = None
        self.retriever = None
        
        # Initialize the RAG components if HF is available
        if self.hf_available:
            self._initialize_rag_components()
    
    def _initialize_rag_components(self) -> None:
        """Initialize the Hugging Face RAG components."""
        try:
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model and tokenizer
            model_name = self.config.get("model_name", "meta-llama/Llama-2-7b-chat-hf")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype="auto"
            )
            
            # Initialize retriever (simplified for demonstration)
            self._initialize_retriever()
            
            logger.info("Hugging Face RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Hugging Face RAG components: {e}")
            self.tokenizer = None
            self.model = None
    
    def _initialize_retriever(self) -> None:
        """Initialize the retriever component."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            
            # Load embeddings
            embeddings_model_name = self.config.get("embeddings_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings_model = SentenceTransformer(embeddings_model_name)
            
            # Load document store if available
            document_store_path = self.config.get("document_store_path")
            
            if document_store_path and os.path.exists(document_store_path):
                # Load documents and index
                with open(document_store_path, 'r') as f:
                    self.documents = json.load(f)
                
                index_path = document_store_path.replace(".json", ".index")
                if os.path.exists(index_path):
                    self.index = faiss.read_index(index_path)
                    
                    # Calculate embeddings for documents if needed
                    if "embeddings" not in self.documents:
                        texts = [doc["text"] for doc in self.documents["documents"]]
                        self.documents["embeddings"] = self.embeddings_model.encode(texts)
                
                logger.info("Document retriever initialized successfully")
                self.retriever = "faiss"  # Mark as initialized
            else:
                logger.warning(f"Document store not found: {document_store_path}")
                self.retriever = None
        
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            self.retriever = None
    
    def _retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if self.retriever is None:
            return []
        
        try:
            # Embed the query
            query_embedding = self.embeddings_model.encode([query])[0]
            
            # Search the index
            D, I = self.index.search(
                np.array([query_embedding]).astype("float32"), 
                top_k
            )
            
            # Return retrieved documents
            results = []
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.documents["documents"]):
                    doc = self.documents["documents"][idx]
                    results.append({
                        "id": f"doc_{idx}",
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {}),
                        "score": float(D[0][i])
                    })
            
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def query(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question text
            metadata: Optional metadata about the question
            
        Returns:
            Response dictionary
        """
        if not self.hf_available or self.model is None or self.tokenizer is None:
            return {
                "success": False,
                "error": "Hugging Face RAG components not initialized",
                "latency": 0,
                "timestamp": time.time()
            }
        
        # Record start time
        start_time = time.time()
        
        try:
            # Retrieve documents
            top_k = self.config.get("top_k", 5)
            contexts = self._retrieve_documents(question, top_k) if self.retriever else []
            
            # Format prompt with retrieved contexts
            context_text = ""
            if contexts:
                context_text = "Context information:\n"
                for ctx in contexts:
                    context_text += f"- {ctx['text']}\n"
            
            # Create prompt
            prompt = f"""
            {context_text}
            
            Question: {question}
            
            Answer:
            """
            
            # Generate answer
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=self.config.get("max_length", 512),
                temperature=self.config.get("temperature", 0.7),
                top_p=self.config.get("top_p", 0.9),
                num_return_sequences=1
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the answer (extract only the part after "Answer:")
            answer_match = re.search(r'Answer:(.*?)(?:\n|$)', answer, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            
            # Record end time
            end_time = time.time()
            latency = end_time - start_time
            
            return {
                "success": True,
                "answer": answer,
                "contexts": contexts,
                "latency": latency,
                "timestamp": time.time()
            }
            
        except Exception as e:
            # Record end time for failed query
            end_time = time.time()
            latency = end_time - start_time
            
            logger.error(f"Error querying Hugging Face RAG system: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "timestamp": time.time()
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the RAG system.
        
        Returns:
            System information dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": "huggingface",
            "model": self.config.get("model_name", "meta-llama/Llama-2-7b-chat-hf"),
            "embeddings_model": self.config.get("embeddings_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "top_k": self.config.get("top_k", 5),
            "max_length": self.config.get("max_length", 512)
        }


class RAGConnectorFactory:
    """Factory for creating RAG system connectors."""
    
    @staticmethod
    def create_connector(config: Dict[str, Any]) -> RAGConnector:
        """
        Create a RAG connector based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            RAG connector instance
        """
        connector_type = config.get("type", "rest_api").lower()
        
        if connector_type == "rest_api":
            return RESTAPIConnector(config)
        elif connector_type == "langchain":
            return LangChainConnector(config)
        elif connector_type == "huggingface":
            return HuggingFaceConnector(config)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")
    
    @staticmethod
    def load_from_config(config_path: str) -> List[RAGConnector]:
        """
        Load RAG connectors from a configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            List of RAG connector instances
        """
        with open(config_path, 'r') as f:
            if config_path.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        connectors = []
        
        # Handle both single and multiple connector configs
        if "target_systems" in config:
            for system_config in config["target_systems"]:
                try:
                    connector = RAGConnectorFactory.create_connector(system_config)
                    connectors.append(connector)
                except Exception as e:
                    logger.error(f"Error creating connector for {system_config.get('name', 'unknown')}: {e}")
        else:
            try:
                connector = RAGConnectorFactory.create_connector(config)
                connectors.append(connector)
            except Exception as e:
                logger.error(f"Error creating connector: {e}")
        
        return connectors


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Connector")
    parser.add_argument("--config", required=True, help="Path to connector configuration")
    parser.add_argument("--question", help="Test question to send to the RAG system")
    
    args = parser.parse_args()
    
    # Load connectors
    connectors = RAGConnectorFactory.load_from_config(args.config)
    
    if not connectors:
        print("No connectors loaded")
        sys.exit(1)
    
    print(f"Loaded {len(connectors)} connectors")
    
    for connector in connectors:
        print(f"\nConnector: {connector.name}")
        system_info = connector.get_system_info()
        print(f"System info: {json.dumps(system_info, indent=2)}")
        
        if args.question:
            print(f"\nSending question: {args.question}")
            response = connector.query(args.question)
            
            if response["success"]:
                print(f"Answer: {response['answer']}")
                print(f"Latency: {response['latency']:.2f} s")
                
                if "contexts" in response:
                    print(f"Retrieved {len(response['contexts'])} contexts")
            else:
                print(f"Error: {response.get('error', 'Unknown error')}")