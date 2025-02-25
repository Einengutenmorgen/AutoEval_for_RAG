# Question Taxonomy for RAG Evaluation

This taxonomy classifies questions for the RAG evaluation dataset to ensure comprehensive testing of different retrieval and generation capabilities.

## 1. Query Type Categories

### 1.1 Factoid Questions
Questions seeking specific facts that can be directly answered from the source documents.

**Examples:**
- "What is the capital of France?"
- "When was the company founded?"
- "How many employees does the organization have?"

**Testing focus:** Basic retrieval accuracy and precise answer extraction

### 1.2 Definitional Questions
Questions seeking explanations or definitions of concepts, terms, or entities.

**Examples:**
- "What is machine learning?"
- "Define quantum computing."
- "Explain what a derivative is in finance."

**Testing focus:** Contextual understanding and summarization capabilities

### 1.3 Procedural Questions
Questions about processes, methods, or steps to accomplish something.

**Examples:**
- "How do you calculate ROI?"
- "What steps are involved in the drug approval process?"
- "How is the annual budget determined?"

**Testing focus:** Sequential information retrieval and structured explanation

### 1.4 Comparative Questions
Questions requiring comparison between multiple entities, concepts, or approaches.

**Examples:**
- "What's the difference between REST and GraphQL?"
- "How does the company's approach differ from its competitors?"
- "Compare the performance of Algorithm A and Algorithm B."

**Testing focus:** Multi-document retrieval and information synthesis

### 1.5 Causal Questions
Questions about causes, effects, or reasons.

**Examples:**
- "Why did the project fail?"
- "What caused the market decline in Q3?"
- "What are the effects of the new policy?"

**Testing focus:** Identifying cause-effect relationships across documents

### 1.6 Quantitative Questions
Questions requiring numerical analysis or calculations.

**Examples:**
- "What was the percentage increase in sales?"
- "What is the average customer acquisition cost?"
- "How much would it cost to implement the new system?"

**Testing focus:** Numerical data extraction and calculation capabilities

### 1.7 Open-ended Questions
Questions requiring comprehensive analysis and synthesis.

**Examples:**
- "What are the main challenges facing the industry?"
- "How can the company improve its customer service?"
- "What are the potential implications of the new regulation?"

**Testing focus:** Comprehensive retrieval and insight generation

## 2. Complexity Levels

### 2.1 Basic (L1)
- Single-hop retrieval required
- Answer typically found in a single document/passage
- Clear and direct question formulation

### 2.2 Intermediate (L2)
- May require multi-hop retrieval
- Answer may need to be synthesized from 2-3 documents
- Some ambiguity in question formulation
- Moderate domain knowledge required

### 2.3 Advanced (L3)
- Complex multi-hop retrieval
- Answer requires synthesis from 4+ documents
- High potential for ambiguity
- Specialized domain knowledge may be required
- May involve temporal reasoning or hypothetical scenarios

## 3. Special Categories

### 3.1 Ambiguous Questions
Questions with multiple valid interpretations to test disambiguation capabilities.

**Examples:**
- "What is the impact of the policy?" (which policy? what kind of impact?)
- "How does the system work?" (which system? what aspect of its working?)

### 3.2 Temporally Sensitive Questions
Questions whose answers depend on specific time periods or change over time.

**Examples:**
- "Who was the CEO in 2018?"
- "What were the quarterly results before the acquisition?"
- "How has the strategy evolved since 2020?"

### 3.3 Impossible Questions
Questions that cannot be answered based on the provided document corpus.

**Examples:**
- Questions about topics not covered in the corpus
- Questions requiring information from after the knowledge cutoff
- Questions with false presuppositions

### 3.4 Multi-part Questions
Questions with multiple components requiring separate retrieval and answering.

**Examples:**
- "What was the revenue in 2022, and how did it compare to 2021?"
- "Who developed the framework, when was it released, and what problem does it solve?"

## 4. Distribution Guidelines

For a balanced evaluation dataset:

| Category | Target Distribution |
|----------|---------------------|
| **Query Types** | |
| Factoid | 20% |
| Definitional | 15% |
| Procedural | 15% |
| Comparative | 15% |
| Causal | 10% |
| Quantitative | 10% |
| Open-ended | 15% |
| **Complexity** | |
| Basic (L1) | 30% |
| Intermediate (L2) | 50% |
| Advanced (L3) | 20% |
| **Special Categories** | |
| Ambiguous | 10% of total |
| Temporally Sensitive | 15% of total |
| Impossible | 5% of total |
| Multi-part | 15% of total |

## 5. Metadata Tags

Each question should be tagged with:
- Primary query type
- Secondary query type (if applicable)
- Complexity level
- Special category tags (if applicable)
- Domain/topic area
- Required number of documents for answering
- Expected answer length (short/medium/long)