# SuperJoin Semantic Search Engine for Spreadsheets

## Project Overview

SuperJoin Semantic Search is an intelligent search engine that bridges the gap between how users think about spreadsheet data and how spreadsheets actually work. It allows users to find relevant spreadsheet content using natural language queries based on business concepts rather than exact cell references or text matches.

### The Problem It Solves

Traditional spreadsheet search functions are limited to structural queries like finding specific text or formula types. However, users think semantically - they want to find "profitability metrics" or "cost calculations" regardless of how those concepts are labeled in their spreadsheets.

This semantic search engine:

1. **Understands Business Concepts**: Recognizes that "Q1 Revenue", "First Quarter Sales", and "Jan-Mar Income" all refer to similar concepts
2. **Interprets Context**: Distinguishes between "Marketing Spend" (cost) vs "Marketing ROI" (efficiency metric)
3. **Finds Conceptual Matches**: When someone searches "profitability", it finds gross margin, net profit, EBITDA calculations
4. **Handles Natural Language**: Processes queries like "show efficiency metrics" or "find budget vs actual comparisons"

## Project Structure

```
SuperJoin_Search_Engine/
│
├── app.py                  # Main Streamlit application for the UI
├── requirements.txt        # Python dependencies
│
├── core/                   # Core functionality modules
│   ├── parser.py           # Parses Excel spreadsheets and extracts formula cells with context
│   ├── enricher.py         # Enriches formula data with semantic meaning using LLMs
│   └── vector_store.py     # Manages the vector database for semantic search
│
└── .env                    # Environment variables for API keys (create from .env.local)
```

## Tech Stack

- **Frontend**: Streamlit for the interactive UI
- **Backend**: Python
- **LLM**: Google Gemini 2.0 Flash (for semantic enrichment)
- **Vector Database**: Chroma DB (with cloud storage)
- **Embeddings**: Google Generative AI Embeddings
- **Spreadsheet Parsing**: OpenPyXL
- **AI/ML Framework**: LangChain for orchestration

## How It Works

1. **Parsing**:

   - The system extracts all formula cells from uploaded Excel spreadsheets using OpenPyXL
   - For each formula, it captures the context (sheet name, cell address, column/row headers)

2. **Semantic Enrichment**:

   - Formula cells are sent in batches to Google Gemini to extract business concepts
   - The LLM identifies key information like concept name, category, functional type, and provides an explanation
   - This enrichment gives the system semantic understanding of what each formula represents

3. **Vector Database**:

   - The enriched data is converted into vector embeddings using Google's embedding model
   - These embeddings are stored in a ChromaDB vector database for efficient semantic search

4. **Query Processing**:
   - When users enter natural language queries, the system converts them to vectors
   - It performs similarity search to find the most relevant formulas
   - Results are ranked and returned with rich contextual information

## Getting Started

### Prerequisites

- Python 3.8 or later
- Git
- Google API key (for Gemini LLM and embeddings)
- ChromaDB Cloud account (optional, for persistent storage)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ravin1100/SuperJoin_Search_Engine.git
   cd SuperJoin_Search_Engine
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:

   - Create a `.env` file in the project root (you can copy from `.env.local`)
   - Add the following keys:

     ```
     GOOGLE_API_KEY=your_google_api_key

     # Optional: For ChromaDB Cloud (persistent storage)
     CHROMA_API_KEY=your_chroma_api_key
     CHROMA_TENANT=your_tenant_name
     CHROMA_COLLECTION=your_collection_name
     CHROMA_DATABASE=your_database_name
     ```

### Running the Application

1. **Start the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Upload and index spreadsheets**:

   - Use the sidebar file uploader to upload Excel (.xlsx) files
   - Click "Index Files" to process the spreadsheets
   - The system will extract formulas, enrich them with semantic understanding, and store them in the vector database

3. **Query the semantic search engine**:
   - Type natural language queries in the chat input
   - Examples:
     - "Find all profitability metrics"
     - "Show me cost calculations"
     - "Where are my growth rates?"
     - "Show percentage formulas"
     - "Find variance analysis between budget and actual"
