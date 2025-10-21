import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. Please set it in your .env file."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


PROMPT_TEMPLATE = """
You are an expert business analyst. Your task is to analyze a single spreadsheet cell's context
and provide its semantic meaning in a structured JSON format. Do not provide any text or explanation
outside of the JSON object.

Here is the data for the cell:
- Filename: "{filename}"
- Sheet Name: "{sheet_name}"
- Cell Address: "{cell_address}"
- Formula: "{formula}"
- Column Header: "{col_header}"
- Row Header: "{row_header}"

Based on this context, provide the following information in a JSON object with these exact keys:
1.  "concept_name": A clear business concept name (e.g., "Total Revenue", "Gross Margin Percentage", "Year-over-Year Sales Growth").
2.  "concept_category": A broader category (e.g., "Profitability Metric", "Growth Rate", "Efficiency Ratio", "Key Performance Indicator").
3.  "explanation": A brief, one-sentence explanation of what this cell represents in plain English for a business user.
4.  "functional_type": The type of calculation being performed (e.g., "Summation", "Percentage", "Ratio", "Average", "Conditional Logic", "Lookup").

JSON Output:
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

parser = StrOutputParser()


enrichment_chain = prompt | llm | parser


# --- Main Enrichment Function ---


def create_langchain_documents(
    parsed_data_list: List[Dict[str, Any]],
) -> List[Document]:
    """
    Enriches parsed spreadsheet data using an LLM and converts it into LangChain Documents.

    This function iterates through a list of cell data dictionaries (from parser.py),
    uses a Gemini-powered chain to infer semantic meaning, and then constructs
    LangChain Document objects that are ready for embedding and indexing.

    Args:
        parsed_data_list: A list of dictionaries, where each dictionary represents
                          a parsed formula cell from the spreadsheet.

    Returns:
        A list of LangChain Document objects, ready for ingestion into a vector store.
    """
    # This list will hold the final, enriched Document objects.
    langchain_documents = []

    # Process each parsed cell one by one.
    for cell_data in parsed_data_list:
        try:
            # The input for our chain requires the context data to be flattened.
            # We create a dictionary that matches the placeholders in our prompt template.
            chain_input = {
                "filename": cell_data.get("filename"),
                "sheet_name": cell_data.get("sheet_name"),
                "cell_address": cell_data.get("cell_address"),
                "formula": cell_data.get("formula", ""),
                "col_header": cell_data.get("context", {}).get("column_header", "N/A"),
                "row_header": cell_data.get("context", {}).get("row_header", "N/A"),
            }

            # Invoke the chain with the prepared input to get the LLM's response.
            llm_response_str = enrichment_chain.invoke(chain_input)

            # The LLM's response is a string containing JSON. We need to parse it.
            # This is a critical step that can fail if the LLM deviates from the prompt.
            semantic_data = json.loads(llm_response_str)

            # --- Construct the Document for Vectorization ---

            # 1. Create the `page_content`: This is the rich text that will be embedded.
            # It's a synthesis of the most important semantic information.
            page_content = (
                f"Concept: {semantic_data.get('concept_name', 'N/A')}. "
                f"Category: {semantic_data.get('concept_category', 'N/A')}. "
                f"Description: {semantic_data.get('explanation', 'N/A')}"
            )

            # 2. Create the `metadata`: This is all the structured data we want to
            # store alongside the vector. It's used for filtering (e.g., by filename)
            # and for displaying results to the user.
            metadata = {
                "source": cell_data.get("filename"),
                "sheet_name": cell_data.get("sheet_name"),
                "cell_address": cell_data.get("cell_address"),
                "formula": cell_data.get("formula"),
                "concept_name": semantic_data.get("concept_name"),
                "concept_category": semantic_data.get("concept_category"),
                "explanation": semantic_data.get("explanation"),
            }

            # Create the final LangChain Document object and add it to our list.
            doc = Document(page_content=page_content, metadata=metadata)
            langchain_documents.append(doc)

        except json.JSONDecodeError:
            # If the LLM returns a malformed string that isn't valid JSON,
            # we print an error and skip this cell to avoid crashing the whole process.
            print(
                f"Error: Could not decode JSON for cell {cell_data.get('cell_address')} in {cell_data.get('filename')}."
            )
            print(f"LLM Response was: {llm_response_str}")
        except Exception as e:
            # Catch any other unexpected errors during processing.
            print(
                f"An unexpected error occurred for cell {cell_data.get('cell_address')}: {e}"
            )

    return langchain_documents
