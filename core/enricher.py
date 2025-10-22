import os
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. Please set it in your .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)


class CellAnalysis(BaseModel):
    concept_name: str = Field(
        description="A clear business concept name (e.g., 'Total Revenue', 'Gross Margin Percentage' etc)."
    )
    concept_category: str = Field(
        description="A broader category (e.g., 'Profitability Metric', 'Growth Rate', 'KPI' etc)."
    )
    explanation: str = Field(
        description="A brief, one-sentence explanation of this cell's business purpose."
    )
    functional_type: str = Field(
        description="The type of calculation (e.g., 'Summation', 'Percentage', 'Ratio', 'Average', 'Conditional Logic', 'Lookup' etc)."
    )


class AnalysisList(BaseModel):
    """A list of cell analysis results."""

    analyses: List[CellAnalysis]


# Use the .with_structured_output method to bind our Pydantic model to the LLM.
# This forces the LLM's output into our desired format and handles parsing internally.
structured_llm = llm.with_structured_output(AnalysisList)

# --- Prompt Engineering for Batch Processing ---
# The prompt is updated to handle a LIST of cell contexts at once.
PROMPT_TEMPLATE = """
You are an expert business analyst. Your task is to analyze a batch of spreadsheet cell contexts
and provide their semantic meaning. For each cell context provided in the list, generate a
corresponding JSON object with the required semantic details.

Respond with a single JSON object that contains a key "analyses", which holds a list of your analysis results.
The order of your results in the list MUST match the order of the cell contexts in the input.

Here is the list of cell contexts to analyze:
{batch_input}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
enrichment_chain = prompt | structured_llm


# --- Main Enrichment Function (Now with Batching) ---
def create_langchain_documents(
    parsed_data_list: List[Dict[str, Any]],
) -> List[Document]:
    langchain_documents = []
    batch_size = 10  # Process 10 cells per API call to respect rate limits

    # Process the data in chunks (batches)
    for i in range(0, len(parsed_data_list), batch_size):
        # Get the current batch of cell data
        batch_data = parsed_data_list[i : i + batch_size]

        # Format the batch data into a string for the prompt
        # We use json.dumps for clean, structured formatting
        batch_input_str = json.dumps(batch_data, indent=2)

        print(f"Processing batch {i//batch_size + 1} with {len(batch_data)} cells...")

        try:
            # Invoke the chain with the entire batch
            response_model = enrichment_chain.invoke({"batch_input": batch_input_str})

            # The response is now a Pydantic object, not a string!
            semantic_data_list = response_model.analyses

            if len(semantic_data_list) != len(batch_data):
                print(
                    f"Warning: Mismatch between batch size ({len(batch_data)}) and response count ({len(semantic_data_list)}). Skipping batch."
                )
                continue

            # Correlate the results back to the original batch data
            for original_cell, semantic_data in zip(batch_data, semantic_data_list):
                page_content = (
                    f"Concept: {semantic_data.concept_name}. "
                    f"Category: {semantic_data.concept_category}. "
                    f"Function: {semantic_data.functional_type}. "
                    f"Description: {semantic_data.explanation}"
                )

                metadata = {
                    "source": original_cell.get("filename"),
                    "sheet_name": original_cell.get("sheet_name"),
                    "cell_address": original_cell.get("cell_address"),
                    "formula": original_cell.get("formula"),
                    "concept_name": semantic_data.concept_name,
                    "concept_category": semantic_data.concept_category,
                    "explanation": semantic_data.explanation,
                    "functional_type": semantic_data.functional_type,
                }

                doc = Document(page_content=page_content, metadata=metadata)
                langchain_documents.append(doc)

        except Exception as e:
            print(f"An error occurred during batch processing: {e}")

        # Add a small delay to be extra safe with rate limits
        time.sleep(2)

    return langchain_documents
