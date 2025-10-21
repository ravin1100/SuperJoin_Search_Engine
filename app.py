import streamlit as st
import os
from typing import List, Dict, Any

# Import the core logic functions from your backend modules.
# This clean separation of concerns makes the app easier to manage.
from core.parser import parse_spreadsheet
from core.enricher import create_langchain_documents
from core.vector_store import add_documents, delete_documents, get_retriever

# --- Helper Functions ---


def format_results(documents: List[Any]) -> str:
    """
    Formats the search results from the vector store into a human-readable Markdown string.

    Args:
        documents: A list of LangChain Document objects returned by the retriever.

    Returns:
        A formatted string ready for display in the Streamlit UI.
    """
    if not documents:
        return "I couldn't find any relevant calculations for your query. Please try asking in a different way."

    # Build the response string using Markdown for better formatting.
    response_parts = [f"**Found {len(documents)} relevant results:**\n\n---\n"]
    for doc in documents:
        # Access the metadata stored alongside the vector.
        metadata = doc.metadata

        # Safely get each piece of metadata with a default value.
        concept = metadata.get("concept_name", "N/A")
        category = metadata.get("concept_category", "N/A")
        filename = metadata.get("source", "N/A")
        sheet = metadata.get("sheet_name", "N/A")
        cell = metadata.get("cell_address", "N/A")
        formula = metadata.get("formula", "N/A")
        explanation = metadata.get("explanation", "No explanation available.")

        # Append the formatted details for each result.
        response_parts.append(
            f"### {concept}\n"
            f"- **Category:** `{category}`\n"
            f"- **Explanation:** {explanation}\n"
            f"- **Location:** `{filename}` | `{sheet}`!**{cell}**\n"
            f"- **Formula:** `{formula}`\n\n---\n"
        )

    return "".join(response_parts)


# --- Streamlit Page Configuration ---

# Set the page title, icon, and layout. `wide` layout gives more space.
st.set_page_config(
    page_title="Superjoin Semantic Search 🧠", page_icon="🔍", layout="wide"
)

# --- Session State Initialization ---
# Session state is used to store variables that need to persist across reruns,
# such as chat history and the list of indexed files.

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()  # Using a set to avoid duplicate filenames.

# --- Sidebar UI for File Management ---

with st.sidebar:
    st.header("📄 File Management")
    st.write(
        "Upload your spreadsheets and manage the knowledge base for the search engine."
    )

    # File uploader allows multiple .xlsx files.
    uploaded_files = st.file_uploader(
        "Upload Spreadsheets",
        type=["xlsx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Index Files", use_container_width=True, type="primary"):
        if uploaded_files:
            # Process each uploaded file.
            for file in uploaded_files:
                # To avoid re-indexing the same file content if uploaded again.
                if file.name not in st.session_state.indexed_files:
                    with st.spinner(f"Processing {file.name}..."):
                        # 1. Read file content.
                        file_content = file.getvalue()

                        # 2. Parse the spreadsheet to get formula contexts.
                        parsed_data = parse_spreadsheet(file_content, file.name)

                        if not parsed_data:
                            st.warning(f"No formulas found in {file.name}. Skipping.")
                            continue

                        # 3. Enrich the data with the LLM to create LangChain Documents.
                        documents = create_langchain_documents(parsed_data)

                        print("++++++++++++++++")
                        print(documents)
                        print("++++++++++++++++")

                        # 4. Add the enriched documents to the vector store.
                        add_documents(documents)

                        # 5. Update session state and show success message.
                        st.session_state.indexed_files.add(file.name)
                        st.success(f"Successfully indexed {file.name}!")
                else:
                    st.info(f"{file.name} is already indexed.")
        else:
            st.warning("Please upload at least one file to index.")

    st.divider()

    # UI for unindexing files.
    if st.session_state.indexed_files:
        st.subheader("Indexed Files")

        # Create a list from the set for the selectbox.
        files_list = list(st.session_state.indexed_files)
        file_to_unindex = st.selectbox("Select a file to unindex:", files_list)

        if st.button("Unindex File", use_container_width=True):
            with st.spinner(f"Unindexing {file_to_unindex}..."):
                # Call the delete function from the vector store module.
                delete_documents(file_to_unindex)

                # Update session state and show a success message.
                st.session_state.indexed_files.remove(file_to_unindex)
                st.success(f"Successfully unindexed {file_to_unindex}!")
                st.rerun()  # Rerun the app to update the selectbox.

# --- Main Chat Interface ---

st.title("Superjoin Semantic Search 🧠")
st.write(
    "Ask questions in natural language about the concepts and calculations in your indexed spreadsheets."
)

# Display the chat history.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat box at the bottom of the screen.
if query := st.chat_input(
    "e.g., 'Find all profitability metrics' or 'Show me cost calculations'"
):
    # Add user's message to history and display it.
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Process the query and get the AI's response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Get the retriever from the vector store.
            retriever = get_retriever()

            # 2. Invoke the retriever to find relevant documents.
            # This performs the core semantic search in Chroma Cloud.
            relevant_docs = retriever.invoke(query)

            # 3. Format the results into a readable response.
            response = format_results(relevant_docs)

            # 4. Display the response.
            st.markdown(response)

    # Add the AI's response to the chat history.
    st.session_state.chat_history.append({"role": "assistant", "content": response})
