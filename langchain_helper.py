from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ NEW (Correct)


from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings  # Using InstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables (for Google API Key)
load_dotenv()

# ✅ Use Google Gemini instead of deprecated Palm
api_key = os.getenv("GOOGLE_API_KEY")  # Fetch API key from .env file
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# ✅ FIX: Use HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# FAISS vector database file path
vectordb_file_path = "faiss_index"


def create_vector_db():
    """Creates and saves a FAISS vector database from the CSV data."""
    print("Loading CSV data...")
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt")
    data = loader.load()

    print("Creating FAISS vector store...")
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

    print("Saving FAISS index locally...")
    vectordb.save_local(vectordb_file_path)
    print("Vector database created successfully!")


def get_qa_chain():
    """Loads FAISS vector DB and sets up a LangChain RetrievalQA model."""
    print("Loading FAISS vector store...")
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Custom prompt template
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the 'response' section in the source document context without making many changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # ✅ FIX: Use HuggingFaceInstructEmbeddings and Gemini LLM
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

    return chain


if __name__ == "__main__":
    # Create FAISS DB and test QA system
    create_vector_db()
    chain = get_qa_chain()
    response = chain.invoke("Do you have a JavaScript course?")
    print(response)
