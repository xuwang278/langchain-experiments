import json
import gradio as gr

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Create the prompt from the template.
promptTemplate = """Answer the question as precise as possible using the provided context. If the answer is
    not contained in the context, say "answer not available in context" \n\n
    Context: {context}
    Question: {question}
    Answer:

     """
modelSel = ""


# Load the PDF file to ChromaDB
import time


def loadDataFromPDFFile(filePath):
    print("Loading PDF...")
    loader = PyPDFLoader(filePath)
    pages = loader.load_and_split()
    print(f"Number of pages: {len(pages)}")
    chunks = filter_complex_metadata(pages)
    print(f"Number of chunks: {len(chunks)}")

    for doc in chunks[:5]:  # Print vectors for the first 5 documents
        print(doc)  # Ensure embeddings are not all zero or meaningless

    embedding_model = OllamaEmbeddings()
    print("Creating vector store...")
    start_time = time.time()
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    print(f"Vector store created in {time.time() - start_time:.2f} seconds.")
    return vector_store


def modelResponse(message, history):
    llm = ChatOllama(model=conf["model"])

    prompt = PromptTemplate(
        template=promptTemplate, input_variables=["context", "question"]
    )

    # Initiate the retriever
    dbLoaded = loadDataFromPDFFile("./c2n_short2.pdf")
    retriever = dbLoaded.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 2, "score_threshold": 0.001},
    )

    hpChain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return hpChain.invoke(message)


if __name__ == "__main__":
    # read configuration file
    conf = {}
    with open("config.json", "r") as confFile:
        conf = json.load(confFile)
        print(conf["model"])

    chatUI = gr.ChatInterface(fn=modelResponse, title="C2N")
    chatUI.launch()
