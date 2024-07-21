import json
import gradio as gr

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Create the prompte from the template.
promptTemplate = """Answer the question as precise as possible using the provided context. If the answer is
    not contained in the context, say "answer not available in context" \n\n
    Context: {context}
    Question: {question}
    Answer:

     """
modelSel = ""


# Load the PDF file to ChromaDB
def loadDataFromPDFFile(filePath):
    loader = PyPDFLoader(filePath)
    pages = loader.load_and_split()
    chunks = filter_complex_metadata(pages)

    # Use Ollama's Embedding
    embedding_model = OllamaEmbeddings()

    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    return vector_store


def modelResponse(message, history):
    llm = ChatOllama(model=conf["model"])

    prompt = PromptTemplate(
        template=promptTemplate, input_variables=["context", "question"]
    )

    # Initiate the retriever
    dbLoaded = loadDataFromPDFFile("./c2n.pdf")
    retriever = dbLoaded.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2},
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
