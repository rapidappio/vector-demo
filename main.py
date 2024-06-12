import os

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():

    # See docker command above to launch a postgres instance with pgvector enabled.
    connection = os.getenv('DB_URL') # Uses psycopg3!
    collection_name = "my_docs"
    embeddings = OllamaEmbeddings()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
    ##vectorstore.drop_tables()

    urls = ["https://prometheus.io/docs/prometheus/latest/getting_started/", "https://prometheus.io/docs/prometheus/latest/federation/"]
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    htmlToText = Html2TextTransformer()
    docs_transformed = htmlToText.transform_documents(docs)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    docs = splitter.split_documents(docs_transformed)
    vectorstore.add_documents(docs)

    retriever = vectorstore.as_retriever()

    llm = Ollama()
    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    response = rag_chain.invoke("how to federate on prometheus")
    print(response)


if __name__ == "__main__":
    main()