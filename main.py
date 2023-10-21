from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from pdfminer.pdfparser import PDFSyntaxError

def main():
    try:
        # Replace "book.pdf" with the path to your PDF file
        pdf_file_path = "book.pdf"
        loader = UnstructuredPDFLoader(pdf_file_path)
        pages = loader.load_and_split()

        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(pages, embeddings).as_retriever()

        # Choose any query of your choice
        query = "Who is Rich Dad?"
        docs = docsearch.get_relevant_documents(query)

        chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        output = chain.run(input_documents=docs, question=query)
        print(output)

    except PDFSyntaxError as e:
        print(f"PDF Syntax Error: {e}")
        # Handle the PDF parsing error as needed

if __name__ == "__main__":
    main()
