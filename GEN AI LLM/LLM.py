from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# Loading data info
loader = TextLoader(r"C:\Users\asush\Downloads\plants_diseases.txt") 
documents = loader.load()

# Embedding
embeddings = OpenAIEmbeddings()

# LLM config
llm = OpenAI(
    temperature=0.7,
  
)

# Create a RetrievalQA chain with OpenAI LLM and Chroma vectorstore
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Use "stuff" for a simple approach
    retriever=vectorstore.as_retriever(),
)

# Function to get plant disease information from user input
def get_disease_info(disease_name):
    """
    Gets information about a plant disease, including symptoms, causes, prevention, and treatment.

    Args:
        disease_name (str): The name of the disease.

    Returns:
        str: A string containing information about the disease.
    """
    query = f"What are the symptoms, causes, prevention, and treatment for {disease_name}?"
    
    result = qa_chain.invoke({"query": query})  

    # Extract the result from dictionary
    return result['result']  

# Get disease information from user input
if __name__ == "__main__":
    disease_name = input("Enter the name of the plant disease: ")
    disease_info = get_disease_info(disease_name)
    print("\nDisease Information:")
    print(disease_info)
