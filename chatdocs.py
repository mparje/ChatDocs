import os
import streamlit as st
from langchain import OpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    QuestionAnswerPrompt, 
)

# Retrieve the OpenAI API key from an environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load documents
documents = SimpleDirectoryReader("documents").load_data()

# Create LLMPredictor and ServiceContext objects
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# Create and save the index
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
index.save_to_disk('index.json')

# Load the index
index = GPTSimpleVectorIndex.load_from_disk('index.json', llm_predictor=llm_predictor)

# Define the question prompt template
QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n" 
    "Given this information, please answer the question: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

def run_streamlit_app():
    st.title("Question Answering System")

    # Display the input form to enter the question
    query_str = st.text_input("Enter your question")

    if st.button("Submit"):
        # Query the index and get the response
        response = index.query(query_str, text_qa_template=QA_PROMPT)

        # Display the response
        st.markdown(f"**Response:** {response}")

if __name__ == "__main__":
    run_streamlit_app()
