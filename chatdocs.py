import os
from langchain import OpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    QuestionAnswerPrompt,
)

os.environ["OPENAI_API_KEY"] = "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"

documents = SimpleDirectoryReader("documents").load_data()

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))