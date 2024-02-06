from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFaceEndpoint

import os

# TODO: Complete this prompt to ask the model for general information on a {topic}:
prompt_template = "Tell me some general information on: {topic} in under 100 words."
prompt = ChatPromptTemplate.from_template(prompt_template)

model = HuggingFaceEndpoint(
    endpoint_url=os.environ['HF_ENDPOINT'],
    huggingfacehub_api_token=os.environ['HF_TOKEN'],
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 1024
    }
)


# Use a simple output parser that converts output to a string
output_parser = StrOutputParser()

# TODO: Create/return a chain using the prompt, model, and output_parser
# Make sure you use LCEL to achieve this. 
# Hint: The function body can be as short as a single line
def get_basic_chain():
    chain = prompt | model | output_parser
    return chain

# Using the chain created in basic_chain, invoke the chain with a topic.
# PLEASE DO NOT edit this function
def basic_chain_invoke(topic):
    chain = get_basic_chain()
    try:
        response = chain.invoke({"topic": topic})
    except Exception as e:
        return "Something went wrong: {}".format(e)
    return response
