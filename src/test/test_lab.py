"""
This file will contain test cases for the automatic evaluation of your
solution in lab/lab.py. You should not modify the code in this file. You should
also manually test your solution by running main/app.py.
"""
import os
import unittest

from langchain.schema.runnable.base import RunnableSequence
from langchain.llms import HuggingFaceEndpoint
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from src.main.lab import get_basic_chain, basic_chain_invoke


class TestLLMResponse(unittest.TestCase):
    """
    This test will verify that the connection to an external LLM is made. If it does not
    work, this may be because the API key is invalid, or the service may be down.
    If that is the case, this lab may not be completable.
    """

    def test_llm_sanity_check(self):
        llm = HuggingFaceEndpoint(
        endpoint_url=os.environ['HF_ENDPOINT'],
        huggingfacehub_api_token=os.environ['HF_TOKEN'],
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 1024
        }
    )

    """
    The variable returned from the lab function should be an langchain AI response. If this test
    fails, then the AI message request either failed, or you have not properly configured the lab function
    to return the result of the LLM chat.
    """

    def test_return_type_basic_chain(self):
        chain = get_basic_chain()
        self.assertIsInstance(chain, RunnableSequence)
    
    def test_basic_chain_relevancy(self):
        result = basic_chain_invoke("honey bees")
        self.assertIsInstance(result, str)
        self.assertTrue(classify_relevancy(result, "Can you tell me about honey bees?"))
    
   
def classify_relevancy(message, question):
    prompt_template = PromptTemplate.from_template(
        """
        <|system|>
        You are a chatbot who determines if a given message properly answers a question by replying "yes" or "no".</s>
        <|user|>
        Does the following message answer the question: {question}? message: {message}</s>
        <|assistant|>
        """
    )

    model = HuggingFaceEndpoint(
        endpoint_url=os.environ['HF_ENDPOINT'],
        huggingfacehub_api_token=os.environ['HF_TOKEN'],
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 1024
        }
    )

    chain = prompt_template | model | StrOutputParser()

    result = chain.invoke({"message": message, "question": question})
    print("Result: " + result)
    print(message)
    print(question)
    if ("yes" in result.lower()):
        return True
    else:
        print(message)
        return False

if __name__ == '__main__':
    unittest.main()