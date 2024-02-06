"""
This file will contain test cases for the automatic evaluation of your
solution in lab/lab.py. You should not modify the code in this file. You should
also manually test your solution by running main/app.py.
"""
import unittest

from langchain.schema.runnable.base import RunnableSequence
from langchain_core.outputs import LLMResult
from src.main.lab import get_basic_chain, basic_chain_invoke
from src.utilities.llm_testing_util import classify_relevancy, llm_wakeup, llm_connection_check


class TestLLMResponse(unittest.TestCase):
    """
    This function is a sanity check for the Language Learning Model (LLM) connection.
    It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
    it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
    operational before running tests and should not be modified without understanding the
    implications.
    Raises:
        Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
    """
    def test_llm_sanity_check(self):

        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")

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


if __name__ == '__main__':
    unittest.main()
