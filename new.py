import json
from typing import Sequence, List
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
import nest_asyncio
nest_asyncio.apply()


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

llm = MistralAI(model="mistral-large-latest")


from llama_index.core.agent import FunctionCallingAgent

# enable parallel function calling
agent = FunctionCallingAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)

response = agent.chat("What is (121 + 2) * 5?")
print(str(response))

# inspect sources
print(response.sources)



import asyncio

async def main():
    response = await agent.achat("What is (121 * 3) + (5 * 8)?")
    print(str(response))

asyncio.run(main())


