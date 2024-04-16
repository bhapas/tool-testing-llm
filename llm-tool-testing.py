"""Main entrypoint for the sei-llm CLI."""

import os
from operator import itemgetter
from typing import Union

import yaml
from dotenv import load_dotenv
from langchain.agents import tool
from langchain.globals import set_debug, set_verbose
from langchain.output_parsers import JsonOutputToolsParser
from langchain_core.runnables import (Runnable, RunnableLambda,
                                      RunnablePassthrough)
from langchain_openai import AzureChatOpenAI

set_debug(True)
set_verbose(True)


@tool
def ecs_mapping_function(ecs_key: str) -> str:
    """Returns a description for the provided ecs_key.

    Args:
        ecs_key: The full path to the ECS field using dot-notations, example "event.action"
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output = os.path.join(dir_path, "output.yml")

    with open(output, "r", encoding="utf-8") as file:
        ecs_data = yaml.safe_load(file)
    return ecs_data[ecs_key]


load_dotenv()

llm = AzureChatOpenAI(  # type: ignore
    api_version="2024-02-15-preview",
    model_version="1106-Preview",
    model="gpt-4-turbo",
    azure_deployment="seic-gpt4-turbo",
    temperature=0.05,
    max_tokens=4096,
    model_kwargs={
        "top_p": 0.4,
        "stop": ["Human:"],
    },
)
tools = [ecs_mapping_function]
model = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}


def call_tool(tool_invocation: dict) -> Union[str, Runnable]:
    """Function for dynamically constructing the end of the chain based on the model-selected tool."""
    tool = tool_map[tool_invocation["type"]]
    # type: ignore
    return RunnablePassthrough.assign(output=itemgetter("args") | tool)


call_tool_list = RunnableLambda(call_tool).map()
chain = model | JsonOutputToolsParser() | call_tool_list


data = chain.invoke(input="""
    You are a Elastic Common Schema expert, that answers questions about the schema.
    What is the 'network.type' field used for?
    """)
