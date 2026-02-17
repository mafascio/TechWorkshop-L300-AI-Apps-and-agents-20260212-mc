# # # Implement the Product Management Agent # # #
# This agent will handle product-related queries and interact with the A2A server.


# #  import and load statements
import asyncio
import logging
import os
from collections.abc import AsyncIterable
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal
import httpx
import openai
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import BaseModel
from agent_framework import (
    AgentThread,
    ChatContext,
    # ChatAgent,  # renamed to RawAgent in agent_framework 1.0.0b260210
    RawAgent,
    BaseChatClient,
    # ai_function,  # renamed to tool in agent_framework 1.0.0b260210
    tool,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient

logger = logging.getLogger(__name__)
load_dotenv()

# # service configuration code
# 
# This code sets up the configuration for connecting to either Azure OpenAI or OpenAI services, depending on the environment variables provided.
# We'll  Azure OpenAI, but the code supports both services. 
# The _get_azure_openai_chat_completion_service() function uses the GPT deployment details
# 

class ChatServices(str, Enum):
    """Enum for supported chat completion services."""

    AZURE_OPENAI = 'azure_openai'
    OPENAI = 'openai'


service_id = 'default'


def get_chat_completion_service(
    service_name: ChatServices,
) -> 'BaseChatClient':
    """Return an appropriate chat completion service based on the service name.

    Args:
        service_name (ChatServices): Service name.

    Returns:
        BaseChatClient: Configured chat completion service.

    Raises:
        ValueError: If the service name is not supported or required environment variables are missing.
    """
    if service_name == ChatServices.AZURE_OPENAI:
        return _get_azure_openai_chat_completion_service()
    if service_name == ChatServices.OPENAI:
        return _get_openai_chat_completion_service()
    raise ValueError(f'Unsupported service name: {service_name}')


def _get_azure_openai_chat_completion_service() -> AzureOpenAIChatClient:
    """Return Azure OpenAI chat completion service with managed identity.

    Returns:
        AzureOpenAIChatClient: The configured Azure OpenAI service.
    """
    endpoint = os.getenv('gpt_endpoint')
    deployment_name = os.getenv('gpt_deployment')
    api_version = os.getenv('gpt_api_version')
    api_key = os.getenv('gpt_api_key')

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")
    if not api_version:
        raise ValueError("gpt_api_version is required")

    # Use managed identity if no API key is provided
    if not api_key:
        # Create Azure credential for managed identity
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        
        # Create OpenAI client with managed identity
        async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        
        return AzureOpenAIChatClient(
            service_id=service_id,
            deployment_name=deployment_name,
            async_client=async_client,
        )
    else:
        # Fallback to API key authentication for local development
        return AzureOpenAIChatClient(
            service_id=service_id,
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )

def _get_openai_chat_completion_service() -> OpenAIChatClient:
    """Return OpenAI chat completion service.

    Returns:
        OpenAIChatClient: Configured OpenAI service.
    """
    return OpenAIChatClient(
        service_id=service_id,
        model_id=os.getenv('OPENAI_MODEL_ID'),
        api_key=os.getenv('OPENAI_API_KEY'),
    )


# # code to define a response format model:
# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# endregion

# create the Product Management Agent
# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    # agent: ChatAgent  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
    agent: RawAgent
    thread: AgentThread = None
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)

        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        # self.agent = ChatAgent(  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
        self.agent = RawAgent(
            # chat_client=chat_service,  # param renamed to 'client' in agent_framework 1.0.0b260210
            client=chat_service,
            name='ProductManagerAgent',
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized '
                'assistance promptly.\n\n'
                'IMPORTANT: You must ALWAYS respond with a valid JSON object in the following format:\n'
                '{"status": "<status>", "message": "<your response>"}\n\n'
                'Where status is one of: "input_required", "completed", or "error".\n'
                '- Use "input_required" when you need more information from the user.\n'
                '- Use "completed" when the task is finished.\n'
                '- Use "error" when something went wrong.\n\n'
                'Never respond with plain text. Always use the JSON format above.'
            ),
            tools=[],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        """Handle synchronous tasks (like tasks/send).

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Returns:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_thread_exists(session_id)

        # Use Agent Framework's run for a single shot
        response = await self.agent.run(
            messages=user_input,
            thread=self.thread,
            response_format=ResponseFormat,
        )
        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        """For streaming tasks we yield the Agent Framework agent's run_stream progress.

        Args:
            user_input (str): User input message.
            session_id (str): Unique identifier for the session.

        Yields:
            dict: A dictionary containing the content, task completion status,
            and user input requirement.
        """
        await self._ensure_thread_exists(session_id)

        # text_notice_seen = False
        # chunks: list[ChatContent] = []  # ChatContent not in agent_framework 1.0.0b260210; using str
        chunks: list[str] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            thread=self.thread,
        ):
            if chunk.text:
                chunks.append(chunk.text)

        if chunks:
            yield self._get_agent_response(sum(chunks[1:], chunks[0]))

    def _get_agent_response(
        # self, message: ChatContent  # ChatContent not in agent_framework 1.0.0b260210; using str
        self, message: str
    ) -> dict[str, Any]:
        """Extracts the structured response from the agent's message content.

        Args:
            message (str): The message content from the agent.

        Returns:
            dict: A dictionary containing the content, task completion status, and user input requirement.
        """
        structured_response = ResponseFormat.model_validate_json(
            message
        )

        default_response = {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

        if isinstance(structured_response, ResponseFormat):
            response_map = {
                'input_required': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'error': {
                    'is_task_complete': False,
                    'require_user_input': True,
                },
                'completed': {
                    'is_task_complete': True,
                    'require_user_input': False,
                },
            }

            response = response_map.get(structured_response.status)
            if response:
                return {**response, 'content': structured_response.message}

        return default_response

    async def _ensure_thread_exists(self, session_id: str) -> None:
        """Ensure the thread exists for the given session ID.

        Args:
            session_id (str): Unique identifier for the session.
        """
        if self.thread is None or self.thread.service_thread_id != session_id:
            self.thread = self.agent.get_new_thread(thread_id=session_id)


# endregion

# The __init__ method initializes the agent with specific instructions and an 
# empty list of plugins, as you will not be implementing additional agents in 
# this task. It defines one ChatCompletionAgent using the label self.agent. 
# This default agent will serve as the entryway for all incoming requests, 
# as the other agents will not directly receive requests.

# Next, the invoke method is responsible for handling synchronous tasks. 
# It ensures that a chat thread exists for the given session ID and then uses 
# the agent to get a response based on the user’s input. 
# The response is processed to extract relevant information, such as
# whether the task is complete and if further user input is required.

# The stream method handles streaming tasks, yielding progress updates as 
# the agent processes the user’s input. It also ensures that a chat thread 
# exists for the session ID and uses the agent to invoke a streaming response.

# The _get_agent_response method extracts structured responses from 
# the agent’s message content, mapping them to a dictionary format that 
# includes task completion status and user input requirements.

# The _ensure_thread_exists method ensures that a chat thread is 
# created or reused based on the session ID.

