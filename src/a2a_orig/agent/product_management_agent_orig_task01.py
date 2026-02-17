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
    ChatAgent,
    BaseChatClient,
    ai_function,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework.azure import AzureOpenAIChatClient

logger = logging.getLogger(__name__)
load_dotenv()


# region Chat Service Configuration

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


# endregion


# region Response Format


class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# endregion


# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    agent: ChatAgent
    thread: AgentThread = None
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)

        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        self.agent = ChatAgent(
            chat_client=chat_service,
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
        chunks: list[ChatContent] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            thread=self.thread,
        ):
            if chunk.text:
                chunks.append(chunk.text)

        if chunks:
            yield self._get_agent_response(sum(chunks[1:], chunks[0]))

    def _get_agent_response(
        self, message: ChatContent
    ) -> dict[str, Any]:
        """Extracts the structured response from the agent's message content.

        Args:
            message (ChatContent): The message content from the agent.

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
