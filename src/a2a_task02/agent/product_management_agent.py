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
    ChatOptions,  # needed for response_format in agent_framework 1.0.0b260210
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
# end region (Chat Service Configuration)


# The two agents that are going to work as tools are creative in nature: 
# they do not acquire factual information but instead generate responses 
# based on the instructions you provided. 
# In practice, many agents will need to access specific information 
# or perform particular tasks. To facilitate this, 
#functions can be created that encapsulate specific functionalities. 
# Here below we will create a function that allows an agent to look up product information 
# from a predefined list.


# region Get Products

# @ai_function(  # ai_function renamed to tool in agent_framework 1.0.0b260210
#     name='get_products',
#     description='Retrieves a set of products based on a natural language user query.'
# )
@tool(name='get_products', description='Retrieves a set of products based on a natural language user query.')
def get_products(
    # self,  # removed: get_products is a standalone function, not a method — 'self' caused repeated call failures
    question: Annotated[
        str, 'Natural language query to retrieve products, e.g. "What kinds of paint rollers do you have in stock?"'
    ],
) -> list[dict[str, Any]]:
    try:
        # Simulate product retrieval based on the question
        # In a real implementation, this would query a database or external service
        product_dict = [
            {
                "id": "1",
                "name": "Eco-Friendly Paint Roller",
                "type": "Paint Roller",
                "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
                "punchLine": "Roll with the best, paint with the rest!",
                "price": 15.99
            },
            {
                "id": "2",
                "name": "Premium Paint Brush Set",
                "type": "Paint Brush",
                "description": "A set of premium paint brushes for detailed work and fine finishes.",
                "punchLine": "Brush up your skills with our premium set!",
                "price": 25.49
            },
            {
                "id": "3",
                "name": "All-Purpose Paint Tray",
                "type": "Paint Tray",
                "description": "A durable paint tray suitable for all types of rollers and brushes.",
                "punchLine": "Tray it, paint it, love it!",
                "price": 9.99
            }
        ]
        return product_dict
    except Exception as e:
        return f'Product recommendation failed: {e!s}'


# end region (Get Products)

# The code immediately above defines a function, get_products, 
# which simulates retrieving product information based on a natural language query. In a real-world scenario, this method would likely query a database or an external service to fetch relevant product data, but for the sake of simplicity, it returns a hardcoded list of products.

# # Define a response format model:
# region Response Format
class ResponseFormat(BaseModel):
    """A Response Format model to direct how the model should respond."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


# endregion (Response Format)

# create the Product Management Agent
# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework-based agents to handle Zava product management tasks."""

    # agent: ChatAgent  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
    agent: RawAgent
    # thread: AgentThread = None  # single thread doesn't preserve history across sessions
    threads: dict[str, AgentThread] = None  # store threads by session_id for conversation history
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        # Configure the chat completion service explicitly
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)
        
        ## Define two new agents. They do not make use of any special context 
        ## or information, so they are defined with only a name and instructions. 
        
        # Define an MarketingAgent to handle marketing-related tasks
        # marketing_agent = ChatAgent(  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
        #     chat_client=chat_service,  # param renamed to 'client'
        marketing_agent = RawAgent(
            client=chat_service,
            name='MarketingAgent',
            instructions=(
                'You specialize in planning and recommending marketing strategies for products. '
                'This includes identifying target audiences, making product descriptions better, and suggesting promotional tactics. '
                'Your goal is to help businesses effectively market their products and reach their desired customers.'
            ),
        )

        # Define an RankerAgent to sort and recommend results
        # ranker_agent = ChatAgent(  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
        #     chat_client=chat_service,  # param renamed to 'client'
        ranker_agent = RawAgent(
            client=chat_service,
            name='RankerAgent',
            instructions=(
                'You specialize in ranking and recommending products based on various criteria. '
                'This includes analyzing product features, customer reviews, and market trends to provide tailored suggestions. '
                'Your goal is to help customers find the best products for their needs.'
            ),
        )
        # Define a ProductAgent to retrieve products from the Zava catalog
        # product_agent = ChatAgent(  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
        #     chat_client=chat_service,  # param renamed to 'client'
        product_agent = RawAgent(
            client=chat_service,
            name='ProductAgent',
            instructions=("""
                You are a product catalog lookup assistant. You have access to a get_products tool
                that returns the COMPLETE list of available products in the Zava catalog.

                STRICT RULES — NEVER VIOLATE THESE:
                1. ALWAYS call the get_products tool BEFORE answering. No exceptions.
                2. Your answer MUST contain ONLY the fields returned by get_products (name, type, description, punchLine, price).
                3. NEVER invent, guess, or add any information that is NOT explicitly present in the get_products result.
                   This means: do NOT mention missing fields, do NOT note what is unavailable, do NOT speculate about sizes, surfaces, stock, SKU, quantities, or any other detail not in the data.
                4. NEVER use your own knowledge about products. You know NOTHING except what get_products returns.
                5. If the user asks about a product that is NOT in the get_products result, simply say:
                   "That product is not available in our catalog."
                6. When asked "how many" of something, count ONLY matching items from get_products.
                7. Present only the data fields returned by get_products. Nothing more.
                8. Do NOT ask the user follow-up questions. Do NOT offer to fetch more details, reserve, or purchase.
                9. Do NOT add notes, caveats, disclaimers, or commentary about missing information.
                10. Keep your answer concise: list the matching product(s) with their available details and stop.
                """
            ),
            tools=get_products,
        )
        # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        # self.agent = ChatAgent(  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
        # self.agent = RawAgent(
        #     # chat_client=chat_service,  # param renamed to 'client' in agent_framework 1.0.0b260210
        #     client=chat_service,
        #     name='ProductManagerAgent',
        #     instructions=(
        #         "Your role is to carefully analyze the user's request and respond as best as you can. "
        #         'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized '
        #         'assistance promptly.\n\n'
        #         'IMPORTANT: You must ALWAYS respond with a valid JSON object in the following format:\n'
        #         '{"status": "<status>", "message": "<your response>"}\n\n'
        #         'Where status is one of: "input_required", "completed", or "error".\n'
        #         '- Use "input_required" when you need more information from the user.\n'
        #         '- Use "completed" when the task is finished.\n'
        #         '- Use "error" when something went wrong.\n\n'
        #         'Never respond with plain text. Always use the JSON format above.'
        #     ),
        #     ## tools
        #     # tools=[],
        #     # Product Manager Agent will be  delegate tasks to the Marketing Agent and Ranker Agent as needed.
        #     tools=[marketing_agent.as_tool(), ranker_agent.as_tool()],
        # )
           # Define the main ProductManagerAgent to delegate tasks to the appropriate agents
        # self.agent = ChatAgent(  # ChatAgent renamed to RawAgent in agent_framework 1.0.0b260210
        #     chat_client=chat_service,  # param renamed to 'client'
        self.agent = RawAgent(
            client=chat_service,
            name='ProductManagerAgent',
            # instructions=(  # original instructions — too ambiguous, "product description" routed to ProductAgent instead of MarketingAgent
            #     "Your role is to carefully analyze the user's request and respond as best as you can. "
            #     'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized assistance promptly.'
            #     'Whenever a user query is related to retrieving product information, you MUST delegate the task to the ProductAgent.'
            #     'Use the MarketingAgent for marketing-related queries and the RankerAgent for product ranking and recommendation tasks.'
            #     'You may use these agents in conjunction with each other to provide comprehensive responses to user queries.'
            # ),
            instructions=(
                "Your role is to carefully analyze the user's request and delegate to the most appropriate agent. "
                'Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized assistance promptly.\n\n'
                'DELEGATION RULES:\n'
                '- MarketingAgent: Use for requests to IMPROVE, REWRITE, or CREATE product descriptions, slogans, or marketing copy. '
                'Do NOT look up the product first — pass the request directly to the MarketingAgent.\n'
                '- RankerAgent: Use for requests to RANK, COMPARE, or RECOMMEND products.\n'
                '- ProductAgent: Use ONLY when the user wants to SEARCH, LIST, or LOOK UP products from the catalog.\n\n'
                'You may chain agents when needed (e.g., ProductAgent to retrieve data, then MarketingAgent to improve a description).\n'
            ),
            tools=[product_agent.as_tool(), marketing_agent.as_tool(), ranker_agent.as_tool()],
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
        thread = self._ensure_thread_exists(session_id)

        # Use Agent Framework's run for a single shot
        # response = await self.agent.run(  # response_format not a direct param in agent_framework 1.0.0b260210
        #     messages=user_input,
        #     thread=self.thread,
        #     response_format=ResponseFormat,
        # )
        response = await self.agent.run(
            messages=user_input,
            thread=thread,
            options=ChatOptions(response_format=ResponseFormat),  # pass via ChatOptions in 1.0.0b260210
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
        thread = self._ensure_thread_exists(session_id)

        # text_notice_seen = False
        # chunks: list[ChatContent] = []  # ChatContent not in agent_framework 1.0.0b260210; using str
        chunks: list[str] = []

        async for chunk in self.agent.run_stream(
            messages=user_input,
            # thread=self.thread,  # changed to use per-session thread dict
            thread=thread,
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
        # structured_response = ResponseFormat.model_validate_json(  # moved below with try/except guard
        #     message
        # )

        default_response = {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

        # Handle case where LLM returns plain text instead of JSON
        if not isinstance(message, str) or not message.strip().startswith('{'):
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': str(message),
            }

        try:
            structured_response = ResponseFormat.model_validate_json(message)
        except Exception:
            # Fallback: treat plain-text LLM response as a completed response
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': str(message),
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

    # async def _ensure_thread_exists(self, session_id: str) -> None:  # original: single thread, no history
    def _ensure_thread_exists(self, session_id: str) -> AgentThread:
        """Ensure the thread exists for the given session ID.
        Returns the thread for the session, creating one if needed.

        Args:
            session_id (str): Unique identifier for the session.

        Returns:
            AgentThread: The thread for the given session.
        """
        if self.threads is None:
            self.threads = {}
        if session_id not in self.threads:
            self.threads[session_id] = self.agent.get_new_thread(thread_id=session_id)
        return self.threads[session_id]
        # if self.thread is None or self.thread.service_thread_id != session_id:
        #     self.thread = self.agent.get_new_thread(thread_id=session_id)


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

