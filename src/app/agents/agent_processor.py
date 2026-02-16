import os
import re
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

# Ensure src/ is on the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from azure.ai.projects.models import FunctionTool
from openai.types.responses.response_input_param import FunctionCallOutput, ResponseInputParam
from app.servers.mcp_inventory_client import MCPShopperToolsClient, get_mcp_client

from opentelemetry import trace
from azure.monitor.opentelemetry import configure_azure_monitor
from azure.ai.agents.telemetry import trace_function
# from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

# # Enable Azure Monitor tracing
application_insights_connection_string = os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
# configure_azure_monitor(connection_string=application_insights_connection_string)
# OpenAIInstrumentor().instrument()

# scenario = os.path.basename(__file__)
# tracer = trace.get_tracer(__name__)

# Increase thread pool size for better concurrency
_executor = ThreadPoolExecutor(max_workers=8)

# Cache for toolset configurations to avoid repeated initialization
_toolset_cache: Dict[str, List[FunctionTool]] = {}

_mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp-inventory/sse")


# ---------------------------------------------------------------------------
# Helper: run an async MCP call synchronously (used inside ThreadPoolExecutor)
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Run an async coroutine synchronously. Safe when called from a thread
    that has no running event loop (e.g. inside ThreadPoolExecutor)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Should not happen inside executor threads, but guard anyway
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# MCP-based tool wrapper functions (all synchronous — called from executor)
# ---------------------------------------------------------------------------

def mcp_create_image(prompt: str, image_url: str) -> str:
    """Generate an AI image based on a text description using DALL-E."""
    async def _call():
        client = await get_mcp_client(_mcp_server_url)
        return await client.call_tool(
            "generate_product_image",
            {"prompt": prompt, "image_url": image_url}
        )
    return _run_async(_call())


def mcp_product_recommendations(question: str) -> str:
    """Search for product recommendations based on user query."""
    async def _call():
        client = await get_mcp_client(_mcp_server_url)
        return await client.call_tool("get_product_recommendations", {"question": question})
    return _run_async(_call())


def mcp_calculate_discount(customer_id: str) -> str:
    """Calculate the discount based on customer data."""
    async def _call():
        client = await get_mcp_client(_mcp_server_url)
        return await client.call_tool("get_customer_discount", {"customer_id": customer_id})
    return _run_async(_call())


def mcp_inventory_check(product_list: List[str]) -> list:
    """Check inventory for products using MCP client."""
    async def _call():
        client = await get_mcp_client(_mcp_server_url)
        results = []
        for product_id in product_list:
            try:
                data = await client.check_inventory(product_id)
                # Flatten: check_inventory returns a list of dicts; extend instead of append
                # to avoid double-nesting like [[{...}]] which confuses the agent
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
            except Exception as e:
                print(f"Error checking inventory for {product_id}: {e}")
                results.append(None)
        return results
    return _run_async(_call())


# Dict-based dispatch for function calls (used by _run_conversation_sync)
_TOOL_DISPATCH: Dict[str, Any] = {
    "mcp_create_image": mcp_create_image,
    "mcp_product_recommendations": mcp_product_recommendations,
    "mcp_calculate_discount": mcp_calculate_discount,
    "mcp_inventory_check": mcp_inventory_check,
}

class AgentProcessor:
    def __init__(self, project_client, assistant_id, agent_type: str, thread_id=None):
        self.project_client = project_client
        self.agent_id = assistant_id
        self.agent_type = agent_type
        self.thread_id = thread_id
        
        # Use cached toolset or create new one
        self.toolset = self._get_or_create_toolset(agent_type)

    def _get_or_create_toolset(self, agent_type: str) -> List[FunctionTool]:
        """Get cached toolset or create new one to avoid repeated initialization."""
        if agent_type in _toolset_cache:
            return _toolset_cache[agent_type]
        
        functions = create_function_tool_for_agent(agent_type)
        
        # Cache the toolset
        _toolset_cache[agent_type] = functions
        return functions
    
    def run_conversation_with_text(self, input_message: str = ""):
        print("Running async!")
        start_time = time.time()
        openai_client = self.project_client.get_openai_client()
        thread_id = self.thread_id
        if thread_id:
            conversation = openai_client.conversations.retrieve(conversation_id=thread_id)
            openai_client.conversations.items.create(
                conversation_id=thread_id,
                items=[{"type": "message", "role": "user", "content": input_message}]
            )
        else:
            conversation = openai_client.conversations.create(
                items=[{"role": "user", "content": input_message}]
            )
            thread_id = conversation.id
            self.thread_id = thread_id
        print(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")
        messages = openai_client.responses.create(
            conversation=thread_id,
            extra_body={"agent": {"name": self.agent_id, "type": "agent_reference"}},
            input="",
            stream=True
        )
        for message in messages:
            yield message.response.output_text
        print(f"[TIMELOG] Total run_conversation_with_text time: {time.time() - start_time:.2f}s")

    def _run_conversation_sync(self, input_message: str = ""):
        """Optimized synchronous conversation runner with better error handling."""
        thread_id = self.thread_id
        start_time = time.time()
        print("Running sync!")
        
        try:
            openai_client = self.project_client.get_openai_client()
            # Create message
            if thread_id:
                print(f"Using existing thread_id: {thread_id}")
                conversation = openai_client.conversations.retrieve(conversation_id=thread_id)
                openai_client.conversations.items.create(
                    conversation_id=thread_id,
                    items=[{"type": "message", "role": "user", "content": input_message}]
                )
            else:
                print("Creating new conversation thread")
                conversation = openai_client.conversations.create(
                    items=[{"role": "user", "content": input_message}]
                )
                print("Conversation created:", conversation)
                thread_id = conversation.id
                self.thread_id = thread_id
            print(f"[TIMELOG] Message creation took: {time.time() - start_time:.2f}s")

            # Message retrieval
            message = openai_client.responses.create(
                conversation=thread_id,
                extra_body={"agent": {"name": self.agent_id, "type": "agent_reference"}},
                input="",
                stream=False
            )

            messages_start = time.time()
            print(f"[TIMELOG] Message retrieval took: {time.time() - messages_start:.2f}s")

            # Always check for function calls in the output (not just when output_text is empty)
            # The agent may return both text AND function calls in the same response
            has_function_calls = any(
                hasattr(item, 'type') and item.type == "function_call" 
                for item in message.output
            )
            
            print(f"[DEBUG] output_text length: {len(message.output_text)}, has_function_calls: {has_function_calls}")
            print(f"[DEBUG] output items: {[(item.type, getattr(item, 'name', None)) for item in message.output]}")
            
            # Track generated image URL from mcp_create_image for fallback injection
            generated_image_url = None

            if has_function_calls:
                print("[DEBUG] Function calls found in response. Executing...")
                input_list : ResponseInputParam = []
                for item in message.output:
                    if item.type == "function_call":
                        print(f"[DEBUG] Calling function: {item.name} with args: {item.arguments}")
                        # Perform the function call first, then extract final text value below
                        # Dispatch function call via dict lookup
                        handler = _TOOL_DISPATCH.get(item.name)
                        if handler:
                            func_result = handler(**json.loads(item.arguments))
                        else:
                            func_result = f"Unknown function: {item.name}"
                        print(f"[DEBUG] Function {item.name} executed with result: {func_result}")

                        # Track image URL from mcp_create_image for fallback
                        if item.name == "mcp_create_image" and func_result and isinstance(func_result, str) and func_result.startswith("http"):
                            generated_image_url = func_result
                            print(f"[DEBUG] Captured generated image URL: {generated_image_url}")

                        input_list.append(FunctionCallOutput(
                            type="function_call_output",
                            call_id=item.call_id,
                            output=json.dumps({"result": func_result})
                        ))

                # Re-run response creation to get final text output after function calls
                print("[DEBUG] Re-running response creation to get final text output after function calls.")
                message = openai_client.responses.create(
                    input=input_list,
                    previous_response_id=message.id,
                    extra_body={"agent": {"name": self.agent_id, "type": "agent_reference"}},
                )

            # Extract text output (output_text is always a string from Responses API)
            result_text = str(message.output_text)

            # Fallback: if agent returned empty text after function calls, build a
            # response from the raw function results so the user isn't left hanging
            if not result_text.strip() and has_function_calls and input_list:
                print("[DEBUG] Agent returned empty text after function calls — using fallback.")
                fallback_parts = []
                for fco in input_list:
                    try:
                        payload = json.loads(fco.output)
                        fallback_parts.append(json.dumps(payload.get("result", payload)))
                    except (json.JSONDecodeError, AttributeError):
                        fallback_parts.append(str(fco.output))
                result_text = json.dumps({"answer": "Here are the results: " + "; ".join(fallback_parts)})

            # Fallback: inject generated image URL if the agent omitted it from its response
            if generated_image_url:
                try:
                    # Extract JSON from code block or raw text
                    json_str = result_text
                    cb_match = re.search(r'```(?:json)?\s*([\[{].*[\]}])\s*```', json_str, re.DOTALL)
                    if cb_match:
                        json_str = cb_match.group(1)
                    else:
                        jm = re.search(r'([\[{].*[\]}])', json_str, re.DOTALL)
                        if jm:
                            json_str = jm.group(1)
                    parsed = json.loads(json_str)
                    target = parsed[0] if isinstance(parsed, list) and parsed else parsed
                    if isinstance(target, dict) and not target.get("image_output") and not target.get("image_url"):
                        target["image_output"] = generated_image_url
                        result_text = json.dumps(parsed)
                        print(f"[DEBUG] Injected missing image URL into agent response")
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

            result = [result_text]
            return result
                
        except Exception as e:
            print(f"[ERROR] Conversation failed: {str(e)}")
            return [f"Error processing message: {str(e)}"]

    async def run_conversation_with_text_stream(self, input_message: str = ""):
        """Async wrapper for conversation processing with better error handling."""
        print(f"[DEBUG] Async conversation pipeline initiated", flush=True)
        loop = asyncio.get_running_loop()
        try:
            messages = await loop.run_in_executor(
                _executor, self._run_conversation_sync, input_message
            )
            for i, msg in enumerate(messages):
                yield msg
        except Exception as e:
            print(f"[ERROR] Async conversation failed: {str(e)}")
            yield f"Error processing message: {str(e)}"

    @classmethod
    def clear_toolset_cache(cls):
        """Clear the toolset cache if needed."""
        global _toolset_cache
        _toolset_cache.clear()

    @classmethod
    def get_cache_stats(cls):
        """Get cache statistics for monitoring."""
        return {
            "toolset_cache_size": len(_toolset_cache),
            "cached_agent_types": list(_toolset_cache.keys())
        }

def create_function_tool_for_agent(agent_type: str) -> List[Any]:
    define_mcp_create_image =FunctionTool(
            name="mcp_create_image",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate"
                    },
                    "image_url": {
                        "type": "string",
                        "description": "URL of the product image to use as a reference for the generated image"
                    }
                },
                "required": ["prompt", "image_url"],
                "additionalProperties": False
            },
            description="Generate an AI image based on a text description and a reference product image URL using the GPT image model of choice.",
            strict=True
        )
    define_mcp_product_recommendations = FunctionTool(
        name="mcp_product_recommendations",
        parameters={
            "type": "object",
            "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language user query describing what products they're looking for"
                    }
                },
                "required": ["question"],
                "additionalProperties": False
            },
            description="Search for product recommendations based on user query.",
            strict=True
        )
    define_mcp_calculate_discount = FunctionTool(
        name="mcp_calculate_discount",
        parameters={
            "type": "object",
            "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "The ID of the customer."
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False
            },
            description="Calculate the discount based on customer data.",
            strict=True
        )
    define_mcp_inventory_check = FunctionTool(
        name="mcp_inventory_check",
        parameters={
            "type": "object",
            "properties": {
                    "product_list": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of product IDs to check inventory for."
                    }
                },
            "required": ["product_list"],
            "additionalProperties": False
        },
        description="Check inventory for a product using MCP client.",
        strict=True
        )

    functions = []

    if agent_type == "interior_designer":
        functions = [define_mcp_create_image, define_mcp_product_recommendations]
    elif agent_type == "customer_loyalty":
        functions = [define_mcp_calculate_discount]
    elif agent_type == "inventory_agent":
        functions = [define_mcp_inventory_check]
    elif agent_type == "cart_manager":
        # Cart manager uses conversation context, minimal tools needed
        functions = []
    elif agent_type == "cora":
        # Cora is a general assistant with product recommendations
        functions = [define_mcp_product_recommendations]
    return functions
