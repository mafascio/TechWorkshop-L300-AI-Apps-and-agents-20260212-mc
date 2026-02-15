"""
Test script for mcp_create_image function via MCP server.

Usage:
    python test_mcp_create_image.py

Requires:
    - MCP server running at MCP_SERVER_URL (default: http://localhost:8000/mcp-inventory/sse)
    - .env file with required environment variables (or set them in your shell)
"""
import asyncio
import os
import sys
import json
from pathlib import Path

# Add src to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Import MCP client directly (bypass app/__init__.py which pulls in azure.cosmos)
from app.servers.mcp_inventory_client import MCPShopperToolsClient, get_mcp_client


MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp-inventory/sse")

# Test parameters
TEST_PROMPT = "A modern living room with a minimalist white sofa, natural light, and indoor plants"
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/220px-Golde33443.jpg"  # placeholder reference image


# ---- Define both versions of mcp_create_image inline to avoid importing app package ----

def mcp_create_image_fixed(prompt: str, image_url: str) -> str:
    """FIXED version: synchronous wrapper with internal async helper (same pattern as other wrappers)."""
    async def _create_image():
        mcp_client = await get_mcp_client(MCP_SERVER_URL)
        result = await mcp_client.call_tool(
            "generate_product_image",
            {"prompt": prompt, "image_url": image_url}
        )
        return result

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(_create_image())


async def mcp_create_image_original(prompt: str, image_url: str) -> str:
    """ORIGINAL (buggy) version: async function that creates a nested event loop."""
    mcp_client = await get_mcp_client(MCP_SERVER_URL)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            mcp_client.call_tool("generate_product_image", {"prompt": prompt, "image_url": image_url})
        )
        return result
    finally:
        loop.close()


async def test_list_tools():
    """Step 1: Verify the MCP server is reachable and list available tools."""
    print("=" * 60)
    print("TEST 1: List available MCP tools")
    print("=" * 60)
    client = MCPShopperToolsClient(MCP_SERVER_URL)
    tools = await client.list_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
        if hasattr(tool, 'inputSchema'):
            print(f"    Input schema: {json.dumps(tool.inputSchema, indent=4)}")
    print(f"\nTotal tools: {len(tools)}")
    # Check that generate_product_image exists
    tool_names = [t.name for t in tools]
    assert "generate_product_image" in tool_names, (
        f"'generate_product_image' not found in MCP tools! Available: {tool_names}"
    )
    print("  ✓ 'generate_product_image' tool is registered on the MCP server.\n")
    return tools


async def test_call_tool_directly():
    """Step 2: Call generate_product_image directly via the MCP client."""
    print("=" * 60)
    print("TEST 2: Call generate_product_image via MCP client directly")
    print("=" * 60)
    client = MCPShopperToolsClient(MCP_SERVER_URL)
    print(f"  Prompt:     {TEST_PROMPT}")
    print(f"  Image URL:  {TEST_IMAGE_URL}")
    result = await client.call_tool(
        "generate_product_image",
        {"prompt": TEST_PROMPT, "image_url": TEST_IMAGE_URL}
    )
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result:      {json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result}")
    print("  ✓ Direct MCP call succeeded.\n")
    return result


async def test_mcp_create_image_new():
    """
    Step 3: Test the FIXED mcp_create_image (synchronous wrapper).
    
    The fixed version is a synchronous function (matching the other wrappers) that
    internally creates an async helper and runs it via the event loop. We call it
    from a thread so it gets its own event loop (simulating how _run_conversation_sync
    is called via run_in_executor in production).
    """
    print("=" * 60)
    print("TEST 3: Test FIXED mcp_create_image (sync wrapper in thread)")
    print("=" * 60)

    print(f"  Prompt:     {TEST_PROMPT}")
    print(f"  Image URL:  {TEST_IMAGE_URL}")

    # Run the sync function in a thread (same as production: run_in_executor)
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor, mcp_create_image_fixed, TEST_PROMPT, TEST_IMAGE_URL
        )
    print(f"  Result type: {type(result).__name__}")
    print(f"  Result:      {json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result}")
    print("  ✓ Fixed mcp_create_image succeeded.\n")
    return result


async def test_mcp_create_image_original_bug():
    """
    Step 4: Attempt to run the ORIGINAL mcp_create_image to demonstrate the bug.
    
    This is expected to fail with:
      RuntimeError: Cannot run the event loop while another loop is running
    """
    print("=" * 60)
    print("TEST 4: Test ORIGINAL mcp_create_image (expected to fail)")
    print("=" * 60)

    print(f"  Prompt:     {TEST_PROMPT}")
    print(f"  Image URL:  {TEST_IMAGE_URL}")
    try:
        result = await mcp_create_image_original(TEST_PROMPT, TEST_IMAGE_URL)
        print(f"  Result: {result}")
        print("  ⚠ Original function succeeded (unexpected — may depend on runtime).\n")
    except RuntimeError as e:
        print(f"  ✗ RuntimeError as expected: {e}")
        print("  This confirms the bug: cannot call loop.run_until_complete() "
              "inside an already-running async context.\n")
    except Exception as e:
        print(f"  ✗ Unexpected error: {type(e).__name__}: {e}\n")


async def main():
    print(f"\nMCP Server URL: {MCP_SERVER_URL}\n")

    # Test 1 - list tools
    try:
        await test_list_tools()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        print("  Is the MCP server running? Check MCP_SERVER_URL.\n")
        return

    # Test 2 - direct call_tool
    try:
        await test_call_tool_directly()
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")

    # Test 3 - fixed version from agent_processor_new.py
    try:
        await test_mcp_create_image_new()
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")

    # Test 4 - original (buggy) version
    try:
        await test_mcp_create_image_original_bug()
    except Exception as e:
        print(f"  ✗ FAILED: {e}\n")

    print("=" * 60)
    print("All tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
