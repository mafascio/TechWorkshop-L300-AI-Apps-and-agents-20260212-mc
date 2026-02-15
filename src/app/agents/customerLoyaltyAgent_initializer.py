import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from agent_processor import create_function_tool_for_agent
from agent_initializer import initialize_agent

load_dotenv()

CL_PROMPT_TARGET = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'prompts', 'CustomerLoyaltyAgentPrompt.txt')
with open(CL_PROMPT_TARGET, 'r', encoding='utf-8') as file:
    CL_PROMPT = file.read()


## Define MS Foundry Azure AI prj info
project_endpoint = os.environ["FOUNDRY_ENDPOINT"]
## Create the AI Project client:
project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(),
)

# Define the set of user-defined callable functions to use as tools (from MCP client)
functions = create_function_tool_for_agent("customer_loyalty")

# This makes reference to a function called create_function_tool_for_agent() 
# in src/app/agents/agent_processor.py. 
# This function takes in an agent type and defines the functions that particular agent will use.
# In this case, it defines a JSON schema describing the mcp_calculate_discount() function as a tool for the customer loyalty agent. 
# This function calls get_customer_discount(), which finally calls calculate_discount() in src/app/tools/discountLogic.py. 
# This function takes a customer ID and returns a discount percentage based on the customer’s loyalty tier. You can review the code in this file to understand how it works. 
# This particular tool is more complex than others because it communicates with the GPT model to determine the appropriate discount based on the customer’s transaction history. 
# It also simulates connecting to two separate databases to retrieve customer information.

## Create the customer loyalty agent in MS Foundry
initialize_agent(
    project_client=project_client,
    model=os.environ["gpt_deployment"],
    name="customer-loyalty",
    description="Zava Customer Loyalty Agent",
    instructions=CL_PROMPT,
    tools=functions
)


# This code initializes the agent with the specified model, name, instructions, and toolset. 
# It then creates the agent in Microsoft Foundry and prints the agent ID to the console. 
