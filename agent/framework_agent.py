import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from openai import AsyncOpenAI

# 1. Map Pydantic AI to our local Mock Server via OpenAI-compatible provider
client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed" # Mock server doesn't require a real API key
)

model = OpenAIChatModel(
    model_name="mock-model",
    openai_client=client
)

# 2. Initialize the Pydantic AI Agent
agent = Agent(
    model=model,
    system_prompt="You are a precise autonomous engineering agent. Use tools safely to solve tasks."
)

# 3. Register Core Tools with Type Hints (F3: Trust Boundary)
@agent.tool
def read_file(ctx: RunContext, path: str) -> str:
    """Safely read text files within the workspace."""
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace"))
    target_path = os.path.abspath(os.path.join(workspace_dir, path))
    
    if not target_path.startswith(workspace_dir):
        return "Error: Security Violation - Path traversal blocked."
    
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@agent.tool
def send_email(ctx: RunContext, to: str, subject: str, body: str) -> str:
    """Simulated irreversible side-effect tool."""
    # F2: In a full production setup, this would hook into storage.py intent ledger.
    return f"Success: Email dispatched to {to} with subject '{subject}'."

if __name__ == "__main__":
    import asyncio
    
    async def main():
        result = await agent.run("Read test.txt from the workspace and summarize it.")
        print("Agent Response:", result.data)
        
    asyncio.run(main())