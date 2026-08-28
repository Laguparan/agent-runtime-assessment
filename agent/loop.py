import json
from memory import compact_history
from storage import get_connection
from mock_client import call_llm
from tools import read_file, write_file, run_python, http_get, send_email

# R5: Loop control - Hard ceiling on steps to prevent infinite loops
MAX_STEPS = 15 

def execute_tool(tool_name, tool_args, run_id, step):
    """Routes the LLM's requested tool to the correct Python function."""
    try:
        args = json.loads(tool_args)
    except json.JSONDecodeError:
        # S2: Malformed tool arguments - Return error to LLM instead of crashing
        return "Error: Invalid JSON arguments. Please fix your syntax."

    if tool_name == "read_file":
        return read_file(args.get("path", ""))
    elif tool_name == "write_file":
        return write_file(args.get("path", ""), args.get("content", ""))
    elif tool_name == "run_python":
        return run_python(args.get("code", ""))
    elif tool_name == "http_get":
        return http_get(args.get("url", ""))
    elif tool_name == "send_email":
        return send_email(args.get("to", ""), args.get("subject", ""), args.get("body", ""), run_id, step)
    else:
        # S3: Calls a tool that does not exist
        return f"Error: Tool '{tool_name}' does not exist."

def run_agent(run_id, task_description):
    """The main autonomous loop."""
    print(f"\n🚀 Starting Agent Run: {run_id}")
    print(f"Task: {task_description}\n")
    
    # Initial state
    messages = [
        {"role": "system", "content": "You are a helpful autonomous agent. Use tools to solve the user's task."},
        {"role": "user", "content": task_description}
    ]
    
    # Expose our tools to the LLM
    available_tools = ["read_file", "write_file", "run_python", "http_get", "send_email"]
    
    for step in range(1, MAX_STEPS + 1):
        print(f"--- Step {step} ---")
        
        # Ensure we are under the 8000 token limit before hitting the server
        messages = compact_history(messages)
        
        # 1. Call the LLM (using our resilient client)
        response = call_llm(messages, available_tools)
        
        if "error" in response:
            print(f"Agent stopped due to network failure: {response['error']}")
            break
            
        # 2. Extract the LLM's intent
        llm_msg = response.get("message", "")
        tool_calls = response.get("tool_calls", [])
        
        # Append the assistant's response to history
        messages.append({"role": "assistant", "content": llm_msg, "tool_calls": tool_calls})
        
        if llm_msg:
            print(f"🤖 LLM: {llm_msg}")
            
        # 3. If no tools are called, the agent thinks it is done!
        if not tool_calls:
            print("\n✅ Task Complete!")
            break
            
        # 4. Execute the requested tools
        for tool in tool_calls:
            tool_name = tool.get("name")
            tool_args = tool.get("arguments", "{}")
            
            print(f"🔧 Executing Tool: {tool_name}...")
            
            # Run our secure tool execution function
            result = execute_tool(tool_name, tool_args, run_id, step)
            
            print(f"📄 Result: {result[:100]}...") # Print first 100 chars
            
            # 5. Append the result back to the conversation history so the LLM can read it
            messages.append({
                "role": "tool",
                "tool_name": tool_name,
                "content": result
            })
            
    else:
        print(f"\n⚠️ Agent terminated: Reached maximum step limit of {MAX_STEPS} without finishing.")

if __name__ == "__main__":
    # A quick test run
    run_agent("test_run_001", "Please write a Python script that calculates the 10th Fibonacci number and save it to workspace/fib.py")