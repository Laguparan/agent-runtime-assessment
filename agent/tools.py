import os
import subprocess

# 1. Define the absolute boundary of our sandbox
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")

def _get_safe_path(file_path):
    """
    Security Gate: Ensures the requested file stays strictly inside the workspace.
    This prevents path traversal attacks like '../../etc/passwd'.
    """
    # Resolve the absolute path based on the user's input
    target_path = os.path.abspath(os.path.join(WORKSPACE_DIR, file_path))
    
    # Check if the resolved path actually starts with our allowed workspace directory
    if not target_path.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise PermissionError(f"Security Violation: Access to {file_path} is blocked.")
    
    return target_path

def read_file(path):
    """Tool: Read a file from the workspace."""
    try:
        safe_target = _get_safe_path(path)
        with open(safe_target, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path, content):
    """Tool: Write a file securely to the workspace."""
    try:
        safe_target = _get_safe_path(path)
        with open(safe_target, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Success: Wrote contents to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def run_python(code):
    """Tool: Run Python code in an isolated subprocess with a timeout."""
    try:
        # We use subprocess to isolate the AI's code from our main agent loop
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=5.0  # Hard 5-second wall-clock limit
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Execution Failed:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Python execution timed out after 5 seconds."
    except Exception as e:
        return f"System Error: {str(e)}"

import urllib.request
import sqlite3
from storage import DB_PATH  # Import the DB path we created earlier

# --- HTTP GET TOOL ---
ALLOWED_DOMAINS = ["api.github.com", "example.com", "jsonplaceholder.typicode.com"]

def http_get(url):
    """Tool: Fetch data from allow-listed URLs only."""
    try:
        # Simple domain extraction
        domain = url.split("//")[-1].split("/")[0]
        
        if domain not in ALLOWED_DOMAINS:
            return f"Error: HTTP GET refused. Domain '{domain}' is not in the allow-list."
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Agentic-Runtime/1.0'})
        with urllib.request.urlopen(req, timeout=5.0) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        return f"HTTP Error: {str(e)}"

# --- SEND EMAIL TOOL ---
def send_email(to_address, subject, body, run_id, step):
    """
    Tool: Simulates sending an email by logging it to SQLite.
    This is treated as an irreversible action for R2 (Exactly-once guarantee).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # We use a unique idempotency key so we never send the exact same email twice
        idempotency_key = f"{run_id}_step_{step}_email"
        
        # Check if we already sent this (Crash recovery)
        cursor.execute("SELECT status FROM tool_intents WHERE intent_id = ?", (idempotency_key,))
        row = cursor.fetchone()
        
        if row and row[0] == 'COMPLETED':
            conn.close()
            return "Email already sent previously (Recovered from crash)."
            
        # Log the intent and "send" the email
        cursor.execute("""
            INSERT OR REPLACE INTO tool_intents (intent_id, run_id, tool_name, status, result_payload)
            VALUES (?, ?, ?, ?, ?)
        """, (idempotency_key, run_id, 'send_email', 'COMPLETED', f"To: {to_address} | Subject: {subject}"))
        
        conn.commit()
        conn.close()
        
        return f"Success: Email sent to {to_address} with subject '{subject}'"
    except Exception as e:
        return f"Email System Error: {str(e)}"

    
if __name__ == "__main__":
    print("--- Running Sandbox Security Tests ---\n")

    # Test 1: Safe File Write
    print("Test 1: Writing to a safe path (workspace/test.txt)")
    print(write_file("test.txt", "Hello from the sandbox!"))
    print("-" * 40)

    # Test 2: Malicious Path Traversal Attempt
    print("Test 2: Attempting path traversal (../hacked.txt)")
    print(write_file("../hacked.txt", "This should fail!"))
    print("-" * 40)

    # Test 3: Safe Python Execution
    print("Test 3: Running a basic Python script")
    print("Output:", run_python("print('Math test:', 5 * 5)"))
    print("-" * 40)

    # Test 4: Malicious Infinite Loop (Timeout Test)
    print("Test 4: Simulating a hanging subprocess (sleeping for 10 seconds)")
    print("Output:", run_python("import time; time.sleep(10)"))
    print("-" * 40)