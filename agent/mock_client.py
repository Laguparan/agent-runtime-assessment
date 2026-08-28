import json
import urllib.request
import urllib.error
import time

# Assuming the mock server runs locally on a standard port (adjust if the Makefile specifies otherwise)
MOCK_LLM_URL = "http://localhost:8000/v1/chat/completions"

def call_llm(messages, tools, max_retries=5):
    """
    Communicates with the mock LLM and survives S5 (Drops) and S6 (Rate Limits).
    """
    payload = json.dumps({"messages": messages, "tools": tools}).encode('utf-8')
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                MOCK_LLM_URL, 
                data=payload, 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10.0) as response:
                return json.loads(response.read().decode('utf-8'))
                
        except urllib.error.HTTPError as e:
            # Handle Scenario S6: 429 Retry-After and 529 Overload
            if e.code in (429, 529): 
                retry_after = int(e.headers.get('Retry-After', 2))
                print(f"[Network] Server overloaded ({e.code}). Retrying in {retry_after}s...")
                time.sleep(retry_after)
                continue
            return {"error": f"HTTP {e.code}: {e.reason}"}
            
        except (urllib.error.URLError, ConnectionResetError) as e:
            # Handle Scenario S5: Connection reset mid-response at a random byte offset
            backoff = 2 ** attempt
            print(f"[Network] Connection dropped (S5). Backing off for {backoff}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(backoff)
            continue
            
    return {"error": "Critical Failure: Could not reach Mock LLM after maximum retries."}






if __name__ == "__main__":
    print("--- Testing Mock LLM Client Resilience ---\n")
    print("Attempting to send a test message to localhost:8000...")
    
    # A simple dummy message to test the network
    test_messages = [{"role": "user", "content": "Hello, system!"}]
    test_tools = []
    
    # This will fail and trigger our retry logic because the mock server isn't running
    result = call_llm(test_messages, test_tools, max_retries=3)
    
    print("\n--- Final Result ---")
    print(result)