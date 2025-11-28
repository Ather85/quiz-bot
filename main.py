import uvicorn
import os
import json
import ast
import requests
import time
import io
import contextlib
import re
import traceback
from urllib.parse import urljoin

# FastAPI imports
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- OpenAI Import ---
from openai import OpenAI

# --- 1. Load Configuration from Environment Variables ---
print("=" * 60)
print("DEBUG: Checking Environment Variables")
print("=" * 60)

# Get from GitHub secrets (set in Render environment)
MY_SECRET_KEY = os.getenv("MY_SECRET_KEY")
MY_EMAIL = os.getenv("MY_EMAIL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Changed from ATPES_API_KEY
MODEL_NAME = os.getenv("MODEL", "gpt-4o")

print(f"OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")
print(f"MY_EMAIL present: {bool(MY_EMAIL)}")
print(f"MY_SECRET_KEY present: {bool(MY_SECRET_KEY)}")
print(f"MODEL_NAME: {MODEL_NAME}")
print("=" * 60)

if not all([MY_SECRET_KEY, MY_EMAIL, OPENAI_API_KEY]):
    raise ValueError("FATAL ERROR: One or more environment variables (OPENAI_API_KEY, MY_EMAIL, MY_SECRET_KEY) are missing.")

# --- Initialize OpenAI Client ---
try:
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    llm_client = None

# --- 2. Define Data Models ---
class QuizTask(BaseModel):
    email: str
    secret: str
    url: HttpUrl

# --- 3. The "Toolbox" (Python Functions) ---
def read_web_page_tool(url: str, base_url: str = None) -> str:
    """Visits a URL with a headless browser and returns all visible text."""
    print(f"[Tool Call]: read_web_page_tool(url='{url}')")
    
    # FIX: Handle relative URLs
    if url.startswith('/') and base_url:
        url = urljoin(base_url, url)
        print(f"[Tool Fix]: Converted relative URL to: {url}")
    
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        page_text = driver.find_element(By.TAG_NAME, "body").text
        print(f"[Tool Result]: Scraped text (first 150 chars): {page_text[:150]}...")
        return page_text
    except Exception as e:
        print(f"[Tool Error]: read_web_page_tool failed: {e}")
        return f"Error: Could not read page. {e}"
    finally:
        if driver:
            driver.quit()

def extract_submission_url_from_text(text: str) -> str:
    """Extract submission URL from quiz text more reliably"""
    patterns = [
        r'POST.*?to\s+(https?://[^\s<>"\']+/submit)',
        r'Post your answer to\s+(https?://[^\s<>"\']+)',
        r'https://[^\s<>"\']+/submit'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group(1)
            # Ensure it's a complete URL
            if not url.startswith('http'):
                url = 'https://' + url
            print(f"[URL Extract]: Found submission URL: {url}")
            return url
    
    # Default fallback
    default_url = "https://tds-llm-analysis.s-anand.net/submit"
    print(f"[URL Extract]: Using default submission URL: {default_url}")
    return default_url

def smart_data_extraction(text_input: str) -> str:
    """Extract common patterns from quiz text more reliably"""
    patterns = [
        (r'Secret code is\s+(\d+)', 'secret_code'),
        (r'code is\s+(\d+)', 'code'),
        (r'answer["\']?\s*[:=]\s*["\']?(\d+)', 'answer_num'),
        (r'\b(\d{4})\b', 'four_digit'),
        (r'\b(\d+)\b', 'any_digit')
    ]
    
    for pattern, pattern_type in patterns:
        match = re.search(pattern, text_input)
        if match:
            print(f"[Smart Extract]: Found {pattern_type}: {match.group(1)}")
            return match.group(1)
    
    return "NOT_FOUND"

def run_python_tool(code_to_run: str, text_input: str = None) -> str:
    r"""Executes a string of Python code and returns its print() output."""
    print(f"[Tool Call]: run_python_tool(code_snippet_length={len(code_to_run)})")
    
    # CRITICAL FIX: Unescape newlines that LLM might have double-escaped
    code_to_run = code_to_run.encode().decode('unicode_escape')
    
    print(f"[Tool Debug]: Code after unescaping (first 200 chars):\n{code_to_run[:200]}")
    
    # Safety checks
    if "input(" in code_to_run:
        err = "Error: Code attempts to call input(), which is not allowed in automated execution."
        print(f"[Tool Error]: {err}")
        return f"Error: Code execution blocked. {err}"
    
    # Prepare safe globals/locals for exec
    safe_globals = {
        "re": re,
        "json": json,
        "__name__": "__run_python_tool__",
    }
    local_scope = {"text_input": text_input}
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            exec(code_to_run, safe_globals, local_scope)
        result = stream.getvalue().strip()
        if not result:
            result = "Code executed successfully, but produced no print output."
        print(f"[Tool Result]: {result}")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Tool Error]: run_python_tool failed: {e}\n{tb}")
        return f"Error: Code execution failed. {e}"

def submit_answer_tool(submit_url: str, answer_payload) -> str:
    """Posts the final JSON answer to the submission URL."""
    print(f"[Tool Call]: submit_answer_tool(url='{submit_url}', payload_type={type(answer_payload)})")
    try:
        data = None
        if isinstance(answer_payload, dict):
            data = answer_payload
        elif isinstance(answer_payload, str):
            try:
                data = json.loads(answer_payload)
            except Exception:
                try:
                    data = ast.literal_eval(answer_payload)
                except Exception as e:
                    print(f"[Tool Error]: Could not parse answer_payload string as JSON or Python dict: {e}")
                    return f"Error: Could not parse answer_payload. {e}"
        else:
            print(f"[Tool Error]: Unsupported answer_payload type: {type(answer_payload)}")
            return f"Error: Unsupported answer_payload type: {type(answer_payload)}"

        if not isinstance(data, dict):
            return "Error: Submission payload is not a JSON object."

        print(f"[Tool Debug]: Submitting payload: {json.dumps(data, indent=2)}")
        response = requests.post(submit_url, json=data, timeout=30)
        
        print(f"[Tool Debug]: Response status: {response.status_code}")
        print(f"[Tool Debug]: Response body: {response.text[:500]}")
        
        response.raise_for_status()
        response_json = response.json()
        print(f"[Tool Result]: Submission successful. Response: {response_json}")
        return json.dumps(response_json)
    except requests.exceptions.HTTPError as e:
        print(f"[Tool Error]: HTTP Error: {e}")
        print(f"[Tool Error]: Response was: {response.text if 'response' in locals() else 'No response'}")
        return f"Error: Answer submission failed. {e}"
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Tool Error]: submit_answer_tool failed: {e}\n{tb}")
        return f"Error: Answer submission failed. {e}"

# --- Tool Definitions for the LLM ---
TOOLS_DEFINITION = """
[
  {
    "name": "read_web_page_tool",
    "description": "Visits a URL with a headless browser and returns all visible text.",
    "parameters": {
      "url": "The URL to scrape"
    }
  },
  {
    "name": "run_python_tool",
    "description": "Executes Python code using text_input injected from the previous step. Use text_input='<last_result>' when needed.",
    "parameters": {
      "code_to_run": "The Python code to execute.",
      "text_input": "(Optional) Use '<last_result>' to pass previous output."
    }
  },
  {
    "name": "submit_answer_tool",
    "description": "Submits the answer JSON to the quiz endpoint.",
    "parameters": {
      "submit_url": "Submission URL",
      "answer_payload": "Complete answer JSON"
    }
  }
]
"""

TOOLS_MAP = {
    "read_web_page_tool": read_web_page_tool,
    "run_python_tool": run_python_tool,
    "submit_answer_tool": submit_answer_tool,
}

# --- LLM "Brain" Function ---
def call_llm_brain(scraped_text: str, current_task_url: str, email: str, secret: str, previous_error: str | None = None) -> list[dict]:
    """Ask OpenAI to produce a JSON plan (list of tool calls)."""
    if llm_client is None:
        print("[Brain Error]: LLM client is not initialized.")
        return []

    # SIMPLIFIED system prompt - much more directive
    system_prompt = f"""
You are a quiz-solving agent. Output ONLY valid JSON array of steps.

USER INFO:
- Email: {email}
- Secret: {secret} 
- Current Quiz URL: {current_task_url}

AVAILABLE TOOLS: {TOOLS_DEFINITION}

CRITICAL RULES:
1. Output MUST be valid JSON array: [{{"name": "tool", "parameters": {{...}}}}]
2. ALWAYS use this 3-step pattern:
   - Step 1: read_web_page_tool (get quiz content)
   - Step 2: run_python_tool (build answer JSON)
   - Step 3: submit_answer_tool (submit answer)

3. For run_python_tool:
   - Use text_input="<last_result>" to get previous output
   - Code MUST print final JSON using print(json.dumps(answer))
   - Use SINGLE backslash n for newlines: \\n (not double)
   - text_input is always a STRING, never use json.loads() on it

4. Answer JSON format (ALWAYS use this exact structure):
{{
    "email": "{email}",
    "secret": "{secret}", 
    "url": "{current_task_url}",
    "answer": "YOUR_EXTRACTED_ANSWER_HERE"
}}

5. To extract data from text_input:
   - Use regex: match = re.search(r'pattern', text_input)
   - Always handle None case: if match: value = match.group(1); else: value = 'DEFAULT'
   - For secret codes: r'Secret code is (\\d+)' or r'code is (\\w+)'

6. Extract submission URL from quiz text:
   - Look for "POST this JSON to https://..." pattern
   - Default: "https://tds-llm-analysis.s-anand.net/submit"

EXAMPLE FOR DEMO QUIZ:
[
  {{
    "name": "read_web_page_tool",
    "parameters": {{"url": "{current_task_url}"}}
  }},
  {{
    "name": "run_python_tool", 
    "parameters": {{
      "code_to_run": "import json\\nimport re\\n\\n# Extract submission URL first\\nsubmit_match = re.search(r'POST.*?to\\\\s+(https?://[^\\\\s<>\\\\\\"\\\\']+/submit)', text_input, re.IGNORECASE)\\nif submit_match:\\n    submission_url = submit_match.group(1)\\nelse:\\n    submission_url = 'https://tds-llm-analysis.s-anand.net/submit'\\n\\n# Extract answer\\nanswer_match = re.search(r'answer[\\\\\\"\\\\']?\\\\s*[:=]\\\\s*[\\\\\\"\\\\']?(\\\\d+)[\\\\\\"\\\\']?', text_input)\\nif answer_match:\\n    answer_val = answer_match.group(1)\\nelse:\\n    answer_val = '123'  # fallback\\n\\n# Build answer JSON\\nanswer = {{'email': '{email}', 'secret': '{secret}', 'url': '{current_task_url}', 'answer': answer_val}}\\nprint(json.dumps(answer))",
      "text_input": "<last_result>"
    }}
  }},
  {{
    "name": "submit_answer_tool",
    "parameters": {{
      "submit_url": "<last_result>",
      "answer_payload": "<last_result>"
    }}
  }}
]

Output ONLY the JSON array. No other text.
"""

    user_prompt = f"Quiz text from {current_task_url}:\n---\n{scraped_text}\n---"
    if previous_error:
        user_prompt += f"\n\nPREVIOUS ERROR: {previous_error}\nFix the above error in your new plan."

    print("\n[Brain]: Calling OpenAI to get a plan...")

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent JSON
        )
        
        plan_json = response.choices[0].message.content.strip()
        print(f"[Brain DEBUG]: Raw response:\n{plan_json}\n")

        # Clean JSON - remove markdown code blocks
        if plan_json.startswith("```json"):
            plan_json = plan_json[7:]
        elif plan_json.startswith("```"):
            plan_json = plan_json[3:]
        if plan_json.endswith("```"):
            plan_json = plan_json[:-3]
        plan_json = plan_json.strip()

        # Parse JSON
        plan_data = json.loads(plan_json)
        
        # Convert to our plan format
        plan = []
        for step in plan_data:
            if isinstance(step, dict) and "name" in step and "parameters" in step:
                plan.append({
                    "tool": step["name"],
                    "args": step["parameters"]
                })
        
        print(f"[Brain]: Successfully parsed plan with {len(plan)} steps.")
        return plan
        
    except json.JSONDecodeError as e:
        print(f"[Brain Error]: JSON parse failed: {e}")
        print(f"[Brain Debug]: Failed to parse:\n{plan_json}")
        return []
    except Exception as e:
        print(f"[Brain Error]: OpenAI call failed: {e}")
        return []

# --- 4. The "Solver" Agent (Main Loop) ---
def solve_quiz_in_background(task_url: str, email: str, secret: str):
    print(f"\n--- AGENT TASK STARTED ---")
    current_task_url = task_url
    quiz_count = 0
    MAX_QUIZZES = 10

    while current_task_url and (quiz_count < MAX_QUIZZES):
        print(f"\n--- Processing Quiz: {current_task_url} (Quiz #{quiz_count + 1}) ---")
        quiz_count += 1

        last_error = None
        retry_count = 0
        MAX_RETRIES = 2

        while retry_count <= MAX_RETRIES:
            if last_error:
                print(f"\n[Agent]: Retrying... (Attempt {retry_count + 1}/{MAX_RETRIES + 1} for this URL)")

            # FIX: Pass base_url for relative URL resolution
            scraped_text = read_web_page_tool(current_task_url, base_url=current_task_url)
            if scraped_text.startswith("Error:"):
                print("Failed to scrape initial page. Aborting task.")
                current_task_url = None
                break

            plan = call_llm_brain(scraped_text, current_task_url, email, secret, previous_error=last_error)
            last_error = None

            if not plan:
                print("Failed to get a valid plan from LLM.")
                last_error = "LLM failed to return a valid plan."
                retry_count += 1
                time.sleep(1)
                continue

            last_tool_output = None
            plan_failed = False

            for i, step in enumerate(plan):
                tool_name = step.get("tool")
                tool_args = step.get("args", {})
                print(f"\n[Agent]: Executing Step {i+1}: {tool_name}")

                if tool_name in TOOLS_MAP:
                    tool_function = TOOLS_MAP[tool_name]

                    # Inject last_tool_output recursively
                    try:
                        def inject_last_result(data):
                            if isinstance(data, dict):
                                newd = {}
                                for k, v in data.items():
                                    newd[k] = inject_last_result(v)
                                return newd
                            elif isinstance(data, list):
                                return [inject_last_result(x) for x in data]
                            elif isinstance(data, str) and data == "<last_result>":
                                return last_tool_output
                            else:
                                return data
                        tool_args = inject_last_result(tool_args)
                    except Exception as e:
                        print(f"[Agent Error] during argument injection: {e}")

                    try:
                        # Special handling for read_web_page_tool to pass base_url
                        if tool_name == "read_web_page_tool" and "url" in tool_args:
                            result = tool_function(tool_args["url"], base_url=current_task_url)
                        else:
                            result = tool_function(**tool_args)
                        last_tool_output = result

                        if isinstance(result, str) and result.startswith("Error:"):
                            print(f"[Agent Error]: Tool {tool_name} reported a failure.")
                            last_error = f"Step {i+1} ({tool_name}) failed with error: {result}"
                            plan_failed = True
                            break

                        if tool_name == "submit_answer_tool":
                            try:
                                submit_response = json.loads(result)
                                new_url = submit_response.get("url")
                                if new_url:
                                    if submit_response.get("correct") is True:
                                        print(f"[Agent]: ✓ CORRECT! Received new quiz URL: {new_url}")
                                    else:
                                        reason = submit_response.get("reason", "No reason given.")
                                        print(f"[Agent]: ✗ Answer incorrect (Reason: {reason}), but a new URL was provided. Proceeding to: {new_url}")
                                    current_task_url = new_url
                                else:
                                    reason = submit_response.get("reason", "No reason given.")
                                    correct = submit_response.get("correct", False)
                                    if correct:
                                        print(f"[Agent]: ✓ FINAL ANSWER CORRECT! Quiz chain completed. Response: {submit_response}")
                                    else:
                                        print(f"[Agent]: ✗ Submission response had no new URL (Reason: {reason}). Ending quiz chain.")
                                    current_task_url = None
                                plan_failed = False
                                break
                            except Exception as e:
                                print(f"[Agent Error]: Failed to parse submission response: {e}")
                                print(f"[Agent Debug]: Raw response was: {result}")
                                last_error = f"Step {i+1} ({tool_name}) failed to parse response: {e}"
                                plan_failed = True
                                break

                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"[Agent Error]: Step {i+1} ({tool_name}) crashed with exception: {e}\n{tb}")
                        last_error = f"Step {i+1} ({tool_name}) crashed with exception: {e}"
                        plan_failed = True
                        break
                else:
                    print(f"[Agent Error]: Unknown tool '{tool_name}' in plan. Aborting.")
                    last_error = f"Unknown tool '{tool_name}' in plan."
                    plan_failed = True
                    break

            if not plan_failed:
                break
            else:
                retry_count += 1
                print(f"[Agent]: Plan failed. Error: {last_error}")
                time.sleep(1)

        if retry_count > MAX_RETRIES:
            print(f"[Agent]: FAILED to solve quiz {current_task_url} after {MAX_RETRIES + 1} attempts. Aborting chain.")
            current_task_url = None

    if quiz_count >= MAX_QUIZZES:
        print(f"[Agent]: Reached max quiz limit ({MAX_QUIZZES}). Stopping.")

    print(f"--- AGENT TASK FINISHED ---")

# --- 5. Create the FastAPI App ---
app = FastAPI()

@app.post("/solve-quiz")
async def solve_quiz_endpoint(task: QuizTask, background_tasks: BackgroundTasks):
    print(f"Received new quiz task for: {task.url}")
    if task.secret != MY_SECRET_KEY:
        print(f"Error: Invalid secret.")
        raise HTTPException(status_code=403, detail="Invalid secret")
    if task.email != MY_EMAIL:
        print(f"Warning: Email mismatch.")

    print("Secret validated. Starting agent in the background...")
    background_tasks.add_task(solve_quiz_in_background, str(task.url), task.email, task.secret)

    return {"status": "Task received. Agent is processing in the background."}

@app.get("/")
def read_root():
    return {"status": "Quiz Bot API is running!"}

# --- 6. Run the Server / Test Mode ---
TEST_MODE = False
TEST_URL = "https://tds-llm-analysis.s-anand.net/demo"

if __name__ == "__main__":
    if not llm_client:
        print("Could not start server, LLM client failed to initialize.")
    elif TEST_MODE:
        print(f"--- STARTING IN TEST MODE ---")
        print(f"Test URL: {TEST_URL}")
        if not MY_EMAIL or not MY_SECRET_KEY:
            print("TEST MODE FAILED: MY_EMAIL or MY_SECRET_KEY not set.")
        else:
            solve_quiz_in_background(
                task_url=TEST_URL,
                email=MY_EMAIL,
                secret=MY_SECRET_KEY
            )
        print("--- TEST MODE FINISHED ---")
    else:
        print(f"Starting Quiz Bot server on http://0.0.0.0:8000 (TEST_MODE is False)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
