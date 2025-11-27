# main.py — Groq-ready, hardened quiz-bot agent

import uvicorn              # The server that runs our app
import os                   # To access environment variables
import json                 # For handling JSON data
import ast                  # safe literal_eval fallback
import requests             # For downloading files and submitting answers
import time                 # For any necessary pauses
import io                   # For capturing print output
import contextlib           # Used to capture print() output from exec
import re                   # Import regex for the agent
import traceback

from dotenv import load_dotenv  # To load our .env file

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

# --- NEW Imports ---
# Groq client (make sure `groq` is in requirements.txt)
from groq import Groq
import pandas as pd             # For data analysis
import pdfplumber               # For reading PDFs

# --- 1. Load Configuration ---
load_dotenv(override=True, dotenv_path='.env')  # Load variables from the .env file (local only)

# Enhanced debug
print("=" * 60)
print("DEBUG: Checking API Key and .env location")
print("=" * 60)
print(f"Current working directory: {os.getcwd()}")
print(f".env file exists here: {os.path.exists('.env')}")

MY_SECRET_KEY = os.getenv("MY_SECRET_KEY")
MY_EMAIL = os.getenv("MY_EMAIL")
AIPES_API_KEY = os.getenv("AIPES_API_KEY")
AIPES_BASE_URL = os.getenv("AIPES_BASE_URL")  # optional for Groq; kept for backwards compatibility
MODEL_NAME = os.getenv("MODEL", "llama-3.1-8b-instant")

print(f"AIPES_API_KEY present: {bool(AIPES_API_KEY)}")
print(f"AIPES_BASE_URL present: {bool(AIPES_BASE_URL)}")
print(f"MY_EMAIL present: {bool(MY_EMAIL)}")
print("=" * 60)

# Minimal required check
if not all([MY_SECRET_KEY, MY_EMAIL, AIPES_API_KEY]):
    raise ValueError("FATAL ERROR: One or more environment variables (AIPES_API_KEY, MY_EMAIL, MY_SECRET_KEY) are missing.")

# --- Initialize Groq Client ("The Brain") ---
try:
    llm_client = Groq(api_key=AIPES_API_KEY)
    print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    llm_client = None

# --- 2. Define Data Models ---
class QuizTask(BaseModel):
    email: str
    secret: str
    url: HttpUrl

# --- 3. The "Toolbox" (Python Functions) ---
def read_web_page_tool(url: str) -> str:
    """Visits a URL with a headless browser and returns all visible text."""
    print(f"[Tool Call]: read_web_page_tool(url='{url}')")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        page_text = driver.find_element(By.TAG_NAME, "body").text
        print(f"[Tool Result]: Scraped text (first 150 chars): {page_text[:150]}...")
        return page_text
    except Exception as e:
        print(f"[Tool Error]: read_web_page_tool failed: {e}")
        return f"Error: Could not read page. {e}"
    finally:
        if driver:
            driver.quit()

def find_links_tool(url: str) -> str:
    """Visits a URL and returns a JSON string of all links {'text': '...', 'href': '...'} found on the page."""
    print(f"[Tool Call]: find_links_tool(url='{url}')")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
        links = driver.find_elements(By.TAG_NAME, 'a')
        link_list = []
        for link in links:
            href = link.get_attribute('href')
            if href:
                link_list.append({"text": link.text, "href": href})
        result = json.dumps(link_list)
        print(f"[Tool Result]: Found {len(link_list)} links. (first 150 chars): {result[:150]}...")
        return result
    except Exception as e:
        print(f"[Tool Error]: find_links_tool failed: {e}")
        return f"Error: Could not find links. {e}"
    finally:
        if driver:
            driver.quit()

def download_file_tool(url: str, filename: str) -> str:
    """Downloads a file from a URL and saves it locally. Handles raw URLs or JSON link lists."""
    print(f"[Tool Call]: download_file_tool(url='{url}', filename='{filename}')")
    clean_url = url.strip()
    if clean_url.startswith("[") or clean_url.startswith("{"):
        print("[Tool Logic]: Input looks like a JSON string. Attempting to extract URL...")
        try:
            data = json.loads(clean_url)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "href" in data[0]:
                clean_url = data[0]["href"]
            elif isinstance(data, dict) and "href" in data:
                clean_url = data["href"]
            print(f"[Tool Logic]: Extracted URL: {clean_url}")
        except Exception as e:
            print(f"[Tool Logic]: JSON parsing failed ({e}). Treating as raw string.")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(clean_url, headers=headers, timeout=30)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"[Tool Result]: File saved as {filename}")
        return f"Success: File downloaded and saved as {filename}"
    except Exception as e:
        print(f"[Tool Error]: download_file_tool failed: {e}")
        return f"Error: Could not download file. {e}"

def read_pdf_tool(filename: str) -> str:
    """Reads all text from a local PDF file."""
    print(f"[Tool Call]: read_pdf_tool(filename='{filename}')")
    try:
        full_text = ""
        with pdfplumber.open(filename) as pdf:
            for i, page in enumerate(pdf.pages):
                extracted = page.extract_text() or ""
                full_text += f"\n--- PDF Page {i+1} ---\n{extracted}"
        print(f"[Tool Result]: Extracted text (first 150 chars): {full_text[:150]}...")
        return full_text
    except Exception as e:
        print(f"[Tool Error]: read_pdf_tool failed: {e}")
        return f"Error: Could not read PDF. {e}"

def run_python_tool(code_to_run: str, text_input: str = None) -> str:
    r"""Executes a string of Python code and returns its print() output.

    The code MUST NOT call input() or block. The agent must pass any previous results via the `text_input` variable.
    """
    print(f"[Tool Call]: run_python_tool(code_snippet_length={len(code_to_run)})")
    # Safety checks to prevent blocking or dangerous calls
    if "input(" in code_to_run:
        err = "Error: Code attempts to call input(), which is not allowed in automated execution."
        print(f"[Tool Error]: {err}")
        return f"Error: Code execution blocked. {err}"
    # Prepare safe globals/locals for exec
    safe_globals = {
        "pd": pd,
        "re": re,
        "json": json,
        "__name__": "__run_python_tool__",
    }
    local_scope = {"text_input": text_input}
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            # Execute the user code
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
        # If payload is a dict already, use it
        if isinstance(answer_payload, dict):
            data = answer_payload
        elif isinstance(answer_payload, str):
            # Try JSON parse
            try:
                data = json.loads(answer_payload)
            except Exception:
                # Fallback: try ast.literal_eval for Python-style dict strings
                try:
                    data = ast.literal_eval(answer_payload)
                except Exception as e:
                    print(f"[Tool Error]: Could not parse answer_payload string as JSON or Python dict: {e}")
                    return f"Error: Could not parse answer_payload. {e}"
        else:
            print(f"[Tool Error]: Unsupported answer_payload type: {type(answer_payload)}")
            return f"Error: Unsupported answer_payload type: {type(answer_payload)}"

        # Basic sanity check
        if not isinstance(data, dict):
            return "Error: Submission payload is not a JSON object."

        response = requests.post(submit_url, json=data, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        print(f"[Tool Result]: Submission successful. Response: {response_json}")
        return json.dumps(response_json)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[Tool Error]: submit_answer_tool failed: {e}\n{tb}")
        return f"Error: Answer submission failed. {e}"

# --- Tool Definitions for the LLM (safe instructions) ---
TOOLS_DEFINITION = r"""
[
  {
    "name": "read_web_page_tool",
    "description": "Visits a URL with a headless browser and returns all visible text. Use this to get the *text* of the quiz.",
    "parameters": {"url": "The URL to scrape"}
  },
  {
    "name": "find_links_tool",
    "description": "Visits a URL and returns a JSON list of all links on the page. Use this instead of read_web_page_tool if you need a download link.",
    "parameters": {"url": "The URL to scrape for links"}
  },
  {
    "name": "download_file_tool",
    "description": "Downloads a file from a URL and saves it locally.",
    "parameters": {"url": "The file URL", "filename": "Local filename"}
  },
  {
    "name": "read_pdf_tool",
    "description": "Reads all text from a local PDF file.",
    "parameters": {"filename": "Local PDF filename"}
  },
  {
    "name": "run_python_tool",
    "description": "Executes Python code. The code MUST read from a predefined variable named 'text_input' if needed. DO NOT embed <last_result> inside code_to_run. Instead ALWAYS pass it via the 'text_input' parameter like: { 'text_input': '<last_result>' }. Example valid code: code_to_run='import json\\n data=json.loads(text_input)\\n print(data[\"email\"])'",
    "parameters": {
      "code_to_run": "Python code as a string. Must print the answer.",
      "text_input": "(Optional) The previous result. ALWAYS pass '<last_result>' here instead of embedding inside code."
    }
  },
  {
    "name": "submit_answer_tool",
    "description": "Submits the final answer to the quiz submission URL. Must be the last step.",
    "parameters": {
      "submit_url": "Submission URL",
      "answer_payload": "Full JSON payload including email, secret, url, answer."
    }
  }
]
"""

# Map tool names to implementations
TOOLS_MAP = {
    "read_web_page_tool": read_web_page_tool,
    "find_links_tool": find_links_tool,
    "download_file_tool": download_file_tool,
    "read_pdf_tool": read_pdf_tool,
    "run_python_tool": run_python_tool,
    "submit_answer_tool": submit_answer_tool,
}

# --- LLM "Brain" Function ---
def call_llm_brain(scraped_text: str, current_task_url: str, email: str, secret: str, previous_error: str | None = None) -> list[dict]:
    """Ask Groq to produce a JSON plan (list of tool calls). Returns parsed list of tool calls or [] on failure."""
    if llm_client is None:
        print("[Brain Error]: LLM client is not initialized.")
        return []

    # Strict system prompt that forbids embedding <last_result> inside code strings
    system_prompt = rf"""
You are an autonomous tool-using agent. You MUST output ONLY valid JSON (no prose).
You will receive a webpage's text. Produce a JSON list of steps to solve the quiz using the following tools: {TOOLS_DEFINITION}

CRITICAL RULES (follow exactly):
1) Output must be a valid JSON array (list) of objects. No extra text before/after.
2) Each step object must use keys: "name" (tool name) and "parameters" (object).
3) NEVER embed the literal <last_result> inside a code string. Instead, if a step needs previous text as input, use:
   "text_input": "<last_result>"
   in the parameters for run_python_tool.
4) run_python_tool.code_to_run MUST reference text_input (if needed) and MUST print the final JSON answer using json.dumps({...}).
   Example (valid):
     {"name":"run_python_tool","parameters":{"code_to_run":"import json\\nobj=json.loads(text_input)\\nprint(json.dumps({'answer':obj['value']}))","text_input":"<last_result>"}}
5) submit_answer_tool must be the last step and MUST receive its "answer_payload" as a PRE-BUILT JSON string (i.e., the previous run_python_tool should print the exact JSON to submit, then submit_answer_tool should use "<last_result>" for its payload).
   Example:
     {"name":"submit_answer_tool","parameters":{"submit_url":"https://example/submit","answer_payload":"<last_result>"}}
6) JSON must use double quotes for keys and strings.
7) Do NOT use input(), do not request human interaction.
8) Keep Python code short and use json.loads(text_input) or re on text_input. Do not try to re-embed raw HTML or JSON into Python strings.

When you receive previous_error, analyze it and produce a corrected plan.
Respond with only the JSON array.
"""

    user_prompt = f"Here is the quiz text from {current_task_url}:\n---\n{scraped_text}\n---\n"
    if previous_error:
        user_prompt += f"Last error: {previous_error}\nPlease produce a corrected JSON plan."

    print("\n[Brain]: Calling LLM to get a plan...")

    try:
        # call Groq chat API (robust extraction)
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        # Try multiple ways to extract text depending on response shape
        plan_json = None
        try:
            # common Groq shape: response.choices[0].message.content
            plan_json = getattr(response.choices[0].message, "content", None)
        except Exception:
            plan_json = None

        if not plan_json:
            # try dict-like access
            try:
                plan_json = response.choices[0].message["content"]
            except Exception:
                plan_json = None

        if not plan_json:
            # fallback to text or string
            plan_json = getattr(response, "text", None) or str(response)

        plan_json = plan_json or ""
        print(f"\n[Brain DEBUG]: Raw response extracted (truncated):\n{plan_json[:1000]}\n")

        # Strip code fences
        plan_json = plan_json.strip()
        if plan_json.startswith("```json"):
            plan_json = plan_json[7:]
        if plan_json.startswith("```"):
            plan_json = plan_json[3:]
        if plan_json.endswith("```"):
            plan_json = plan_json[:-3]
        plan_json = plan_json.strip()

        print(f"[Brain DEBUG]: Cleaned JSON (first 1000 chars):\n{plan_json[:1000]}\n")

        # Parse JSON
        plan_data = json.loads(plan_json)
    except Exception as e:
        print(f"[Brain Error]: LLM call or JSON parse failed: {e}")
        # provide as much debug as possible
        try:
            print(f"[Brain Debug Raw Resp]: {response}")
        except Exception:
            pass
        return []

    # Normalize into a list of tool calls
    raw_tool_calls = []
    if isinstance(plan_data, list):
        raw_tool_calls = plan_data
    elif isinstance(plan_data, dict):
        # try to find a key with list
        for k, v in plan_data.items():
            if isinstance(v, list):
                raw_tool_calls = v
                break
        if not raw_tool_calls:
            # if dict-of-tools, convert dict values
            for k in sorted(plan_data.keys()):
                if isinstance(plan_data[k], dict):
                    raw_tool_calls.append(plan_data[k])
    else:
        print("[Brain DEBUG]: Plan JSON not list/dict.")
        return []

    # Convert into normalized plan structure
    plan = []
    for tool_call in raw_tool_calls:
        if not isinstance(tool_call, dict):
            print(f"[Brain DEBUG]: Skipping non-dict plan item: {tool_call}")
            continue
        tool_name = None
        tool_args = None
        if "name" in tool_call and "parameters" in tool_call:
            tool_name = tool_call["name"]
            tool_args = tool_call["parameters"]
        elif "tool" in tool_call and "parameters" in tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call["parameters"]
        elif "tool" in tool_call and "args" in tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]
        if tool_name and tool_args is not None:
            plan.append({"tool": tool_name, "args": tool_args})
        else:
            print(f"[Brain DEBUG]: Skipping invalid structure: {tool_call}")

    if not plan:
        print("[Brain Error]: No valid steps found in plan after normalization.")
        return []

    print(f"[Brain]: Received plan with {len(plan)} steps.")
    return plan

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

            scraped_text = read_web_page_tool(current_task_url)
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
                continue

            last_tool_output = None
            plan_failed = False

            for i, step in enumerate(plan):
                tool_name = step.get("tool")
                tool_args = step.get("args", {})
                print(f"\n[Agent]: Executing Step {i+1}: {tool_name}")

                if tool_name in TOOLS_MAP:
                    tool_function = TOOLS_MAP[tool_name]

                    # Inject last_tool_output recursively into args where '<last_result>' appears
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
                                        print(f"[Agent]: CORRECT! Received new quiz URL: {new_url}")
                                    else:
                                        reason = submit_response.get("reason", "No reason given.")
                                        print(f"[Agent]: Answer incorrect (Reason: {reason}), but a new URL was provided. Proceeding to: {new_url}")
                                    current_task_url = new_url
                                else:
                                    reason = submit_response.get("reason", "No reason given.")
                                    print(f"[Agent]: Submission response had no new URL (Reason: {reason}). Ending quiz chain.")
                                    current_task_url = None
                                plan_failed = False
                                break
                            except Exception as e:
                                print(f"[Agent Error]: Failed to parse submission response: {e}")
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
TEST_MODE = False  # SET TO True FOR TESTING
TEST_URL = "https://tds-llm-analysis.s-anand.net/demo"

if __name__ == "__main__":
    if not llm_client:
        print("Could not start server, LLM client failed to initialize.")
    elif TEST_MODE:
        print(f"--- STARTING IN TEST MODE ---")
        print(f"Test URL: {TEST_URL}")
        if not MY_EMAIL or not MY_SECRET_KEY:
            print("TEST MODE FAILED: MY_EMAIL or MY_SECRET_KEY not in .env file.")
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
