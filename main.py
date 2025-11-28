import uvicorn              # The server that runs our app
import os                   # To access environment variables
import json                 # For handling JSON data
import requests             # For downloading files and submitting answers
import time                 # For any necessary pauses
import io                   # For capturing print output
import contextlib           # Used to capture print() output from exec
import re                   # Import regex for the agent

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
from openai import OpenAI       # The "Brain" (works with AIPipes)
import pandas as pd             # For data analysis
import pdfplumber               # For reading PDFs

# --- 1. Load Configuration ---
load_dotenv() # Load variables from the .env file
MY_SECRET_KEY = os.getenv("MY_SECRET_KEY")
MY_EMAIL = os.getenv("MY_EMAIL")
AIPES_API_KEY = os.getenv("AIPES_API_KEY")
AIPES_BASE_URL = os.getenv("AIPES_BASE_URL")

if not all([MY_SECRET_KEY, MY_EMAIL, AIPES_API_KEY, AIPES_BASE_URL]):
    raise ValueError("FATAL ERROR: One or more environment variables (AIPES_API_KEY, AIPES_BASE_URL, etc.) are missing from .env")

# --- Initialize OpenAI Client ("The Brain") ---
try:
    llm_client = OpenAI(
        api_key=AIPES_API_KEY,
        base_url=AIPES_BASE_URL
    )
    print(f"LLM client initialized to use custom base URL: {AIPES_BASE_URL}")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
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
            if href: # Only include links that have an href
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
    
    # Clean up URL if it came from find_links_tool
    clean_url = url.strip()
    if clean_url.startswith("[") or clean_url.startswith("{"):
        print("[Tool Logic]: Input looks like a JSON string. Attempting to extract URL...")
        try:
            data = json.loads(clean_url)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "href" in data[0]:
                clean_url = data[0]["href"]
                print(f"[Tool Logic]: Extracted URL: {clean_url}")
            elif isinstance(data, dict) and "href" in data:
                clean_url = data["href"]
                print(f"[Tool Logic]: Extracted URL: {clean_url}")
        except Exception as e:
             print(f"[Tool Logic]: JSON parsing failed ({e}). Treating as raw string.")

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(clean_url, headers=headers)
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
                full_text += f"\n--- PDF Page {i+1} ---\n{page.extract_text()}"
        print(f"[Tool Result]: Extracted text (first 150 chars): {full_text[:150]}...")
        return full_text
    except Exception as e:
        print(f"[Tool Error]: read_pdf_tool failed: {e}")
        return f"Error: Could not read PDF. {e}"

def run_python_tool(code_to_run: str, text_input: str = None) -> str:
    """Executes a string of Python code and returns its print() output."""
    print(f"[Tool Call]: run_python_tool(code='{code_to_run}')")
    
    local_scope = {"pd": pd, "re": re, "text_input": text_input}
    
    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(stream):
            exec(code_to_run, {"pd": pd, "re": re}, local_scope)
        
        result = stream.getvalue().strip()
        if not result:
            result = "Code executed successfully, but produced no print output."
        
        print(f"[Tool Result]: {result}")
        return result
    except Exception as e:
        print(f"[Tool Error]: run_python_tool failed: {e}")
        return f"Error: Code execution failed. {e}"

def submit_answer_tool(submit_url: str, answer_payload: dict) -> str:
    """Posts the final JSON answer to the submission URL."""
    print(f"[Tool Call]: submit_answer_tool(url='{submit_url}', payload={answer_payload})")
    try:
        response = requests.post(submit_url, json=answer_payload)
        response.raise_for_status()
        response_json = response.json()
        print(f"[Tool Result]: Submission successful. Response: {response_json}")
        return json.dumps(response_json)
    except Exception as e:
        print(f"[Tool Error]: submit_answer_tool failed: {e}")
        return f"Error: Answer submission failed. {e}"

# --- Tool Definitions for the LLM ---
TOOLS_DEFINITION = """
[
  {
    "name": "read_web_page_tool",
    "description": "Visits a URL with a headless browser and returns all visible text. Use this to get the *text* of the quiz.",
    "parameters": {"url": "The URL to scrape"}
  },
  {
    "name": "find_links_tool",
    "description": "Visits a URL and returns a JSON list of all links on the page. Use this *instead of* `read_web_page_tool` if you need to find a file's download URL. Example: [{'text': 'data file', 'href': '.../data.csv'}]",
    "parameters": {"url": "The URL to scrape for links"}
  },
  {
    "name": "download_file_tool",
    "description": "Downloads a file from a URL and saves it locally. Use this for PDFs, CSVs, images, etc.",
    "parameters": {"url": "The URL of the file to download", "filename": "The local filename to save it as (e.g., 'data.pdf', 'image.png')"}
  },
  {
    "name": "read_pdf_tool",
    "description": "Reads all text from a local PDF file.",
    "parameters": {"filename": "The local filename of the PDF to read"}
  },
  {
    "name": "run_python_tool",
    "description": "Executes a string of Python code to analyze data or parse text. Use 'pandas' (aliased as 'pd') for CSV/data analysis, or 're' (auto-imported) for text parsing. The code *must* print the final answer to stdout. IMPORTANT: To use the text output from a *previous step* (like `read_web_page_tool` or `read_pdf_tool`), write your code to accept a variable named `text_input`. For example: `code_to_run=\"print(re.search(r'is (\d+)', text_input).group(1))\"`. You must also pass the string `\"<last_result>\"` as the `text_input` value in your 'args'.",
    "parameters": {
        "code_to_run": "The string of Python code to execute.",
        "text_input": "(Optional) Pass the string `\"<last_result>\"` here to inject the text output of the previous step into your code."
    }
  },
  {
    "name": "submit_answer_tool",
    "description": "Submits the final answer to the quiz submission URL. This MUST be the last step.",
    "parameters": {"submit_url": "The URL to POST the answer to (found in the quiz text)", "answer_payload": "The *full* JSON payload, including email, secret, url, and the 'answer' field."}
  }
]
"""

# Dictionary to map tool names to the actual Python functions
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
    if not llm_client:
        return [{"error": "LLM client not initialized"}]

    # --- UPDATED SYSTEM PROMPT (Added Rule 12) ---
    system_prompt = f"""
    You are an autonomous data analysis agent. Your goal is to solve a quiz.
    You will be given the text from a quiz webpage.
    You must create a step-by-step plan to solve the quiz.
    Your plan must be a JSON object or list of tool calls.
    You have this toolbox available: {TOOLS_DEFINITION}
    
    RULES:
    1. The quiz text will contain the question AND the URL to submit your answer to.
    2. Your plan *must* end with a call to `submit_answer_tool`.
    3. The `answer_payload` for `submit_answer_tool` must be a complete JSON object.
       To pass the final answer, you MUST use the placeholder string "<last_result>".
       Example:
       "email": "{email}"
       "secret": "{secret}"
       "url": "{current_task_url}"
       "answer": "<last_result>"
    4. For `run_python_tool`, the code *must* print the result *to stdout*. Do not just assign to a variable.
    5. To use the output of a previous step, pass the string \"<last_result>\" as the `text_input` argument in `run_python_tool`. Your Python code will then receive that text in a variable named `text_input`.
    6. Be smart. If the quiz asks you to parse text, the plan is: [read_web_page_tool, run_python_tool (with `re`, `text_input`, and a `print` statement), submit_answer].
    7. If the quiz text mentions a file (e.g., "CSV file", "Download this") but you cannot see the full URL, you MUST use `find_links_tool` on the *current task URL* to get a list of all links, then use that to find the correct download URL for `download_file_tool`.
    8. When using `re` (regex), be robust. The text is human-readable and may have slight variations. Use flags like `re.IGNORECASE` and flexible patterns (like `r'is:? (\d+)'`) to avoid errors.
    9. If you are given a "previous_error" message, it means your last plan failed. Analyze the error and the original text, then provide a *new*, corrected plan. Do not repeat the same mistake.
    10. **Expert Tip 1:** If you see a `pandas` error like "Error tokenizing data" or "C error", the CSV is malformed. Do NOT use `error_bad_lines`. Instead, try to fix it by passing parameters to `pd.read_csv()`, such as `header=None`, `delimiter=','`, or `on_bad_lines='skip'`.
    11. **Expert Tip 2:** When analyzing a CSV or Dataframe, NEVER assume column names (like 'value' or 'cutoff'). Your Python code should ALWAYS print `df.head()` or `df.columns` first to inspect the data structure, and then proceed to filtering in the same script.
    12. Respond *only* with the JSON. No other text.
    """

    user_prompt = f"""
    Here is the quiz text from {current_task_url}:
    ---
    {scraped_text}
    ---
    """
    
    if previous_error:
        user_prompt += f"""
        ATTENTION: Your last attempt to generate a plan for this task failed.
        The error was: {previous_error}
        Please analyze this error and the quiz text, and provide a new, corrected JSON plan.
        """
    else:
        user_prompt += "Please provide the JSON plan to solve this quiz."


    print("\n[Brain]: Calling LLM to get a plan...")
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        plan_json = response.choices[0].message.content
        print(f"\n[Brain DEBUG]: Raw JSON response from LLM:\n{plan_json}\n")

        plan_data = json.loads(plan_json)
        raw_tool_calls = [] 

        if isinstance(plan_data, list):
            print("[Brain DEBUG]: Received plan as a direct list.")
            raw_tool_calls = plan_data
        elif isinstance(plan_data, dict):
            found_plan_list = False
            for key, value in plan_data.items():
                if isinstance(value, list):
                    raw_tool_calls = value
                    found_plan_list = True
                    print(f"[Brain DEBUG]: Found plan list inside key: '{key}'")
                    break
            if not found_plan_list:
                print("[Brain DEBUG]: No list key found. Assuming tool-keyed object format (e.g., 'tool_1').")
                try:
                    sorted_keys = sorted(plan_data.keys())
                except Exception:
                    sorted_keys = plan_data.keys()
                for key in sorted_keys:
                    if isinstance(plan_data[key], dict):
                        raw_tool_calls.append(plan_data[key])
                    else:
                        print(f"[Brain DEBUG]: Skipping key '{key}' - value is not a dict.")
        else:
            raise ValueError("LLM response was not a JSON list or a dict.")

        plan = [] 
        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                print(f"[Brain DEBUG]: Skipping non-dict item in plan: {tool_call}")
                continue
            tool_name, tool_args = None, None
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
                print(f"[Brain DEBUG]: Skipping invalid tool call structure: {tool_call}")

        if not plan:
            raise ValueError("LLM returned JSON, but no valid tool calls were found after parsing.")
        
        print(f"[Brain]: Received plan with {len(plan)} steps.")
        return plan

    except Exception as e:
        print(f"[Brain Error]: LLM call failed: {e}")
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
                    
                    try:
                        def inject_last_result(data):
                            if isinstance(data, dict):
                                if "answer" in data and data["answer"] == "<last_result>":
                                    converted_answer = last_tool_output
                                    try: converted_answer = int(last_tool_output)
                                    except (ValueError, TypeError):
                                        try: converted_answer = float(last_tool_output)
                                        except (ValueError, TypeError): pass
                                    print(f"[Agent]: Injected last tool output ({converted_answer}) into final answer")
                                    data["answer"] = converted_answer
                                for key, value in data.items(): data[key] = inject_last_result(value)
                            elif isinstance(data, list):
                                for i, item in enumerate(data): data[i] = inject_last_result(item)
                            elif isinstance(data, str) and data == "<last_result>":
                                print(f"[Agent]: Injected last tool output into arguments")
                                return last_tool_output
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
                                        print(f"[Agent]: Answer was incorrect (Reason: {reason}), but a new URL was provided. Proceeding to: {new_url}")
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
                        print(f"[Agent Error]: Step {i+1} ({tool_name}) crashed with exception: {e}")
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

TEST_MODE = False
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
        print(f"Starting Quiz Bot server on http://localhost:8000 (TEST_MODE is False)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
