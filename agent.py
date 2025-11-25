import os
import json
import asyncio
import google.generativeai as genai
from playwright.async_api import async_playwright
# We need to import load_dotenv to correctly load environment variables from the .env file
from dotenv import load_dotenv
import tools

# Load environment variables from a .env file if it exists
load_dotenv()

# --- CONFIGURATION ---
# The 'genai' client will inherit the global configuration set in main.py.

# FIX: Changing model name from 'gemini-1.5-flash' to the more general and widely available 'gemini-2.5-flash'
model = genai.GenerativeModel('gemini-2.5-flash',
    generation_config={"response_mime_type": "application/json"})

async def run_agent_loop(start_url: str, email: str, secret: str):
    """The main execution loop for the Quiz Solving Agent."""
    print(f"🤖 [AGENT] Starting Task for {start_url}")

    # Initialize Playwright browser
    async with async_playwright() as p:
        # Launch Chromium in headless mode (no visible browser window)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        current_url = start_url
        history = [] # Short-term memory of actions and results

        # Main Loop: We set a limit to 15 steps to prevent infinite loops on complex problems
        # FIX: Increased step limit from 10 to 15 to allow completion of multi-step challenges.
        for step in range(15):

            # 1. Construct the Prompt (The Agent's System Instruction)
            # This is where we define the agent's role, goal, and available tools.
            # CRITICAL: We now include the current URL in the challenge strategy to force context switching.
            system_msg = f"""
            You are an automated agent solving a data quiz.
            Your Goal: Solve the quiz at {current_url} and get the next URL or the final success message.

            TOOLS AVAILABLE:
            1. SCRAPE(url): Get text from page. Default url is {current_url}.
            2. DOWNLOAD(url, filename): Download a file found in the text.
            3. READ_PDF(filename): Read text from a local PDF.
            4. PYTHON(code): Run python code (pandas/re). The code MUST print() the final result (the answer to the quiz question).
            5. SUBMIT(url, answer): Submit final answer. The 'url' must be the FULL absolute URL (e.g., 'https://...') extracted from the quiz page.

            HISTORY (Last 3 actions/results):
            {json.dumps(history[-3:])}

            INSTRUCTIONS:
            - Analyze the current page content (or last result) and decide the NEXT logical step.

            - **CHALLENGE STRATEGY BASED ON URL: {current_url}**
              - If the URL contains 'audio': The immediate goal is file processing.
                1. SCRAPE the current page to find the file URL. (Done in Step 7)
                2. **CRITICAL STEP: URL EXTRACTION.** You MUST use PYTHON to extract the *full absolute file URL* (starting with 'https://...') from the scraped content. The correct regex is `r'(https://tds-llm-analysis\\.s-anand\\.net/demo-audio/\\d+\\.csv)'`.
                3. Use DOWNLOAD with the **extracted URL** and an appropriate filename (e.g., 'audio_data.csv').
                4. **CRITICAL STEP: DATA PROCESSING.** After successful download, use PYTHON to load the file, calculate the sum of the 'number' column, and print the result. **USE THE EXACT FILENAME 'audio_data.csv'.**
                    ```python
                    import pandas as pd
                    df = pd.read_csv('audio_data.csv')
                    total_sum = df['number'].sum()
                    print(total_sum)
                    ```
                5. **ABSOLUTE RULE:** If the URL contains 'audio', DO NOT submit '4122' or guess the answer. Follow steps 1-4 precisely.
              - If the URL contains 'scrape', follow the extraction flow below.

            - **CRITICAL TWO-STEP SUBMISSION FLOW (For Scrape/Data challenges):**
              1. **EXTRACTION:** If the HISTORY shows the answer (e.g., 'Secret code is XXXX'), you MUST use the PYTHON tool FIRST to cleanly extract just the number or string (e.g., '4122') and print it.
              2. **SUBMISSION:** If the HISTORY shows the result of a successful PYTHON extraction (just the clean answer), you MUST use the SUBMIT tool immediately.

            - **URL RULE:** The SUBMIT tool 'url' argument MUST be the FULL absolute URL, typically 'https://tds-llm-analysis.s-anand.net/submit'. Do NOT use relative paths like '/submit'.

            - Return a JSON object with ONE action: {{ "tool": "NAME", "args": {{...}} }}
            - Do not use SCRAPE repeatedly if the content is already in HISTORY.
            """

            try:
                # 2. Ask Gemini what to do (The Reasoning Step)
                print(f"🤔 [AGENT] Thinking (Step {step})...")

                # Gemini will return a structured JSON response based on the prompt
                response = model.generate_content(system_msg)
                action = json.loads(response.text)
                print(f"👉 [AGENT] Action: {action.get('tool')}")

                # 3. Execute the Tool (The Action Step)
                result = ""
                tool = action.get('tool', '').upper()
                args = action.get('args', {})

                if tool == "SCRAPE":
                    target = args.get('url', current_url)
                    # FIX: Prevent updating current_url if scraping a temporary data source.
                    # This ensures the main challenge URL is preserved for the SUBMIT payload.
                    is_data_url = any(keyword in target for keyword in ['-data', 'file', 'download', 'csv', 'pdf', 'audio'])

                    if target == current_url:
                        # Continue if scraping the main challenge page
                        pass
                    elif not is_data_url:
                        # Only update if the URL is different AND it's not a temporary data URL
                        current_url = target

                    result = await tools.scrape_page(page, target)

                elif tool == "DOWNLOAD":
                    # Added explicit check for URL format
                    download_url = args.get('url')
                    # Use a fixed, unique name for the audio challenge file to ensure consistency
                    filename = args.get('filename', 'audio_data.csv')

                    if not download_url or not (download_url.startswith('http://') or download_url.startswith('https://')):
                         result = f"Error: Download URL '{download_url}' is not a full absolute URL with a scheme (http/https)."
                    else:
                        # Pass the fixed filename
                        result = await tools.download_file(download_url, filename)

                elif tool == "READ_PDF":
                    result = tools.read_pdf(args.get('filename'))

                elif tool == "PYTHON":
                    # Pass previous result as context for the Python code
                    last_text = history[-1]['result'] if history else ""
                    result = tools.execute_python_code(args.get('code'), text_input=last_text)
                    print(f"🐍 [PYTHON OUTPUT] {result[:200].strip()}...")

                elif tool == "SUBMIT":
                    # Added explicit check for URL format, though the LLM is now instructed to do this
                    submit_url = args.get('url')
                    if not submit_url or not (submit_url.startswith('http://') or submit_url.startswith('https://')):
                         result = f"Error: Submission URL '{submit_url}' is not a full absolute URL with a scheme (http/https). Please correct the URL in the SUBMIT action."
                    else:
                        payload = {
                            "email": email,
                            "secret": secret,
                            "url": current_url, # CRITICAL: This now correctly uses the main challenge URL
                            "answer": args.get('answer')
                        }
                        submit_res = await tools.submit_quiz(submit_url, payload)
                        print(f"✅ [SUBMIT RESULT] {submit_res}")

                        # Check submission result
                        if submit_res.get('correct') == True:
                            new_url = submit_res.get('url')
                            if new_url:
                                print(f"🎉 Correct! Moving to next quiz: {new_url}")
                                current_url = new_url
                                history = [] # Reset history for the new challenge
                                continue # Skip history update for next step
                            else:
                                print("🏆 QUIZ FINISHED SUCCESSFULLY!")
                                break # Agent done
                        else:
                            result = f"Submission failed. Reason: {submit_res.get('reason')}"

                # 4. Update History (Observation Step)
                history.append({"step": step, "tool": tool, "result": str(result)[:800]})

            except Exception as e:
                print(f"❌ [AGENT ERROR] {e}")
                history.append({"error": str(e)})

        await browser.close()


if __name__ == "__main__":
    # --- Agent Execution ---
    # Retrieve configuration from environment variables
    START_URL = os.environ.get('START_URL')
    USER_EMAIL = os.environ.get('USER_EMAIL')
    USER_SECRET = os.environ.get('USER_SECRET')

    if not all([START_URL, USER_EMAIL, USER_SECRET]):
        print("❌ [AGENT] Missing environment variables. Please ensure START_URL, USER_EMAIL, and USER_SECRET are set in your environment or .env file.")
    else:
        # Start the asynchronous event loop and run the agent
        asyncio.run(run_agent_loop(START_URL, USER_EMAIL, USER_SECRET))