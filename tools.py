import asyncio
import httpx
import json
import io
import contextlib
import os
import sys
import pandas as pd
import re
import pdfplumber

# --- Constants ---
# Submission details
SUBMISSION_URL = "https://tds-llm-analysis.s-anand.net/submit" 

# --- 1. Web Scraping Tool (Async) ---
async def scrape_page(page, url: str) -> str:
    """Visits a URL and returns all visible text using Playwright."""
    print(f"🌐 [TOOL: SCRAPE] Visiting {url}...")
    try:
        await page.goto(url, wait_until="domcontentloaded")
        
        # Ensure the page fully loads, waiting for common content
        await page.wait_for_selector('body')
        
        content = await page.inner_text('body')
        
        # Clean up common excessive newlines/spaces
        text = ' '.join(content.split())
        print(f"🌐 [TOOL: SCRAPE] Success. Content length: {len(text)}. (First 100 chars: {text[:100]}...)")
        return text
    except Exception as e:
        error_msg = f"Error during scraping {url}: {e}"
        print(f"🌐 [TOOL: SCRAPE] {error_msg}")
        return error_msg

# --- 2. File Download Tool (Async) ---
async def download_file(url: str, filename: str) -> str:
    """Downloads a file from a URL using httpx."""
    print(f"⬇️ [TOOL: DOWNLOAD] Attempting to download {url} to {filename}...")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            success_msg = f"File successfully downloaded and saved as {filename} (Size: {len(response.content)} bytes)."
            print(f"⬇️ [TOOL: DOWNLOAD] {success_msg}")
            return success_msg
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP Error {e.response.status_code} during download from {url}. Reason: {e.response.text[:100]}"
        print(f"⬇️ [TOOL: DOWNLOAD] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Generic error during download from {url}: {e}"
        print(f"⬇️ [TOOL: DOWNLOAD] {error_msg}")
        return error_msg

# --- 3. PDF Reading Tool (Sync) ---
def read_pdf(filename: str) -> str:
    """Reads all text from a local PDF file using pdfplumber."""
    print(f"📄 [TOOL: READ_PDF] Reading text from {filename}...")
    if not os.path.exists(filename):
        error_msg = f"File not found: {filename}"
        print(f"📄 [TOOL: READ_PDF] {error_msg}")
        return error_msg
        
    full_text = ""
    try:
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                full_text += page.extract_text() or "" # Handles pages with no text
        
        print(f"📄 [TOOL: READ_PDF] Success. Extracted text (first 100 chars): {full_text[:100].strip()}...")
        return full_text
    except Exception as e:
        error_msg = f"Error reading PDF {filename}: {e}"
        print(f"📄 [TOOL: READ_PDF] {error_msg}")
        return error_msg

# --- 4. Python Execution Tool (Sync) ---
def execute_python_code(code_to_run: str, text_input: str = None) -> str:
    """Executes a string of Python code and returns its print() output."""
    # Ensure necessary modules are available in the execution scope
    local_scope = {"pd": pd, "re": re, "text_input": text_input}
    
    stream = io.StringIO()
    print(f"🐍 [TOOL: PYTHON] Executing code...")
    try:
        # Redirect stdout to capture the print() output
        with contextlib.redirect_stdout(stream):
            # Execute code using a limited global scope (only necessary modules)
            exec(code_to_run, {"pd": pd, "re": re, "os": os}, local_scope)
        
        result = stream.getvalue().strip()
        
        if not result:
            result = "Code executed successfully, but produced no print() output."
        
        return str(result)
    except Exception as e:
        error_msg = f"Code execution failed. Traceback (first 200 chars): {str(e)[:200]}"
        print(f"🐍 [TOOL: PYTHON] Error: {error_msg}")
        return f"Error: {error_msg}"

# --- 5. Submission Tool (Async) ---
async def submit_quiz(submit_url: str, payload: dict) -> dict:
    """Posts the final JSON answer to the submission URL."""
    # Remove the 'answer' field if it's not a simple type, usually a sign of failure
    answer_value = payload.get('answer')
    if isinstance(answer_value, (dict, list)):
        # This usually means the extraction step failed and returned a complex object
        return {"correct": False, "reason": "PYTHON tool did not return a clean scalar answer."}
        
    print(f"✅ [TOOL: SUBMIT] Submitting answer to {submit_url}...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(submit_url, json=payload)
            response.raise_for_status()
            
            response_json = response.json()
            print(f"✅ [TOOL: SUBMIT] Response: {response_json}")
            return response_json
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP Error {e.response.status_code} during submission. Response: {e.response.text[:200]}"
        print(f"✅ [TOOL: SUBMIT] {error_msg}")
        # Return a standard failure format for the agent to parse
        return {"correct": False, "reason": error_msg}
    except Exception as e:
        error_msg = f"Generic error during submission: {e}"
        print(f"✅ [TOOL: SUBMIT] {error_msg}")
        return {"correct": False, "reason": error_msg}