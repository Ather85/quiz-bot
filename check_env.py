import google.generativeai as genai

genai.configure(api_key="AIzaSyC5H9SzVmISVmhvq__CmA3PQZheVco4ZIc")

print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")