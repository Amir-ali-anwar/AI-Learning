import json
import os

path = r'e:\my-learning\AI-Learning\LLM-Finetuning\LLM-Finetuning-HuggingFace\huggingface_crash_course.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if any('from dotenv import load_dotenv' in line for line in source):
            cell['source'] = [
                "import os\n",
                "from dotenv import load_dotenv\n",
                "from huggingface_hub import login\n",
                "\n",
                "load_dotenv()\n",
                "# We use HF_TOKEN_READ from our .env file\n",
                "hf_token = os.getenv('HF_TOKEN_READ')\n",
                "if hf_token:\n",
                "    print('Successfully loaded token from .env')\n",
                "    login(token=hf_token)\n",
                "else:\n",
                "    print('HF_TOKEN_READ not found in .env')\n"
            ]
            cell['execution_count'] = None
            cell['outputs'] = []

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook successfully updated with proper Hugging Face login!')
