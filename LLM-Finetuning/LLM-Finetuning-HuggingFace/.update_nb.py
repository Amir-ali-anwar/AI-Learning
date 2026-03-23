import json
import os

notebook_path = 'e:/my-learning/AI-Learning/LLM-Finetuning/LLM-Finetuning-HuggingFace/huggingface_crash_course.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    replaced = False
    new_source = [
        "!pip install -q python-dotenv\n",
        "import os\n",
        "from dotenv import load_dotenv\n",
        "\n",
        "# Load variables from the .env file\n",
        "load_dotenv()\n",
        "\n",
        "# Get your token\n",
        "hf_token = os.getenv('HF_TOKEN')\n",
        "print('Token loaded:', bool(hf_token))"
    ]

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'google.colab' in source:
                cell['source'] = [s for s in new_source]
                replaced = True
                break

    if not replaced:
        nb['cells'].append({
            'cell_type': 'code',
            'execution_count': None,
            'id': 'install_dotenv',
            'metadata': {},
            'outputs': [],
            'source': [s for s in new_source]
        })

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Notebook updated successfully.')
except Exception as e:
    print('Error:', e)
