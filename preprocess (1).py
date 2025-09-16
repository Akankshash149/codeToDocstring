from datasets import load_dataset
import ast, re, uuid, json, pathlib
from unidecode import unidecode
import warnings
from transformers import AutoTokenizer


ds = load_dataset("Nan-Do/code-search-net-python")

split = ds["train"].train_test_split(test_size=0.2, seed=42)
test_valid = split["test"].train_test_split(test_size=0.5, seed=42)

ds_splits = {
    "train": split["train"],
    "valid": test_valid["train"],
    "test": test_valid["test"],
}


print("Total dataset size : " , len(ds['train']))
print("Train size : " , len(ds_splits['train']))
print("Validate size : " , len(ds_splits['valid']))
print("Test size : " , len(ds_splits['test']))


def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n")
    s = s.strip()
    s = unidecode(s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def py_extract(src: str):
    try:
        tree = ast.parse(src)
        
        # finding the func deifition
        func = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func = node
                break
        

        # print (tree)
        #  break
        if func is None:
            return None, [], None, False
        
        # getting param names
        param_names = []
        for arg in func.args.args:
            param_names.append(arg.arg)
        
        # building the template for the function signature
        func_sig = f"def {func.name}({', '.join(param_names)}):"
        
        # Remove docstring if present
        source_lines = src.splitlines()
        cleaned_code = src
        
        if len(func.body) > 0:
            first_stmt = func.body[0]
            # check if first statement is a docstring
            if (isinstance(first_stmt, ast.Expr) and 
                isinstance(first_stmt.value, ast.Constant) and
                isinstance(first_stmt.value.value, str)):
                
                doc_start = first_stmt.lineno - 1
                doc_end = getattr(first_stmt, "end_lineno", first_stmt.lineno) - 1
                
                # filter out docstring lines
                filtered_lines = []
                for i, line in enumerate(source_lines):
                    if i < doc_start or i > doc_end:
                        filtered_lines.append(line)
                
                cleaned_code = "\n".join(filtered_lines)
        
        # check if function has return statements
        has_returns = False
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                has_returns = True
                break
        
        return func_sig, param_names, cleaned_code, has_returns
        
    except:
        return None, [], None, False


def row_to_record(row):
    sig, params, code_wo_doc, has_ret = py_extract(normalize_text(row["original_string"]))
    if not sig or not code_wo_doc:
        return None

    rec = {
        "id": str(uuid.uuid4()),
        "lang": (row.get("language") or "python").lower(),
        "func_name": row.get("func_name", ""),
        "signature": sig,
        "code": code_wo_doc,
        "target_docstring": normalize_text(row.get("docstring") or ""),
        "has_return": has_ret,
    }
    return rec

warnings.filterwarnings("ignore", category=SyntaxWarning)
out = pathlib.Path("data_docgen")
out.mkdir(parents=True, exist_ok=True)

for split_name, dataset in ds_splits.items():
    records = []
    counter = 0
    for i, ex in enumerate(dataset):
        counter = counter + 1
        #if counter == 60000: 
         # break
        if len(records) % 5000 == 0:
            print(split_name, "processed:", len(records))
        rec = row_to_record(ex)
        if rec:
            records.append(rec)

    with open(out / f"{split_name}_main.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} with {len(records)} records")


PROMPT_TEMPLATE = """### Instruction:
Generate a concise Python docstring for the following function:

### Code:
{code}

### Response:
"""

def format_example(example):
    prompt = PROMPT_TEMPLATE.format(code=example["code"])
    return {
        "prompt": prompt,
        "docstring": example["target_docstring"]
    }

TRAIN_MAX_CAP = 50000

# isolate run this 

# comment out the preproccesor part
data_files = {
    "train": "data_docgen/train_main.jsonl",
    "validation": "data_docgen/valid_main.jsonl"
}
raw_datasets = load_dataset("json", data_files=data_files)
processed = raw_datasets.map(format_example)
len(processed["train"])
processed["train"] = processed["train"].select(range(TRAIN_MAX_CAP))

len(processed["train"])


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "qwen25-docgen-lora"
MAX_LEN = 512   # shorten to 512 for speed



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



def tokenize(example):
    prompt   = example["prompt"]
    target   = example["docstring"]

    #  prompt + target + eos token suffixed at hte end
    full_txt = prompt + target + tokenizer.eos_token

    # tokenizing the entire sequence
    tok = tokenizer(
        full_txt,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    # tokenizing prompt separately 
    prompt_ids = tokenizer(
        prompt,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    labels = tok["input_ids"][:]

    # mask prompt tokens with padding_tokens
    prompt_len = sum(1 for t in prompt_ids if t != tokenizer.pad_token_id)
    labels[:prompt_len] = [-100] * prompt_len

    tok["labels"] = labels
    return tok


# tokenizing and saving
tokenized = processed.map(tokenize, remove_columns=processed["train"].column_names)
tokenized.save_to_disk("tokenized_docgen")