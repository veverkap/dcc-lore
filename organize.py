import os
import re
from pathlib import Path
from tqdm import tqdm
from llama_cpp import Llama

# --- SETUP CONFIGURATION ---
VAULT_DIR = Path("./content/mechanics")       # Directory where your source md files are
OUTPUT_DIR = Path("./cleaned/mechanics")    # Directory where cleaned files will go
BACKUP_DIR = Path("./backup/mechanics")    # Directory to store original files as backup
MODEL_PATH = "./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" # Path to your local GGUF

OUTPUT_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# 1. INITIALIZE RAW LLAMA.CPP INFERENCE (With Auto-Download)
print("Initializing engine (will download the model if not found)...")
N_CTX = 8192          # Llama-3-8B-Instruct supports 8192 natively
MAX_OUTPUT_TOKENS = 1024
# Leave a safety margin so prompt + output never exceeds the context window
MAX_PROMPT_TOKENS = N_CTX - MAX_OUTPUT_TOKENS - 64
llm = Llama.from_pretrained(
    repo_id="MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=N_CTX,
    n_gpu_layers=-1,
    verbose=False
)

# 2. PHASE 1: DISCOVER ALL ENTITIES (OBJECTS)
print("Scanning file names to index vault objects...")
object_names = set()

for file_path in VAULT_DIR.glob("*.md"):
    # File name without extension becomes our anchor object name (e.g., "Anaconda")
    object_names.add(file_path.stem)

# Sort object names by length descending so we don't accidentally split partial words 
# (e.g., matching "Damascus Steel" before matching "Damascus")
sorted_objects = sorted(list(object_names), key=len, reverse=True)
print(len(sorted_objects), "unique objects found for backlinking.")

def generate_system_prompt():
    return (
        "You are an advanced text optimization script. Your single task is to consolidate "
        "repetitive, noisy markdown lists into a tight, clean, bulleted summary of unique factual points. "
        "CRITICAL RULES:\n"
        "1. Strip out duplicate assertions completely.\n"
        "2. Retain all existing double square-bracket links exactly as they are (e.g., [[desperado club]]).\n"
        "3. Output ONLY the clean markdown list. Do not explain your changes, do not add introductory text, "
        "and do not include frontmatter or titles. Just output the clean bullets."
    )

def inject_wiki_links(text, objects, current_file_stem):
    """
    Scans prose for text matching known file names and wraps them in [[links]].
    Skips strings that are already bracketed.
    """
    for obj in objects:
        # Don't let a file link to itself
        if obj.lower() == current_file_stem.lower():
            continue
            
        # Regex matches the object word ONLY if it is not inside double square brackets
        # Negative lookbehind (?<!\[\[) and negative lookahead (?!\]\])
        pattern = rf"(?<!\[\[)\b({re.escape(obj)})\b(?!\]\])"
        text = re.sub(pattern, r"[[\1]]", text, flags=re.IGNORECASE)
    return text

# 3. PHASE 2: PROCESSING LOOP
print(f"Beginning optimization on {len(sorted_objects)} files...")
for file_path in tqdm(VAULT_DIR.glob("*.md"), desc="Processing Files"):
    print(f"\nProcessing: {file_path.name}")

    # skip if the cleaned file already exists to avoid redundant work
    if (OUTPUT_DIR / file_path.name).exists():
        print("Cleaned file already exists, skipping...")
        continue

    # backup the original file just in case
    backup_path = BACKUP_DIR / file_path.name
    if not backup_path.exists():
        backup_path.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")
        # print(f"Original file backed up as: {backup_path.name}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    # Extract original frontmatter to prepend to the final file later
    frontmatter_match = re.match(r"^---.*?---", raw_content, flags=re.DOTALL)
    frontmatter = frontmatter_match.group(0) if frontmatter_match else ""

    # Parse out the raw text lines below the frontmatter
    body_text = re.sub(r"^---.*?---", "", raw_content, flags=re.DOTALL).strip()

    # Use the Llama.cpp engine directly to run the deduplication prompt
    def build_prompt(body: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{generate_system_prompt()}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Consolidate these lore points:\n{body}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    prompt = build_prompt(body_text)

    # If the prompt is too long for the context window, truncate the body
    # by token count and rebuild. Keep the head of the body (the bullets are
    # repetitive so the front already contains most unique facts).
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    if len(prompt_tokens) > MAX_PROMPT_TOKENS:
        body_tokens = llm.tokenize(body_text.encode("utf-8"))
        overflow = len(prompt_tokens) - MAX_PROMPT_TOKENS
        keep = max(0, len(body_tokens) - overflow)
        if keep == 0:
            print(f"Skipping {file_path.name}: body too large even after truncation.")
            continue
        truncated_bytes = llm.detokenize(body_tokens[:keep])
        body_text = truncated_bytes.decode("utf-8", errors="ignore")
        prompt = build_prompt(body_text)
        print(f"Truncated {file_path.name} body to fit context window.")

    response = llm(
        prompt,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.1, # Low temp keeps factual accuracy strict
        stop=["<|eot_id|>"]
    )
    
    clean_prose = response["choices"][0]["text"].strip()
    # print(clean_prose[:200] + "..." if len(clean_prose) > 200 else clean_prose)  # Print a preview of the cleaned text
    
    # Run the deterministic backlinking layer over the LLM output
    linked_prose = inject_wiki_links(clean_prose, sorted_objects, file_path.stem)

    # Reassemble the file architecture
    final_output = []
    if frontmatter:
        final_output.append(frontmatter)
    
    final_output.append(f"\n# {file_path.stem}\n")
    final_output.append(linked_prose)

    # Write out the clean structural file
    with open(OUTPUT_DIR / file_path.name, "w", encoding="utf-8") as f:
        f.write("\n".join(final_output) + "\n")

print(f"Batch processing completed! Cleaned files saved to: {OUTPUT_DIR.resolve()}")