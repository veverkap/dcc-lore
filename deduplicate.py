import re
from pathlib import Path

def flatten_and_deduplicate(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    unique_sentences = []
    seen_normalized = set()

    # We will preserve the first valid frontmatter block we find for Quartz metadata
    frontmatter = []
    in_frontmatter = False
    frontmatter_captured = False

    for line in lines:
        stripped = line.strip()

        # Handle capturing the very first frontmatter block for the clean file
        if stripped == "---":
            if not frontmatter_captured:
                in_frontmatter = not in_frontmatter
                frontmatter.append(line)
                if not in_frontmatter:
                    frontmatter_captured = True
            continue
        
        if in_frontmatter:
            frontmatter.append(line)
            continue

        # Skip structural headers and metadata noise
        if (
            not stripped or 
            stripped.startswith("title:") or 
            stripped.startswith("type:") or 
            stripped.startswith("tags:") or 
            stripped.startswith("# ") or  # Main headers
            stripped.startswith("## ")     # Section subheaders like Additional Observations
        ):
            continue

        # Normalize the sentence to check for duplicates (ignore casing and brackets)
        normalized = stripped.lower().replace("[[", "").replace("]]", "")
        
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            unique_sentences.append(line.strip())

    # Build the final clean file structure
    output_content = []
    
    # 1. Add the metadata back at the top
    if frontmatter:
        output_content.extend([fl.rstrip() for fl in frontmatter])
    
    # 2. Add a clean main title header
    # Extracts the file name (e.g., 'Anaconda') for the header
    title_header = Path(file_path).stem.capitalize()
    output_content.extend(["", f"# {title_header}", ""])
    
    # 3. Append all the unique prose sentences together as bullet points
    for sentence in unique_sentences:
        # Check if the sentence already starts with a list marker, otherwise add one
        if sentence.startswith(('*', '-', '1.')):
            output_content.append(sentence)
        else:
            output_content.append(f"* {sentence}")

    # Write out the clean narrative file
    output_path = Path(file_path).parent / f"flat_{Path(file_path).name}"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_content) + "\n")

    print(f"Flattened file saved to: {output_path}")

# Run the flattener
flatten_and_deduplicate('/Users/veverkap/Code/personal/dcc-lore/content/mechanics/anaconda.md')