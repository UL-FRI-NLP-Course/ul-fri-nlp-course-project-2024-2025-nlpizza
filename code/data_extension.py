import json
import re
import random
from difflib import SequenceMatcher

# ------------------------------
# Core Text Processing
# ------------------------------
def clean_text(text):
    """Clean and normalize text for comparison"""
    return re.sub(r'\s+', ' ', text.lower().strip())

def extract_core_question(text):
    """
    Extract the core question by removing template markers like Q:, Question:, etc.
    Also handles separators between question and answer parts.
    """
    # Remove leading question markers with various formats
    text = re.sub(r'^\s*(q(uestion)?|quest(ion)?)\s*[-:.,|]*\s*', '', text, flags=re.IGNORECASE)

    # Remove trailing answer markers
    text = re.sub(r'\s*[-:.,|]*\s*(a(nswer)?)\s*[-:.,|]*\s*$', '', text, flags=re.IGNORECASE)

    # Normalize internal whitespace
    text = re.sub(r'\s+', ' ', text)

    return clean_text(text)

def remove_core(text, core):
    """
    Remove the core question from text and return the remaining template wrapper.
    Works at the word level to tolerate formatting differences.
    """
    text_cleaned = clean_text(text)
    core_cleaned = clean_text(core)
    return text_cleaned.replace(core_cleaned, '').strip()


def extract_template_parts(text):
    """
    Extract template structure by identifying question/answer markers and separators.
    Returns a tuple of (question_marker, separator, answer_marker)
    """
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace

    # Identify question marker
    q_marker_match = re.match(r'^(q(uestion)?|quest(ion)?)\s*[-:.,|]*\s*', text, flags=re.IGNORECASE)
    q_marker = q_marker_match.group(0).strip() if q_marker_match else ""

    # Identify separator between question and answer
    separator_match = re.search(r'(?<=[.!?])\s*[-:.,|]*\s*(?=a(nswer)?[-:.,|]*\s*$|\s*$)', text, flags=re.IGNORECASE)
    separator = separator_match.group(0).strip() if separator_match else ""

    # Identify answer marker
    a_marker_match = re.search(r'(a(nswer)?)\s*[-:.,|]*\s*$', text, flags=re.IGNORECASE)
    a_marker = a_marker_match.group(0).strip() if a_marker_match else ""

    return (q_marker.lower(), separator.lower(), a_marker.lower())

def extract_template_wrapper(text, core_question):
    """
    Extract everything before and after the core question as the 'template wrapper'.
    Matches the core content inside the full string and returns prefix/suffix around it.
    """
    raw = text.strip()
    core = clean_text(core_question)

    # Try to find where the cleaned core exists in cleaned raw text
    # But match it in the unprocessed text using fuzzy matching
    match = re.search(re.escape(core), clean_text(raw))
    if match:
        # Try to align the span to the raw text using approximate method
        # We’ll try to locate the best matching index in raw text
        cleaned_raw = clean_text(raw)
        start_idx = cleaned_raw.find(core)
        if start_idx != -1:
            end_idx = start_idx + len(core)
            return cleaned_raw[:start_idx].strip(), cleaned_raw[end_idx:].strip()

    return "", ""


def sequence_similarity(a, b):
    """Calculate text similarity ratio between two strings"""
    return SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()

def word_error_count(a, b):
    """Count word-level differences between two strings"""
    a_words = clean_text(a).split()
    b_words = clean_text(b).split()

    # Count words with edit distance > 0
    errors = 0
    for i in range(min(len(a_words), len(b_words))):
        if a_words[i] != b_words[i] and sequence_similarity(a_words[i], b_words[i]) < 0.7:
            errors += 1

    # Add differences in length
    errors += abs(len(a_words) - len(b_words))
    return errors

# ------------------------------
# Improved Classification Logic
# ------------------------------
def classify_perturbation(variant, original):
    """
    Classify the type of perturbation between variant and original prompts.
    Returns one of: 'template_shift', 'spelling_error', 'paraphrase', or 'noise_injection'
    """
    # Extract core questions (without template markers)
    original_core = extract_core_question(original)
    variant_core = extract_core_question(variant)

    # Extract template structures
    original_template = extract_template_parts(original)
    variant_template = extract_template_parts(variant)


    if clean_text(variant) == clean_text(original):
        return "original"

    # Check if this is a pure template shift (core content identical)
    # If core content is exactly the same but templates differ
    if clean_text(original_core) == clean_text(variant_core):
        orig_wrap = remove_core(original, original_core)
        var_wrap = remove_core(variant, original_core)
        if orig_wrap != var_wrap:
            return "template_shift"


    # Check for noise injection (presence of emojis, keyboard smashes, fillers)
    if (re.search(r'[^\x00-\x7F]', variant) or  # Non-ASCII characters like emojis
        re.search(r'\b(uh+|um+|like|literally|actually|basically|just|kinda|sorta|i mean)\b', variant, re.IGNORECASE) or
        re.search(r'[a-z]{5,}', re.sub(r'\b\w+\b', '', variant))):  # Random character sequences
        return "noise_injection"

    # Calculate similarity between core questions
    core_similarity = sequence_similarity(original_core, variant_core)

    # Calculate word-level errors
    error_count = word_error_count(original_core, variant_core)

    # High similarity but with minor errors - spelling error
    if core_similarity > 0.85 or (core_similarity > 0.75 and error_count <= 3):
        return "spelling_error"

    # Semantic content changes but purpose preserved - paraphrase
    if core_similarity > 0.45:  # Some similarity preserved
        return "paraphrase"

    # Very different content - default to paraphrase
    return "paraphrase"

# ------------------------------
# Testing and Validation
# ------------------------------
def validate_classification():
    """Test the classification function with some examples"""
    test_pairs = [
        # Template shifts
        ("Q:Describe the purpose of a neural network.\nA:", "QUESTION: Describe the purpose of a neural network.\nAnswer:", "template_shift"),
        ("Q:Describe the purpose of a neural network.\nA:", "Q:  Describe the purpose of a neural network.\nA:", "template_shift"),

        # Spelling errors
        ("Q:Describe the purpose of a neural network.\nA:", "Q:Describe the purose of a neiural ntwork .\nA:", "spelling_error"),
        ("Q:Create a dialogue between two characters for a conflict resolution.\nA:", "Q:Create a dialogue betweein two characters for a conflict resolution .\nA:", "spelling_error"),

        # Paraphrases
        ("Q:Create a dialogue between two characters for a conflict resolution.\nA:", "Q: Write a discussion between two individuals to resolve a disagreement and find common understanding.\nA:", "paraphrase"),
        ("Q:Create a dialogue between two characters for a conflict resolution.\nA:", "Q: Formulate a conversation between two individuals to work through their conflicting issues.\nA:", "paraphrase"),

        # Noise injections
        ("Q:Describe the purpose of a neural network.\nA:", "Q:Describe 🕎 the purpose uhh of a neural network. 🏃‍♀️‍➡️ A:", "noise_injection"),
        ("Q:Create a dialogue between two characters for a conflict resolution.\nA:", "Q:Create a dialogue between two 🚣🏻 characters for a nwqqtfdlr conflict resolution. 👃🏻 A:", "noise_injection")
    ]

    results = []
    for original, variant, expected in test_pairs:
        actual = classify_perturbation(variant, original)
        results.append({
            "original": original,
            "variant": variant,
            "expected": expected,
            "actual": actual,
            "correct": expected == actual
        })

    return results

# ------------------------------
# Main Processing
# ------------------------------
def process_data(input_file, output_file):
    """Process the input data file and write the classified data to output"""
    with open(input_file) as f:
        original_data = json.load(f)

    extended_data = []

    for item in original_data:
        group_id = item["id"]
        prompts = item["prompts"]
        original_prompt = prompts[0].strip()

        # Step 1: classify original and variants
        for prompt in prompts:
            pert_type = classify_perturbation(prompt, original_prompt)
            extended_data.append({
                "instruction": prompt.strip(),
                "group_id": group_id,
                "perturbation": pert_type,
                "original_prompt": original_prompt
            })

        # Step 2: add noise-injected variants
        for _ in range(3):
            noisy_prompt = add_noise_to_prompt(original_prompt)
            extended_data.append({
                "instruction": noisy_prompt.strip(),
                "group_id": group_id,
                "perturbation": "noise_injection",
                "original_prompt": original_prompt
            })

    with open(output_file, "w") as f:
        json.dump(extended_data, f, indent=2)

    print(f"Classified data saved to '{output_file}'")
    return extended_data

# ------------------------------
# Noise Generation
# ------------------------------
FILLERS = {"uh", "uhh", "umm", "ummm", "like", "you know", "so", "well", "actually",
           "basically", "literally", "kinda", "sorta", "i mean", "just", "okay", "right",
           "anyway", "sooo", "i guess", "to be honest", "honestly", "technically", "hmm"}

def random_keyboard_smash(min_len=4, max_len=10):
    """Generate random keyboard smash text"""
    keys = list("asdfghjklqwertyuiopzxcvbnm")
    return ''.join(random.choice(keys) for _ in range(random.randint(min_len, max_len)))

# Pre-generate some gibberish and emoji list
GIBBERISH = [random_keyboard_smash() for _ in range(20)]
try:
    import emoji
    EMOJIS = random.sample(list(emoji.EMOJI_DATA.keys()), 30)
except ImportError:
    # Fallback if emoji module not available
    EMOJIS = ["😀", "🙂", "👍", "🎉", "🚀", "💡", "🔥", "⭐", "💯", "🤔"]

NOISE_POOL = list(FILLERS) + GIBBERISH + EMOJIS

def add_noise_to_prompt(prompt, num_insertions=3):
    """Add random noise elements to a prompt"""
    words = prompt.split()
    if len(words) < 3:
        return prompt
    insert_positions = sorted(random.sample(range(1, len(words)), min(num_insertions, len(words)-1)))
    for i, pos in enumerate(insert_positions):
        noise = random.choice(NOISE_POOL)
        words.insert(pos + i, noise)
    return ' '.join(words)

# ------------------------------
# Example Usage
# ------------------------------
if __name__ == "__main__":
    # Validate the classification algorithm
    validation_results = validate_classification()
    correct_count = sum(1 for r in validation_results if r["correct"])
    print(f"Classification validation: {correct_count}/{len(validation_results)} correct")

    # Print any incorrect classifications for debugging
    for result in validation_results:
        if not result["correct"]:
            print(f"  Misclassification:")
            print(f"  Original: {result['original']}")
            print(f"  Variant:  {result['variant']}")
            print(f"  Expected: {result['expected']}")
            print(f"  Got:      {result['actual']}")

    # Example for processing data
    process_data("alpaca_prompts.json", "alpaca_prompts_extended.json")