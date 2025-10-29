import os
import json
import numpy as np
import tiktoken
from openai import AzureOpenAI
from scipy.stats import entropy
import logging

# --- Configure Logging ---
# This helps see the outputs clearly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Your Client Setup (from your image) ---
endpoint = "https://laksh-frame.openai.azure.com/"
model_name = "gpt-4o"  # This variable isn't used by the client, but good for reference
deployment = "gpt-4o"  # This is the deployment name the client WILL use
subscription_key = "YOUR_AZURE_OPENAI_KEY_HERE" # <<< ⚠️ FILL THIS IN
api_version = "2024-05-01-preview" # Using a more standard, recent version

# Note: Your image shows 2024-12-01-preview. I've used 2024-05-01-preview
# as it's a known working version. You can change it back if needed.

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# --- 2. Tool Definitions ---
TOOLS = [
    {"name": "UnlockAccount", "description": "Unlocks a user's account in Active Directory."},
    {"name": "QueryDatabase", "description": "Queries the user incident database for similar issues."},
    {"name": "EscalateToHuman", "description": "Escalates the incident to an L2 human agent."}
]
TOOL_NAMES = [t['name'] for t in TOOLS]

def get_tool_token_ids(tool_names_list):
    """
    Gets the *first token ID* for each tool name using the gpt-4o tokenizer.
    This is CRITICAL for the logit_bias method.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base") # Tokenizer for gpt-4o
    token_map = {}
    for name in tool_names_list:
        # Get the first token of the tool name
        token_id = tokenizer.encode(name, allowed_special="all")[0]
        
        if token_id in token_map:
             logging.warning(f"Warning: Tool '{name}' and '{token_map.get(token_id)}' share the same first token ID ({token_id}). Logit bias may be ambiguous.")
        
        # Store as string since JSON keys must be strings
        token_map[str(token_id)] = name 
        logging.info(f"Tokenizer Map: '{name}' -> Token ID: {token_id}")
    return token_map

# --- 3. Method 1: Logit Inspection (The "True" Way) ---

def get_action_confidence_via_logits(user_query, tool_names_list, tool_token_map_str_keys):
    """
    Forces the model to choose *only* from the tool list and returns
    the probability distribution over that choice.
    """
    logging.info("--- Starting Method 1: Logit Inspection ---")
    
    # 1. Create the logit_bias: a dict of {token_id_str: 100}
    logit_bias = {token_id: 100.0 for token_id in tool_token_map_str_keys.keys()}

    system_prompt = f"""
    You are an agent that must decide the *single best tool* to use.
    The user will state a problem. You will respond with *only* the name
    of the single best tool to use from this list: {', '.join(tool_names_list)}
    """
    
    try:
        response = client.chat.completions.create(
            model=deployment, # Use your deployment name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            logprobs=True,      # <<< KEY: Ask for log probabilities
            top_logprobs=5,   # <<< KEY: Get the top 5 tokens (max allowed)
            logit_bias=logit_bias, # <<< KEY: Force the model to pick from our list
            max_tokens=1      # We only want the single token decision
        )

        # 2. Parse the response
        logprobs_content = response.choices[0].logprobs.content
        if not logprobs_content:
            logging.error("Logit parsing failed: No logprobs content returned.")
            return None

        # We only care about the first token's probabilities
        top_logprobs = logprobs_content[0].top_logprobs
        
        raw_probs = {}
        for logprob_entry in top_logprobs:
            # Find which of our tools appeared in the top logprobs
            if logprob_entry.token in tool_names_list:
                # logprob is log(p), so p = exp(logprob)
                raw_probs[logprob_entry.token] = np.exp(logprob_entry.logprob)
        
        # 3. Normalize to get our final [0.8, 0.1, 0.1] style array
        total_prob = sum(raw_probs.values())
        if total_prob == 0:
            logging.error("Logit parsing failed: No tool probabilities found.")
            return None
            
        normalized_probs = {tool: prob / total_prob for tool, prob in raw_probs.items()}
        
        # Fill in 0 for any tools that weren't in the top 5 (unlikely with bias=100)
        for tool in tool_names_list:
            if tool not in normalized_probs:
                normalized_probs[tool] = 0.0
        
        return normalized_probs

    except Exception as e:
        logging.error(f"API call failed for Method 1: {e}")
        return None

# --- 4. Method 2: JSON Scoring (The "Practical" Way) ---

JSON_SCORING_SYSTEM_PROMPT = """
You are a helpful assistant. The user will ask a question.
Your job is to analyze the query and the available tools, and then output a JSON object
scoring your confidence (an integer from 0 to 100) for which tool to use next.

Available Tools:
{tools_json}

You MUST output a single, valid JSON object with the key "tool_scores",
which is a dictionary mapping *each* tool name to its confidence score.
Example: {{"tool_scores": {{"UnlockAccount": 90, "QueryDatabase": 30, "EscalateToHuman": 10}}}}
"""

def get_action_confidence_via_json(user_query, tools_list):
    """
    Prompts the model to self-report its confidence scores in JSON format.
    """
    logging.info("--- Starting Method 2: JSON Scoring ---")
    
    tools_json = json.dumps(tools_list, indent=2)
    prompt = JSON_SCORING_SYSTEM_PROMPT.format(tools_json=tools_json)
    
    try:
        response = client.chat.completions.create(
            model=deployment, # Use your deployment name
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"} # <<< KEY: Force JSON output
        )
        
        # 2. Parse the JSON response
        scores_data = json.loads(response.choices[0].message.content)
        scores = scores_data['tool_scores']
        
        # 3. Normalize the scores to get a probability distribution
        total_score = sum(scores.values())
        if total_score == 0:
            # Avoid division by zero; return a uniform distribution
            num_tools = len(scores)
            return {tool: 1.0 / num_tools for tool in scores}
            
        normalized_probs = {tool: score / total_score for tool, score in scores.items()}
        
        return normalized_probs

    except Exception as e:
        logging.error(f"API call failed for Method 2: {e}")
        return None

# --- 5. Helper Function for SAUP (from our previous PoC) ---

def calculate_step_uncertainty(prob_dist_dict):
    """
    Calculates the normalized entropy (our U_n) from the
    probability distribution dictionary.
    """
    if not prob_dist_dict:
        return 1.0 # Max uncertainty if something failed
        
    prob_array = np.array(list(prob_dist_dict.values()))
    
    # Handle the case of a single-item array (which has 0 entropy)
    if len(prob_array) <= 1:
        return 0.0 # 100% confidence in one choice
        
    # Use scipy.stats.entropy for a stable calculation
    # Normalized entropy = Entropy / max_entropy
    # max_entropy for N choices is log(N)
    return entropy(prob_array) / np.log(len(prob_array))


# --- 6. End-to-End Demonstration ---

if __name__ == "__main__":
    
    if subscription_key == "YOUR_AZURE_OPENAI_KEY_HERE":
        logging.error("="*50)
        logging.error("SCRIPT STOPPED: Please fill in your `subscription_key` at the top of the file.")
        logging.error("="*50)
        exit()

    user_query = "My account is locked, I can't log in to Workday. Please help."
    
    logging.info(f"Client is configured for endpoint: {endpoint}")
    logging.info(f"Processing query: '{user_query}'")
    logging.info("="*40)
    
    # --- Run Method 1 ---
    # We need the token IDs for this method
    tool_token_map = get_tool_token_ids(TOOL_NAMES)
    logit_probs = get_action_confidence_via_logits(user_query, TOOL_NAMES, tool_token_map)
    
    if logit_probs:
        logging.info(f"Method 1 (Logits) Probabilities: {json.dumps(logit_probs, indent=2)}")
        u_n_logit = calculate_step_uncertainty(logit_probs)
        logging.info(f"Method 1 (Logits) Step Uncertainty (U_n): {u_n_logit:.4f}")
    
    logging.info("="*40)

    # --- Run Method 2 ---
    json_probs = get_action_confidence_via_json(user_query, TOOLS)
    
    if json_probs:
        logging.info(f"Method 2 (JSON) Probabilities: {json.dumps(json_probs, indent=2)}")
        u_n_json = calculate_step_uncertainty(json_probs)
        logging.info(f"Method 2 (JSON) Step Uncertainty (U_n): {u_n_json:.4f}")
        
    logging.info("="*4S0)
    logging.info("Demonstration complete.")
