import os
import json
import numpy as np
import tiktoken
from openai import AzureOpenAI
from scipy.stats import entropy
import logging
import math
import time
import torch
from sentence_transformers import SentenceTransformer, util

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Your Client Setup (from your image) ---
endpoint = "https://laksh-frame.openai.azure.com/"
deployment = "gpt-4o"  # This is the deployment name
subscription_key = "YOUR_AZURE_OPENAI_KEY_HERE" # <<< ⚠️ FILL THIS IN
api_version = "2024-05-01-preview" 

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

# --- 3. Live U_n Generation (Method 2) ---

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
    Calls the Azure client to get a JSON-based confidence score.
    Returns: (prob_dist_dict, chosen_action)
    """
    logging.info("  [Agent] Calling LLM to get action confidence...")
    tools_json = json.dumps(tools_list, indent=2)
    prompt = JSON_SCORING_SYSTEM_PROMPT.format(tools_json=tools_json)
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )
        scores_data = json.loads(response.choices[0].message.content)
        scores = scores_data['tool_scores']
        
        total_score = sum(scores.values())
        if total_score == 0:
            num_tools = len(scores)
            normalized_probs = {tool: 1.0 / num_tools for tool in scores}
        else:
            normalized_probs = {tool: score / total_score for tool, score in scores.items()}
            
        chosen_action = max(normalized_probs, key=normalized_probs.get)
        return normalized_probs, chosen_action

    except Exception as e:
        logging.error(f"API call failed for Method 2: {e}")
        return None, None

def calculate_step_uncertainty(prob_dist_dict):
    """
    (PAPER-ACCURATE) Calculates normalized entropy (U_n)
    from the real probability distribution.
    """
    if not prob_dist_dict:
        return 1.0 # Max uncertainty
    prob_array = np.array(list(prob_dist_dict.values()))
    if len(prob_array) <= 1:
        return 0.0 # 100% confidence
    return entropy(prob_array) / np.log(len(prob_array))


# --- 4. SAUP ML Components (GPU-Enabled Distance) ---

class SemanticDistanceModel:
    """
    (GPU-ENABLED) This class is the real ML component.
    It loads a transformer model to calculate semantic distance.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)
        logging.info(f"[SAUP-Config] SemanticDistanceModel loaded on {self.device}.")

    def get_distance(self, text_a, text_b):
        """
        Calculates semantic distance (1 - cosine_similarity).
        """
        embeddings = self.model.encode(
            [text_a, text_b], 
            convert_to_tensor=True, 
            device=self.device
        )
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return 1 - cos_sim.item()

class MockHMMWeights:
    """
    (MOCK) Mocks a trained, stateful HMM.
    """
    def __init__(self):
        self.state_weights = {'ON_TRACK': 0.1, 'DRIFTING': 0.5, 'LOST': 0.9}
        self.thresholds = {'TO_DRIFTING': 0.4, 'TO_LOST': 0.7, 'TO_RECOVER': 0.2}
        
    def get_weights(self, D_list):
        W_list = []
        current_state = 'ON_TRACK'
        for d in D_list:
            if d > self.thresholds['TO_LOST']:
                current_state = 'LOST'
            elif d > self.thresholds['TO_DRIFTING']:
                current_state = 'DRIFTING'
            elif d < self.thresholds['TO_RECOVER']:
                current_state = 'ON_TRACK'
            W_list.append(self.state_weights[current_state])
        return W_list

def saup_aggregation(U_list, W_list):
    """(PAPER-ACCURATE) Aggregates using Weighted RMS."""
    N = len(U_list)
    if N == 0: return 0
    weighted_sq = [(w * u) ** 2 for u, w in zip(U_list, W_list)]
    return np.sqrt(np.sum(weighted_sq) / N)

# --- 5. The E2E SAUP Agent Class ---

class SaupAgent:
    """
    An agent that uses SAUP for oversight on every step.
    """
    def __init__(self, client, deployment_name, tools_list, query, distance_model):
        self.client = client
        self.deployment = deployment_name
        self.tools = tools_list
        self.Q = query  # The initial user query
        self.distance_model = distance_model # The GPU-powered distance model
        
        self.Z_list = [] # List of (A, T, O) tuples
        self.U_list = [] # List of U_n floats
        self.D_a_list = [] # List of D_a floats
        self.D_o_list = [] # List of D_o floats

    def mock_get_thought(self, query, chosen_action):
        """(MOCK) Simulates the agent's 'Thought' process."""
        return f"The user query is '{query[:30]}...'. Based on the tools, the best action seems to be '{chosen_action}'."

    def mock_get_observation(self, action):
        """(MOCK) Simulates the *result* of running the tool."""
        if action == "UnlockAccount":
            return "Observation: User 'jdoe' account has been successfully unlocked."
        elif action == "QueryDatabase":
            # This observation is slightly different from the action, creating a small D_o gap
            return "Observation: Found 3 similar incidents. All resolved by 'UnlockAccount'."
        elif action == "EscalateToHuman":
            return "Observation: Ticket #INC56789 created and assigned to L2 Support."
        else:
            return "Observation: Unknown action."

    def run_step(self, current_query):
        """
        Runs one full step of the ReAct + SAUP process.
        """
        # 1. Get Action Confidence (Calls LLM)
        prob_dist, A_n = get_action_confidence_via_json(current_query, self.tools)
        if not prob_dist:
            logging.error("  [Agent] Failed to get action confidence. Aborting.")
            return None

        # 2. Calculate U_n (Step Uncertainty)
        U_n = calculate_step_uncertainty(prob_dist)
        self.U_list.append(U_n)
        logging.info(f"  [SAUP] Step Uncertainty (U_n): {U_n:.4f}")

        # 3. Get T_n (Thought) and O_n (Observation)
        T_n = self.mock_get_thought(current_query, A_n)
        O_n = self.mock_get_observation(A_n)
        
        self.Z_list.append((A_n, T_n, O_n))

        # 4. Calculate Distances (D_a and D_o) using the REAL model
        D_a_n = self.distance_model.get_distance(A_n + T_n, self.Q)
        D_o_n = self.distance_model.get_distance(A_n, O_n)
        self.D_a_list.append(D_a_n)
        self.D_o_list.append(D_o_n)
        logging.info(f"  [SAUP] Distances: Drift (D_a)={D_a_n:.3f}, Gap (D_o)={D_o_n:.3f}")
        
        return A_n, O_n 

    def calculate_final_saup_score(self):
        """
        Calculates the final U_agent score after the plan is complete.
        """
        logging.info("--- Calculating Final SAUP Score ---")
        
        D_combined = [(da + do) for da, do in zip(self.D_a_list, self.D_o_list)]
        hmm_model = MockHMMWeights()
        W_list = hmm_model.get_weights(D_combined)
        
        U_agent = saup_aggregation(self.U_list, W_list)
        
        print("\n" + "="*60)
        print("Trajectory Analysis Report (GPU-Powered)")
        print("="*60)
        print(f"Initial Query (Q): {self.Q}\n")
        print(f"{'Step':<5} | {'Action (A_n)':<16} | {'U_n':<6} | {'D_a':<6} | {'D_o':<6} | {'W_n':<6}")
        print("-"*54)
        for i in range(len(self.Z_list)):
            A_n, T_n, O_n = self.Z_list[i]
            print(f"{i+1:<5} | {A_n:<16} | {self.U_list[i]:<6.3f} | {self.D_a_list[i]:<6.3f} | {self.D_o_list[i]:<6.3f} | {W_list[i]:<6.3f}")
        print("="*60)

        return U_agent

# --- 6. End-to-End Demonstration ---

if __name__ == "__main__":
    
    if subscription_key == "YOUR_AZURE_OPENAI_KEY_HERE":
        logging.error("="*50)
        logging.error("SCRIPT STOPPED: Please fill in your `subscription_key` at the top of the file.")
        logging.error("="*50)
        exit()

    # --- Load ML Models (once) ---
    try:
        distance_model = SemanticDistanceModel()
    except Exception as e:
        logging.error(f"FATAL: Could not load SentenceTransformer model: {e}")
        logging.error("Please run: pip install torch sentence-transformers")
        exit()
        
    user_query = "My account is locked, I can't log in to Workday. Please help."
    
    logging.info(f"Client configured for: {endpoint}")
    logging.info(f"--- Starting New Incident ---")
    logging.info(f"Query (Q): {user_query}")
    
    OVERSIGHT_THRESHOLD = 0.5
    
    # 1. Initialize the agent (and pass it the distance model)
    agent = SaupAgent(client, deployment, TOOLS, user_query, distance_model)
    
    # 2. Run the agent's plan (simulating a 2-step plan)
    current_context = user_query
    for i in range(2): # Simulate a 2-step plan
        logging.info(f"\n--- Running Agent Step {i+1} ---")
        A_n_O_n = agent.run_step(current_context)
        if not A_n_O_n:
            logging.error("Agent step failed. Stopping plan.")
            break
        
        A_n, O_n = A_n_O_n
        
        if A_n == "EscalateToHuman":
            logging.info("  [Agent] Action is 'EscalateToHuman'. Stopping plan.")
            break
        
        current_context = O_n 

    # 3. Calculate the final score for the *entire plan*
    final_score = agent.calculate_final_saup_score()
    
    logging.info(f"Final Agent Uncertainty (U_agent): {final_score:.4f}")
    
    # 4. Make the final triage decision
    if final_score > OVERSIGHT_THRESHOLD:
        print(f"DECISION: 🚨 ESCALATE TO HUMAN (Score: {final_score:.4f})")
    else:
        print(f"DECISION: ✅ AUTHORIZE AGENT (Score: {final_score:.4f})")
