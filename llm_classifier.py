"""
LLM-based Evidence Tier Classifier using Ollama
FINAL ATTEMPT: Ultra-strict system prompt + forced JSON output
"""

import requests
import json
import re
from typing import Tuple
from config import TIERS


class OllamaClassifier:
    """
    Uses local Ollama LLM for evidence classification.
    FINAL VERSION: System prompt + JSON mode + validation loop
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.rule_fallback = None
        self._check_connection()
    
    def _check_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m['name'] for m in r.json().get('models', [])]
                if self.model_name in models:
                    print(f"[OK] Ollama connected. Model: {self.model_name}")
                else:
                    print(f"[WARN] Model '{self.model_name}' not found.")
                    print(f"[WARN] Available: {models}")
            else:
                print(f"[ERROR] Ollama returned status {r.status_code}")
        except Exception as e:
            print(f"[ERROR] Cannot connect to Ollama: {e}")
    
    def _build_system_prompt(self) -> str:
        """
        Ultra-strict system prompt that forces the model into classification mode.
        """
        return """You are a medical evidence classification machine. You have ONE job:

Read a medical study title and abstract, then output EXACTLY ONE character: A, B, C, or D.

TIER DEFINITIONS:
- A = Randomized Controlled Trial, Systematic Review, Meta-analysis, Clinical Trial
- B = Cohort Study, Observational Study, Cross-sectional, Longitudinal, Registry
- C = Case Report, Case Series, Pilot Study, Animal Study, In Vitro
- D = Editorial, Commentary, Letter, Opinion, News, Blog, Preprint

RULES:
1. Output MUST be exactly one character: A, B, C, or D
2. NO explanations
3. NO punctuation
4. NO extra words
5. If unsure, output B

EXAMPLES:
Input: "A Randomized Trial of Aspirin in Heart Disease. Methods: We randomly assigned 1000 patients..."
Output: A

Input: "Cohort Study of Smoking and Cancer. We followed 5000 participants for 10 years..."
Output: B

Input: "Case Report: Rare Side Effect. A 45-year-old man presented with..."
Output: C

Input: "Editorial: Healthcare Policy Thoughts. In this opinion piece..."
Output: D

Now classify the following study. REMEMBER: Output ONLY A, B, C, or D. Nothing else."""

    def _build_prompt(self, title: str, abstract: str) -> str:
        """Build the user prompt with strict formatting."""
        text = f"Title: {title}\nAbstract: {abstract[:500]}"
        
        prompt = f"""{text}

CLASSIFICATION (respond with ONLY A, B, C, or D):"""
        return prompt
    
    def classify(self, title: str, abstract: str = "") -> Tuple[str, float, str]:
        """
        Classify using Ollama LLM with validation loop.
        Retries up to 3 times if output is invalid.
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_prompt(title, abstract)
                
                # Use chat endpoint with system prompt
                r = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 2,
                            "stop": ["\n", ".", ",", " ", "Explanation", "Because", "The", "This", "It"]
                        }
                    },
                    timeout=30
                )
                
                if r.status_code != 200:
                    raise ConnectionError(f"Status {r.status_code}")
                
                response_data = r.json()
                response = response_data.get('message', {}).get('content', '').strip().upper()
                
                # ULTRA-STRICT validation
                # Must be exactly 1 character and must be A/B/C/D
                if response and len(response) == 1 and response in ['A', 'B', 'C', 'D']:
                    tier = response
                    return tier, TIERS[tier]['weight'], f"LLM: {tier} (attempt {attempt+1})"
                
                # If invalid, retry with stronger prompt
                print(f"  [WARN] Invalid response (attempt {attempt+1}): '{response}' — retrying...")
                
            except Exception as e:
                print(f"  [WARN] LLM error (attempt {attempt+1}): {str(e)[:50]}")
        
        # ALL retries failed — fallback to rule-based
        print(f"  [WARN] All LLM retries failed. Using rule-based fallback.")
        if self.rule_fallback is None:
            from evidence_classifier import RuleBasedClassifier
            self.rule_fallback = RuleBasedClassifier()
        
        tier, weight, reason = self.rule_fallback.classify(title, abstract)
        return tier, weight, f"LLM failed after {max_retries} retries, fallback: {reason}"
    
    def classify_batch(self, evidence_list: list) -> list:
        """Classify a batch one by one."""
        results = []
        for ev in evidence_list:
            tier, weight, reason = self.classify(
                str(ev.get('title', '')),
                str(ev.get('text', ''))
            )
            ev_copy = ev.copy()
            ev_copy['predicted_tier'] = tier
            ev_copy['tier_weight'] = weight
            ev_copy['classification_reason'] = reason
            results.append(ev_copy)
        return results


def get_llm_classifier(model_name: str = "llama3.1:8b"):
    """Factory function."""
    return OllamaClassifier(model_name=model_name)