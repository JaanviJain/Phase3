"""
Evidence Tier Classifier
Classifies medical abstracts into Tier A/B/C/D.
NO TRAINING. Rule-based + optional Ollama LLM fallback.
"""

import re
import os
import json
import requests
from typing import List, Dict, Tuple

# Safe config import — works even if config.py is missing or different
try:
    from config import TIERS, OLLAMA_URL, OLLAMA_MODEL, OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "phase_outputs")
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen2.5:7b"
    TIERS = {
        "A": {"weight": 1.0, "keywords": [
            "randomized controlled trial", "rct", "systematic review", "meta-analysis",
            "cochrane", "preregistered", "clinical trial", "double-blind", "placebo-controlled",
            "parallel-group", "allocation concealment"
        ]},
        "B": {"weight": 0.7, "keywords": [
            "cohort", "observational", "prospective", "retrospective", "longitudinal",
            "cross-sectional", "population-based", "epidemiological", "registry", "surveillance"
        ]},
        "C": {"weight": 0.4, "keywords": [
            "case report", "case series", "expert opinion", "narrative review",
            "qualitative study", "pilot study", "feasibility study", "in-vitro", "cell line"
        ]},
        "D": {"weight": 0.2, "keywords": [
            "editorial", "commentary", "letter", "preprint", "medrxiv", "biorxiv",
            "opinion", "news", "press release", "conference abstract", "meeting abstract"
        ]}
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)


class RuleBasedClassifier:
    """
    Classifies evidence based on keywords in title and FULL abstract.
    NO GPU needed. Uses scoring + smart fallback.
    """
    
    def __init__(self):
        self.tier_patterns = {}
        for tier, info in TIERS.items():
            keywords = info.get('keywords', [])
            if keywords:
                pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
                self.tier_patterns[tier] = re.compile(pattern, re.IGNORECASE)
            else:
                self.tier_patterns[tier] = re.compile(r'(?!)')  # Never match
    
    def classify(self, title: str, abstract: str = "") -> Tuple[str, float, str]:
        text = f"{title} {abstract}".lower()
        # Remove section headers to avoid false keyword matches
        text = re.sub(r'\b(background|objective|aim|methods?|results?|conclusion|discussion|introduction|materials)\s*[:;]', ' ', text)
        
        scores = {}
        matched_keywords = {}
        for tier in ['A', 'B', 'C', 'D']:
            matches = self.tier_patterns[tier].findall(text)
            scores[tier] = len(matches)
            matched_keywords[tier] = list(set(matches))
        
        best_tier = max(scores, key=scores.get)
        best_score = scores[best_tier]
        
        if best_score > 0:
            weight = TIERS.get(best_tier, {}).get('weight', 0.2)
            return best_tier, weight, f"Matched: {', '.join(matched_keywords[best_tier])}"
        
        # SMART FALLBACK
        text_lower = text
        case_signals = [
            'we present', 'we describe', 'here we report', 'patient was',
            'year-old', 'years old', 'admitted to', 'presented with',
            'diagnosed with', 'clinical course', 'case of', 'cases of',
            'rare case', 'unusual case'
        ]
        case_score = sum(1 for s in case_signals if s in text_lower)
        
        study_signals = [
            'patients were', 'participants were', 'subjects were', 'enrolled',
            'eligible', 'inclusion criteria', 'exclusion criteria', 'informed consent',
            'ethics committee', 'institutional review', 'irb approved',
            'randomly', 'allocation', 'intervention', 'treatment group',
            'control group', 'placebo', 'follow-up', 'followed up',
            'primary outcome', 'secondary outcome', 'endpoint', 'p-value',
            'mean age', 'median age', 'baseline characteristics', 'demographics'
        ]
        study_score = sum(1 for s in study_signals if s in text_lower)
        
        editorial_signals = [
            'we believe', 'we argue', 'in our opinion', 'should be',
            'policy', 'policies', 'healthcare system', 'recommend',
            'urgent need', 'call for', 'perspective', 'viewpoint',
            'editorial', 'commentary', 'letter to', 'correspondence'
        ]
        editorial_score = sum(1 for s in editorial_signals if s in text_lower)
        
        if case_score >= 2:
            return 'C', TIERS['C']['weight'], f"Case presentation signals ({case_score} hits)"
        
        if editorial_score >= 2:
            return 'D', TIERS['D']['weight'], f"Editorial/opinion signals ({editorial_score} hits)"
        
        if study_score >= 4:
            return 'B', TIERS['B']['weight'], f"Structured study format detected ({study_score} hits)"
        
        return 'D', TIERS['D']['weight'], "Insufficient study type indicators"
    
    def classify_batch(self, evidence_list: List[Dict]) -> List[Dict]:
        results = []
        for ev in evidence_list:
            tier, weight, reason = self.classify(
                ev.get('title', ''), 
                ev.get('text', ev.get('abstract', ''))
            )
            ev_copy = ev.copy()
            ev_copy['predicted_tier'] = tier
            ev_copy['tier_weight'] = weight
            ev_copy['classification_reason'] = reason
            results.append(ev_copy)
        return results


class OllamaClassifier:
    """
    Optional Ollama-based classifier.
    Uses your local Ollama API (qwen2.5:7b). No model loading here.
    Falls back to rule-based if Ollama is down.
    """
    
    def __init__(self, model_name: str = None, url: str = None):
        self.model_name = model_name or OLLAMA_MODEL
        self.url = url or OLLAMA_URL
        self.rule_fallback = RuleBasedClassifier()
        self._check_connection()
    
    def _check_connection(self):
        try:
            r = requests.get(self.url.replace("/generate", "/tags"), timeout=5)
            if r.status_code == 200:
                print(f"Ollama classifier ready ({self.model_name}).")
            else:
                print("Ollama unreachable. Will fallback to rule-based.")
        except Exception:
            print("Ollama not running. Will fallback to rule-based.")
    
    def _build_prompt(self, title: str, abstract: str) -> str:
        text = f"Title: {title}\nAbstract: {abstract[:800]}"
        return f"""You are a medical evidence quality assessor.
Classify the following medical study into exactly one tier:
- A: RCT, Systematic Review, or Meta-analysis
- B: Cohort, Observational, or Cross-sectional study
- C: Case Report, Case Series, or Expert Opinion
- D: Editorial, Commentary, News, or Preprint

{text}

Respond with ONLY the letter A, B, C, or D. No explanation."""

    def classify(self, title: str, abstract: str = "") -> Tuple[str, float, str]:
        prompt = self._build_prompt(title, abstract)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 10}
        }
        try:
            r = requests.post(self.url, json=payload, timeout=60)
            r.raise_for_status()
            output = r.json().get("response", "").strip()
            match = re.search(r'\b([A-D])\b', output.upper())
            if match:
                tier = match.group(1)
                weight = TIERS.get(tier, {}).get('weight', 0.2)
                return tier, weight, f"Ollama predicted: {tier}"
            else:
                raise ValueError(f"No tier in output: {output}")
        except Exception as e:
            tier, weight, reason = self.rule_fallback.classify(title, abstract)
            return tier, weight, f"Ollama failed ({e}), fallback: {reason}"
    
    def classify_batch(self, evidence_list: List[Dict]) -> List[Dict]:
        from tqdm import tqdm
        results = []
        for ev in tqdm(evidence_list, desc="Ollama Classifying"):
            tier, weight, reason = self.classify(
                ev.get('title', ''),
                ev.get('text', ev.get('abstract', ''))
            )
            ev_copy = ev.copy()
            ev_copy['predicted_tier'] = tier
            ev_copy['tier_weight'] = weight
            ev_copy['classification_reason'] = reason
            results.append(ev_copy)
        return results


def get_classifier(use_llm: bool = False):
    if use_llm:
        return OllamaClassifier()
    return RuleBasedClassifier()


if __name__ == "__main__":
    classifier = get_classifier(use_llm=False)
    test_cases = [
        {"title": "A Randomized Controlled Trial of Aspirin in Myocardial Infarction", "text": "We conducted a double-blind RCT..."},
        {"title": "Cohort Study of Smoking and Lung Cancer", "text": "A prospective longitudinal cohort..."},
        {"title": "Case Report: Rare Side Effect", "text": "We present a 45-year-old patient..."},
        {"title": "Editorial: Thoughts on Healthcare Policy", "text": "In this opinion piece..."}
    ]
    print("=" * 70)
    print("TESTING EVIDENCE CLASSIFIER")
    print("=" * 70)
    for case in test_cases:
        tier, weight, reason = classifier.classify(case['title'], case['text'])
        print(f"\nTitle: {case['title'][:60]}...")
        print(f"  Tier: {tier} (weight: {weight}) | {reason}")
