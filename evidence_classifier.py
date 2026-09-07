"""
Evidence Tier Classifier
Classifies medical abstracts into Tier A/B/C/D.
NO TRAINING. Rule-based + optional LLM prompting.
"""

import re
import os
import json
from typing import List, Dict, Tuple
from config import TIERS, LLM_MODEL, OUTPUT_DIR

class RuleBasedClassifier:
    """
    Classifies evidence based on keywords in title and FULL abstract.
    NO GPU needed. Uses scoring + smart fallback.
    """
    
    def __init__(self):
        self.tier_patterns = {}
        for tier, info in TIERS.items():
            pattern = r'\b(' + '|'.join(re.escape(k) for k in info['keywords']) + r')\b'
            self.tier_patterns[tier] = re.compile(pattern, re.IGNORECASE)
    
    def classify(self, title: str, abstract: str = "") -> Tuple[str, float, str]:
        """
        Classify a single abstract using FULL TEXT search + scoring.
        """
        # Search FULL text, not truncated
        text = f"{title} {abstract}".lower()
        text = re.sub(r'\b(background|objective|aim|methods?|results?|conclusion|discussion|introduction|materials)\s*[:;]', ' ', text)
        
        # Score each tier by counting keyword matches
        scores = {}
        matched_keywords = {}
        for tier in ['A', 'B', 'C', 'D']:
            matches = self.tier_patterns[tier].findall(text)
            scores[tier] = len(matches)
            matched_keywords[tier] = list(set(matches))
        
        # Find tier with highest score
        best_tier = max(scores, key=scores.get)
        best_score = scores[best_tier]
        
        # If we found matches, return the best tier
        if best_score > 0:
            return best_tier, TIERS[best_tier]['weight'], f"Matched: {', '.join(matched_keywords[best_tier])}"
        
        # ============================================
        # SMART FALLBACK — only if NO keywords matched
        # ============================================
        text_lower = text
        
        # Strong case report signals (high confidence)
        case_signals = [
            'we present', 'we describe', 'here we report', 'patient was',
            'year-old', 'years old', 'admitted to', 'presented with',
            'diagnosed with', 'clinical course', 'hospital course',
            'case of', 'cases of', 'rare case', 'unusual case'
        ]
        case_score = sum(1 for s in case_signals if s in text_lower)
        
        # Strong study signals — if it looks like a real study, default to B not D
        study_signals = [
            'patients were', 'participants were', 'subjects were', 'enrolled',
            'eligible', 'inclusion criteria', 'exclusion criteria', 'informed consent',
            'ethics committee', 'institutional review', 'irb approved',
            'randomly', 'allocation', 'intervention', 'treatment group',
            'control group', 'placebo', 'follow-up', 'followed up',
            'primary outcome', 'secondary outcome', 'endpoint', 'endpoints',
            'statistical significance', 'p-value', 'p value', 'ci 95%',
            'mean age', 'median age', 'baseline characteristics', 'demographics'
        ]
        study_score = sum(1 for s in study_signals if s in text_lower)
        
        # Strong editorial/commentary signals
        editorial_signals = [
            'we believe', 'we argue', 'in our opinion', 'should be',
            'policy', 'policies', 'healthcare system', 'recommend',
            'urgent need', 'call for', 'perspective', 'viewpoint',
            'editorial', 'commentary', 'letter to', 'correspondence'
        ]
        editorial_score = sum(1 for s in editorial_signals if s in text_lower)
        
        # Decision logic — be conservative about assigning D
        if case_score >= 2:
            return 'C', TIERS['C']['weight'], f"Case presentation signals ({case_score} hits)"
        
        if editorial_score >= 2:
            return 'D', TIERS['D']['weight'], f"Editorial/opinion signals ({editorial_score} hits)"
        
        # KEY CHANGE: If it looks like a structured study, default to B not D
        if study_score >= 4:
            return 'B', TIERS['B']['weight'], f"Structured study format detected ({study_score} hits)"
        
        # Only default to D if it truly looks like nothing
        return 'D', TIERS['D']['weight'], "Insufficient study type indicators"
    
    def classify_batch(self, evidence_list: List[Dict]) -> List[Dict]:
        """
        Classify a batch of evidence dicts.
        Each dict must have 'title' and optionally 'text'.
        """
        results = []
        for ev in evidence_list:
            tier, weight, reason = self.classify(
                ev.get('title', ''), 
                ev.get('text', '')
            )
            ev_copy = ev.copy()
            ev_copy['predicted_tier'] = tier
            ev_copy['tier_weight'] = weight
            ev_copy['classification_reason'] = reason
            results.append(ev_copy)
        return results


class LLMClassifier:
    """
    Optional LLM-based classifier.
    Uses local LLM for inference (GPU required).
    Falls back to rule-based if LLM fails.
    """
    
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.pipeline = None
        self.rule_fallback = RuleBasedClassifier()
        self._init_model()
    
    def _init_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            print(f"Loading LLM for evidence classification: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            # Load with quantization to save VRAM
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            except ImportError:
                print("bitsandbytes not found, loading in fp16")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                return_full_text=False
            )
            print("LLM classifier loaded.")
            
        except Exception as e:
            print(f"WARNING: Could not load LLM: {e}")
            print("Falling back to rule-based classification.")
            self.pipeline = None
    
    def _build_prompt(self, title: str, abstract: str) -> str:
        text = f"Title: {title}\nAbstract: {abstract[:500]}"
        prompt = f"""You are a medical evidence quality assessor.
Classify the following medical study into exactly one tier:
- A: RCT, Systematic Review, or Meta-analysis
- B: Cohort, Observational, or Cross-sectional study
- C: Case Report, Case Series, or Expert Opinion
- D: Editorial, Commentary, News, or Preprint

{text}

Tier (respond with only A, B, C, or D):"""
        return prompt
    
    def classify(self, title: str, abstract: str = "") -> Tuple[str, float, str]:
        if self.pipeline is None:
            return self.rule_fallback.classify(title, abstract)
        
        try:
            prompt = self._build_prompt(title, abstract)
            output = self.pipeline(prompt)[0]['generated_text'].strip()
            
            # Extract tier from output
            match = re.search(r'\b([A-D])\b', output.upper())
            if match:
                tier = match.group(1)
                return tier, TIERS[tier]['weight'], f"LLM predicted: {tier}"
            else:
                raise ValueError(f"LLM output did not contain tier: {output}")
                
        except Exception as e:
            # Fallback to rule-based
            tier, weight, reason = self.rule_fallback.classify(title, abstract)
            return tier, weight, f"LLM failed ({e}), fallback: {reason}"
    
    def classify_batch(self, evidence_list: List[Dict]) -> List[Dict]:
        from tqdm import tqdm
        results = []
        for ev in tqdm(evidence_list, desc="LLM Classifying"):
            tier, weight, reason = self.classify(
                ev.get('title', ''),
                ev.get('text', '')
            )
            ev_copy = ev.copy()
            ev_copy['predicted_tier'] = tier
            ev_copy['tier_weight'] = weight
            ev_copy['classification_reason'] = reason
            results.append(ev_copy)
        return results


def get_classifier(use_llm: bool = False):
    """
    Factory function.
    use_llm=False (default): Fast rule-based, CPU only.
    use_llm=True: LLM-based, requires GPU.
    """
    if use_llm:
        return LLMClassifier()
    return RuleBasedClassifier()


if __name__ == "__main__":
    # Test
    classifier = get_classifier(use_llm=False)
    
    test_cases = [
        {"title": "A Randomized Controlled Trial of Aspirin in Myocardial Infarction", "text": "We conducted a double-blind RCT..."},
        {"title": "Cohort Study of Smoking and Lung Cancer", "text": "A prospective longitudinal cohort..."},
        {"title": "Case Report: Rare Side Effect of Vaccine", "text": "We present a 45-year-old patient..."},
        {"title": "Editorial: Thoughts on Healthcare Policy", "text": "In this opinion piece..."}
    ]
    
    print("=" * 70)
    print("TESTING EVIDENCE CLASSIFIER")
    print("=" * 70)
    
    for case in test_cases:
        tier, weight, reason = classifier.classify(case['title'], case['text'])
        print(f"\nTitle: {case['title'][:60]}...")
        print(f"  Tier: {tier} (weight: {weight})")
        print(f"  Reason: {reason}")