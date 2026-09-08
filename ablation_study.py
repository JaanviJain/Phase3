"""
Ablation Study Preparation
Generates weighted vs. unweighted evidence for Phase 4.
NO TRAINING. Just data preparation.
CRITICAL FIXES:
- Handles both Phase 2 output formats (flat dict and {"claims": [...]})
- Preserves claim text and true_label for Phase 4
- Truncates to top_k evidence per claim (default 5)
- No hardcoded Windows paths
"""

import json
import os
from typing import List, Dict
from collections import Counter

try:
    from config import (
        PHASE2_DIR, BASE_DIR, OUTPUT_DIR, 
        WEIGHTED_EVIDENCE_PATH, UNWEIGHTED_EVIDENCE_PATH
    )
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "phase_outputs")
    PHASE2_DIR = os.path.join(BASE_DIR, "data", "phase2_evidence")
    WEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_weighted.json")
    UNWEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_unweighted.json")

from evidence_classifier import get_classifier
from hierarchy_weights import EvidenceWeighter

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_phase2_evidence():
    """
    Load Phase 2 output. Handles both formats:
    1. {"metadata": {...}, "claims": [{claim_id, claim, true_label, evidence, ...}]}
    2. {claim_id: {claim, evidence, ...}} (legacy flat dict)
    """
    possible_paths = [
        os.path.join(PHASE2_DIR, "retrieved_evidence.json"),
        os.path.join(BASE_DIR, "data", "phase2_evidence", "retrieved_evidence.json"),
        os.path.join(BASE_DIR, "retrieved_evidence.json"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[OK] Loaded Phase 2 evidence: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Format 1: Standard corrected Phase 2 output
            if isinstance(data, dict) and "claims" in data:
                claims_data = {}
                for c in data["claims"]:
                    cid = c.get("claim_id") or c.get("id")
                    if cid:
                        claims_data[str(cid)] = c
                print(f"[INFO] Extracted {len(claims_data)} claims from standard format")
                return claims_data
            
            # Format 2: Legacy flat dict
            if isinstance(data, dict):
                claims_data = {}
                for claim_id, claim_data in data.items():
                    if isinstance(claim_data, dict):
                        claims_data[str(claim_id)] = claim_data
                print(f"[INFO] Extracted {len(claims_data)} claims from legacy format")
                return claims_data
            
            return {}
    
    raise FileNotFoundError(
        f"No Phase 2 evidence found. Searched: {possible_paths}\n"
        "Run Phase 2 first to generate retrieved_evidence.json"
    )


def prepare_ablation_data(claims_evidence: Dict, use_llm=False, classifier=None, top_k: int = 5):
    """
    Prepare weighted vs. unweighted evidence for each claim.
    
    Args:
        claims_evidence: dict mapping claim_id -> claim dict with 'claim', 'evidence', etc.
        use_llm: bool - whether to use Ollama LLM (slow) or rule-based (fast)
        classifier: optional pre-initialized classifier
        top_k: keep only top_k evidence pieces per claim (prevents Phase 4 overload)
    """
    print("=" * 70)
    print("PREPARING ABLATION DATA FOR PHASE 4")
    print("=" * 70)
    
    if classifier is None:
        classifier = get_classifier(use_llm=use_llm)
    
    weighter = EvidenceWeighter()
    
    weighted_results = []
    unweighted_results = []
    total_evidence = 0
    skipped_claims = 0
    
    for claim_id, claim_data in claims_evidence.items():
        if not isinstance(claim_data, dict):
            print(f"[WARN] Skipping claim {claim_id}: not a dict")
            skipped_claims += 1
            continue
        
        evidence_list = claim_data.get('evidence', [])
        if not isinstance(evidence_list, list):
            print(f"[WARN] Claim {claim_id}: evidence is not a list")
            skipped_claims += 1
            continue
        
        claim_text = claim_data.get('claim', '')
        true_label = claim_data.get('true_label')
        true_label_name = claim_data.get('true_label_name')
        
        if not evidence_list:
            print(f"[WARN] Claim {claim_id}: empty evidence list")
            # Still include claim with empty evidence so Phase 4 knows it exists
            weighted_results.append({
                'claim_id': claim_id,
                'claim': claim_text,
                'true_label': true_label,
                'true_label_name': true_label_name,
                'evidence': [],
                'num_evidence': 0,
                'avg_tier_weight': 0.0
            })
            unweighted_results.append({
                'claim_id': claim_id,
                'claim': claim_text,
                'true_label': true_label,
                'true_label_name': true_label_name,
                'evidence': [],
                'num_evidence': 0,
                'avg_tier_weight': 1.0
            })
            continue
        
        # Classify each evidence piece
        classified = []
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
            text = ev.get('text', ev.get('abstract', ''))
            title = ev.get('title', '')
            tier, weight, reason = classifier.classify(title, text)
            ev_copy = ev.copy()
            ev_copy['predicted_tier'] = tier
            ev_copy['tier_weight'] = weight
            ev_copy['classification_reason'] = reason
            classified.append(ev_copy)
            total_evidence += 1
        
        if not classified:
            continue
        
        # Weighted: apply hierarchy + truncate to top_k
        weighted = weighter.apply_weights(classified)[:top_k]
        
        # Unweighted: flat weights + truncate to top_k
        unweighted = []
        for ev in classified:
            ev_copy = ev.copy()
            ev_copy['tier_weight'] = 1.0
            ev_copy['predicted_tier'] = ev_copy.get('predicted_tier', 'U')
            ev_copy['weighted_score'] = ev_copy.get('score', 0.5)
            unweighted.append(ev_copy)
        unweighted.sort(key=lambda x: x.get('score', 0), reverse=True)
        unweighted = unweighted[:top_k]
        
        weighted_results.append({
            'claim_id': claim_id,
            'claim': claim_text,
            'true_label': true_label,
            'true_label_name': true_label_name,
            'evidence': weighted,
            'num_evidence': len(weighted),
            'avg_tier_weight': sum(e['tier_weight'] for e in weighted) / len(weighted) if weighted else 0
        })
        unweighted_results.append({
            'claim_id': claim_id,
            'claim': claim_text,
            'true_label': true_label,
            'true_label_name': true_label_name,
            'evidence': unweighted,
            'num_evidence': len(unweighted),
            'avg_tier_weight': 1.0
        })
    
    # Save
    with open(WEIGHTED_EVIDENCE_PATH, 'w', encoding='utf-8') as f:
        json.dump(weighted_results, f, indent=2, ensure_ascii=False)
    with open(UNWEIGHTED_EVIDENCE_PATH, 'w', encoding='utf-8') as f:
        json.dump(unweighted_results, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVE] Weighted evidence: {WEIGHTED_EVIDENCE_PATH}")
    print(f"[SAVE] Unweighted evidence: {UNWEIGHTED_EVIDENCE_PATH}")
    print(f"[INFO] Total claims processed: {len(claims_evidence)}")
    print(f"[INFO] Skipped/malformed claims: {skipped_claims}")
    print(f"[INFO] Total evidence pieces classified: {total_evidence}")
    
    # Tier distribution summary
    all_tiers = []
    for claim in weighted_results:
        for ev in claim.get('evidence', []):
            all_tiers.append(ev.get('predicted_tier', 'U'))
    
    if all_tiers:
        tier_counts = Counter(all_tiers)
        print(f"\nEvidence tier distribution (weighted):")
        for tier in ['A', 'B', 'C', 'D', 'U']:
            count = tier_counts.get(tier, 0)
            pct = (count / len(all_tiers) * 100) if all_tiers else 0
            print(f"  Tier {tier}: {count} ({pct:.1f}%)")
    
    if weighted_results and weighted_results[0].get('evidence'):
        print("\nSample top-weighted evidence (first claim):")
        for ev in weighted_results[0]['evidence'][:3]:
            title = str(ev.get('title', 'No title'))[:50]
            print(f"  [{ev.get('predicted_tier', '?')}] w={ev.get('tier_weight', 0):.1f} s={ev.get('score', 0):.3f} | {title}...")
    
    return weighted_results, unweighted_results


if __name__ == "__main__":
    claims_ev = load_phase2_evidence()
    prepare_ablation_data(claims_ev, use_llm=False, top_k=5)
