"""
Ablation Study Preparation
Generates weighted vs. unweighted evidence for Phase 4.
NO TRAINING. Just data preparation.
"""

import json
import os
from typing import List, Dict
from config import PHASE2_DIR, BASE_DIR, OUTPUT_DIR, WEIGHTED_EVIDENCE_PATH, UNWEIGHTED_EVIDENCE_PATH
from evidence_classifier import get_classifier
from hierarchy_weights import EvidenceWeighter


def load_phase2_evidence():
    """Load evidence retrieved from Phase 2."""
    possible_paths = [
        os.path.join(PHASE2_DIR, "retrieved_evidence.json"),
        os.path.join(BASE_DIR, "data", "phase2_evidence", "retrieved_evidence.json"),
        r"C:\Users\stme\Desktop\phase3\data\phase2_evidence\retrieved_evidence.json",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[OK] Loaded Phase 2 evidence: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                claims_evidence = {}
                for claim_id, claim_data in data.items():
                    if isinstance(claim_data, dict) and 'evidence' in claim_data:
                        evidence_list = claim_data['evidence']
                        if isinstance(evidence_list, list):
                            claims_evidence[claim_id] = evidence_list
                        else:
                            claims_evidence[claim_id] = []
                    elif isinstance(claim_data, list):
                        claims_evidence[claim_id] = claim_data
                    else:
                        claims_evidence[claim_id] = []
                
                print(f"[INFO] Extracted evidence for {len(claims_evidence)} claims")
                return claims_evidence
            
            return data
    
    print("[WARN] No Phase 2 evidence found. Using MOCK data.")
    return {
        "claim_001": [
            {"id": "doc1", "title": "Randomized Trial of Statins", "text": "Double-blind RCT...", "score": 0.92, "source": "scifact"},
            {"id": "doc2", "title": "Observational Study of Statin Use", "text": "Prospective cohort...", "score": 0.88, "source": "scifact"},
        ]
    }


def prepare_ablation_data(claims_evidence, use_llm=False, classifier=None):
    """
    Prepare weighted vs. unweighted evidence for each claim.
    
    Args:
        claims_evidence: dict mapping claim_id -> list of evidence dicts
        use_llm: bool - whether to use LLM (deprecated, use classifier param)
        classifier: optional pre-initialized classifier (for LLM mode)
    """
    print("=" * 70)
    print("PREPARING ABLATION DATA FOR PHASE 4")
    print("=" * 70)
    
    # Use provided classifier or create one
    if classifier is None:
        classifier = get_classifier(use_llm=use_llm)
    
    weighter = EvidenceWeighter()
    
    weighted_results = []
    unweighted_results = []
    
    total_evidence = 0
    
    for claim_id, evidence_list in claims_evidence.items():
        if not isinstance(evidence_list, list):
            print(f"[WARN] Skipping claim {claim_id}: not a list")
            continue
        
        if not evidence_list:
            print(f"[WARN] Claim {claim_id}: empty evidence list")
            continue
        
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
        
        weighted = weighter.apply_weights(classified)
        
        unweighted = []
        for ev in classified:
            ev_copy = ev.copy()
            ev_copy['tier_weight'] = 1.0
            ev_copy['predicted_tier'] = 'UNWEIGHTED'
            ev_copy['weighted_score'] = ev_copy.get('score', 0.5)
            unweighted.append(ev_copy)
        unweighted.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        weighted_results.append({
            'claim_id': claim_id,
            'evidence': weighted,
            'num_evidence': len(weighted),
            'avg_tier_weight': sum(e['tier_weight'] for e in weighted) / len(weighted) if weighted else 0
        })
        unweighted_results.append({
            'claim_id': claim_id,
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
    print(f"[INFO] Total claims: {len(claims_evidence)}")
    print(f"[INFO] Total evidence pieces classified: {total_evidence}")
    
    # Show tier distribution
    from collections import Counter
    all_tiers = []
    for claim in weighted_results:
        for ev in claim['evidence']:
            all_tiers.append(ev['predicted_tier'])
    
    if all_tiers:
        tier_counts = Counter(all_tiers)
        print(f"\nEvidence tier distribution:")
        for tier in ['A', 'B', 'C', 'D']:
            count = tier_counts.get(tier, 0)
            pct = (count / len(all_tiers) * 100) if all_tiers else 0
            print(f"  Tier {tier}: {count} ({pct:.1f}%)")
    
    if weighted_results:
        print("\nSample weighted evidence (first claim):")
        for ev in weighted_results[0]['evidence'][:3]:
            title = str(ev.get('title', 'No title'))[:50]
            print(f"  [{ev['predicted_tier']}] weight={ev['tier_weight']:.1f} score={ev.get('score', 0):.3f} | {title}...")
    
    return weighted_results, unweighted_results


# Backward compatibility alias
prepare_ablation_data_llm = prepare_ablation_data


if __name__ == "__main__":
    claims_ev = load_phase2_evidence()
    prepare_ablation_data(claims_ev, use_llm=False)