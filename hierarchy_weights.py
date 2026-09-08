"""
Hierarchy Weighting
Applies tier weights to retrieved evidence and re-ranks.
NO TRAINING. Pure computation.
"""

import json
import os
from typing import List, Dict

try:
    from config import TIERS, WEIGHTED_EVIDENCE_PATH, UNWEIGHTED_EVIDENCE_PATH, OUTPUT_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "phase_outputs")
    WEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_weighted.json")
    UNWEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_unweighted.json")
    TIERS = {
        "A": {"weight": 1.0}, "B": {"weight": 0.7},
        "C": {"weight": 0.4}, "D": {"weight": 0.2}
    }

os.makedirs(OUTPUT_DIR, exist_ok=True)


class EvidenceWeighter:
    def __init__(self):
        self.tiers = TIERS
    
    def apply_weights(self, evidence_list: List[Dict]) -> List[Dict]:
        weighted = []
        for ev in evidence_list:
            tier = ev.get('predicted_tier', 'D')
            base_weight = self.tiers.get(tier, self.tiers.get('D', {'weight': 0.2}))['weight']
            retrieval_score = ev.get('score', 0.5)
            weighted_score = retrieval_score * base_weight
            
            ev_copy = ev.copy()
            ev_copy['weighted_score'] = round(weighted_score, 4)
            weighted.append(ev_copy)
        
        weighted.sort(key=lambda x: x['weighted_score'], reverse=True)
        return weighted
    
    def aggregate_evidence_embedding(self, evidence_list: List[Dict]) -> float:
        if not evidence_list:
            return 0.0
        total_weight = sum(ev.get('tier_weight', 0.2) for ev in evidence_list)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(
            ev.get('score', 0.5) * ev.get('tier_weight', 0.2) 
            for ev in evidence_list
        )
        return round(weighted_sum / total_weight, 4)


def save_evidence_variants(claim_id: str, claim_text: str, true_label, evidence_list: List[Dict], output_dir: str = None):
    """
    Save both weighted and unweighted versions for Phase 4 ablation.
    PRESERVES claim text and true_label from Phase 2.
    """
    if output_dir is None:
        output_dir = os.path.dirname(WEIGHTED_EVIDENCE_PATH) or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    weighter = EvidenceWeighter()
    weighted = weighter.apply_weights(evidence_list)
    
    unweighted = []
    for ev in evidence_list:
        ev_copy = ev.copy()
        ev_copy['tier_weight'] = 1.0
        ev_copy['predicted_tier'] = ev_copy.get('predicted_tier', 'U')
        ev_copy['weighted_score'] = ev_copy.get('score', 0.5)
        unweighted.append(ev_copy)
    unweighted.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return {
        'claim_id': claim_id,
        'claim': claim_text,
        'true_label': true_label,
        'weighted_evidence': weighted,
        'unweighted_evidence': unweighted,
        'aggregate_weighted_score': weighter.aggregate_evidence_embedding(weighted),
        'aggregate_unweighted_score': weighter.aggregate_evidence_embedding(unweighted)
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING HIERARCHY WEIGHTING")
    print("=" * 70)
    sample = [
        {"id": "doc1", "title": "RCT of Drug X", "predicted_tier": "A", "score": 0.85},
        {"id": "doc2", "title": "Cohort Study", "predicted_tier": "B", "score": 0.90},
        {"id": "doc3", "title": "Case Report", "predicted_tier": "C", "score": 0.80},
        {"id": "doc4", "title": "Editorial", "predicted_tier": "D", "score": 0.95}
    ]
    w = EvidenceWeighter()
    out = w.apply_weights(sample)
    for ev in out:
        print(f"{ev['id']}: Tier {ev['predicted_tier']} | Raw {ev['score']:.3f} | Weighted {ev['weighted_score']:.3f}")
