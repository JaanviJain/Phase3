"""
Hierarchy Weighting
Applies tier weights to retrieved evidence and re-ranks.
NO TRAINING. Pure computation.
"""

import json
from typing import List, Dict
from config import TIERS, WEIGHTED_EVIDENCE_PATH, UNWEIGHTED_EVIDENCE_PATH

class EvidenceWeighter:
    """
    Takes classified evidence and applies hierarchy weights.
    """
    
    def __init__(self):
        self.tiers = TIERS
    
    def apply_weights(self, evidence_list: List[Dict]) -> List[Dict]:
        """
        Apply tier weights to evidence.
        Input: list of dicts with 'predicted_tier' and 'score' (from retrieval)
        Output: sorted list with 'weighted_score'
        """
        weighted = []
        for ev in evidence_list:
            tier = ev.get('predicted_tier', 'D')
            base_weight = self.tiers.get(tier, self.tiers['D'])['weight']
            
            # If evidence has a retrieval score, combine it with tier weight
            retrieval_score = ev.get('score', 0.5)
            
            # Weighted score: retrieval relevance * evidence quality
            weighted_score = retrieval_score * base_weight
            
            ev_copy = ev.copy()
            ev_copy['weighted_score'] = round(weighted_score, 4)
            weighted.append(ev_copy)
        
        # Sort by weighted score descending
        weighted.sort(key=lambda x: x['weighted_score'], reverse=True)
        return weighted
    
    def aggregate_evidence_embedding(self, evidence_list: List[Dict]) -> float:
        """
        Compute aggregate trust score from weighted evidence.
        Simple heuristic: average of weighted scores.
        """
        if not evidence_list:
            return 0.0
        
        total_weight = sum(ev.get('tier_weight', 0.2) for ev in evidence_list)
        if total_weight == 0:
            return 0.0
        
        # Weighted average of retrieval scores
        weighted_sum = sum(
            ev.get('score', 0.5) * ev.get('tier_weight', 0.2) 
            for ev in evidence_list
        )
        return round(weighted_sum / total_weight, 4)


def save_evidence_variants(claim_id: str, evidence_list: List[Dict], output_dir: str = None):
    """
    Save both weighted and unweighted versions for Phase 4 ablation.
    """
    if output_dir is None:
        output_dir = WEIGHTED_EVIDENCE_PATH.replace("evidence_weighted.json", "")
    
    weighter = EvidenceWeighter()
    weighted = weighter.apply_weights(evidence_list)
    
    # Unweighted: all weights = 1.0
    unweighted = []
    for ev in evidence_list:
        ev_copy = ev.copy()
        ev_copy['tier_weight'] = 1.0
        ev_copy['weighted_score'] = ev_copy.get('score', 0.5)
        unweighted.append(ev_copy)
    
    result = {
        'claim_id': claim_id,
        'weighted_evidence': weighted,
        'unweighted_evidence': unweighted,
        'aggregate_weighted_score': weighter.aggregate_evidence_embedding(weighted),
        'aggregate_unweighted_score': weighter.aggregate_evidence_embedding(unweighted)
    }
    
    return result


if __name__ == "__main__":
    # Test
    print("=" * 70)
    print("TESTING HIERARCHY WEIGHTING")
    print("=" * 70)
    
    sample_evidence = [
        {"id": "doc1", "title": "RCT of Drug X", "predicted_tier": "A", "score": 0.85},
        {"id": "doc2", "title": "Cohort Study", "predicted_tier": "B", "score": 0.90},
        {"id": "doc3", "title": "Case Report", "predicted_tier": "C", "score": 0.80},
        {"id": "doc4", "title": "Editorial", "predicted_tier": "D", "score": 0.95}
    ]
    
    weighter = EvidenceWeighter()
    weighted = weighter.apply_weights(sample_evidence)
    
    print("\nOriginal order vs. Weighted order:")
    print(f"{'ID':<8} {'Tier':<6} {'Raw Score':<12} {'Weighted':<12}")
    print("-" * 40)
    for ev in weighted:
        print(f"{ev['id']:<8} {ev['predicted_tier']:<6} {ev['score']:<12.3f} {ev['weighted_score']:<12.3f}")
    
    print(f"\nAggregate trust score: {weighter.aggregate_evidence_embedding(weighted):.4f}")