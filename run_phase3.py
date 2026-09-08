"""
Phase 3 Main Runner
Orchestrates: classification -> weighting -> ablation.
NO TRAINING.
"""

import sys
import os
from collections import Counter

try:
    from config import OUTPUT_DIR, WEIGHTED_EVIDENCE_PATH, UNWEIGHTED_EVIDENCE_PATH
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "phase_outputs")
    WEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_weighted.json")
    UNWEIGHTED_EVIDENCE_PATH = os.path.join(OUTPUT_DIR, "evidence_unweighted.json")

from evidence_classifier import get_classifier
from hierarchy_weights import EvidenceWeighter
from ablation_study import prepare_ablation_data, load_phase2_evidence


def main():
    print("=" * 70)
    print("PHASE 3: EVIDENCE HIERARCHY & WEIGHTING")
    print("=" * 70)
    
    # Mode selection
    mode = "rule"
    if len(sys.argv) > 1:
        if sys.argv[1] in ("--llm", "--ollama"):
            mode = "llm"
        elif sys.argv[1] == "--hybrid":
            mode = "hybrid"
    
    print(f"Mode: {mode.upper()}")
    if mode == "rule":
        print("Using: Rule-based classifier (CPU, fast)")
    elif mode == "llm":
        print("Using: Ollama classifier (slower, but more accurate)")
        print("Requires: ollama serve running")
    
    print(f"Output directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print()
    
    # ==================================================================
    # STEP 1: Load Phase 2 evidence
    # ==================================================================
    print("STEP 1: Loading Phase 2 evidence...")
    claims_ev = load_phase2_evidence()
    print(f"Loaded {len(claims_ev)} claims from Phase 2.\n")
    
    # ==================================================================
    # STEP 2: Classify + Weight + Prepare Ablation
    # ==================================================================
    print("STEP 2: Classifying evidence and preparing ablation data...")
    weighted, unweighted = prepare_ablation_data(
        claims_ev, 
        use_llm=(mode == "llm"),
        top_k=5  # Keep top 5 evidence per claim for Phase 4
    )
    
    # ==================================================================
    # STEP 3: Summary Statistics
    # ==================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Phase 3 Summary")
    print("=" * 70)
    
    if weighted:
        all_tiers = []
        for claim in weighted:
            for ev in claim.get('evidence', []):
                all_tiers.append(ev.get('predicted_tier', 'U'))
        
        tier_counts = Counter(all_tiers)
        total = sum(tier_counts.values())
        
        print(f"\nEvidence tier distribution across {len(weighted)} claims:")
        tier_weights = {'A': '1.0', 'B': '0.7', 'C': '0.4', 'D': '0.2', 'U': '1.0'}
        for tier in ['A', 'B', 'C', 'D', 'U']:
            count = tier_counts.get(tier, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Tier {tier} (weight={tier_weights.get(tier, '0.0')}): {count:4d} ({pct:5.1f}%)")
        
        print(f"\nTotal evidence pieces: {total}")
        print(f"Average evidence per claim: {total / len(weighted):.1f}")
        
        # Check Phase 4 readiness
        labels_present = sum(1 for c in weighted if c.get('true_label') is not None)
        print(f"\nClaims with true_label (Phase 4 ready): {labels_present}/{len(weighted)}")
        if labels_present == 0:
            print("WARNING: No true_labels found. Phase 4 training will need manual labels.")
    
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print("Outputs generated:")
    print(f"  1. Weighted evidence:   {WEIGHTED_EVIDENCE_PATH}")
    print(f"  2. Unweighted evidence: {UNWEIGHTED_EVIDENCE_PATH}")
    print("\nNext step: Train Phase 4 verifier.")
    print("  - Use weighted file for hierarchy experiment")
    print("  - Use unweighted file for baseline experiment")
    print("  - Compare F1 scores to prove hierarchy helps.")
    print("=" * 70)


if __name__ == "__main__":
    main()
