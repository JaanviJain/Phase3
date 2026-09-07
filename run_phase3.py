"""
Phase 3 Main Runner
Orchestrates: PubMed evaluation → weighting → ablation.
NO TRAINING.
"""

import sys
from config import OUTPUT_DIR
from evidence_classifier import get_classifier
from hierarchy_weights import EvidenceWeighter
from evaluate_pubmed import evaluate_on_pubmed, evaluate_on_pubmed_llm
from ablation_study import prepare_ablation_data, load_phase2_evidence
from collections import Counter


def main():
    print("=" * 70)
    print("PHASE 3: EVIDENCE HIERARCHY & WEIGHTING")
    print("=" * 70)
    
    # Mode selection
    mode = "rule"
    if len(sys.argv) > 1:
        if sys.argv[1] == "--llm":
            mode = "llm"
        elif sys.argv[1] == "--hybrid":
            mode = "hybrid"
    
    print(f"Mode: {mode.upper()}")
    if mode == "rule":
        print("Using: Rule-based classifier (CPU, fast)")
    elif mode == "llm":
        print("Using: LLM classifier (GPU, accurate, slow)")
        print("Requires: Ollama running with llama3.1:8b")
    elif mode == "hybrid":
        print("Using: Hybrid (Rule for clear cases, LLM for ambiguous)")
    
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # ======================================================================
    # STEP 1: Evaluate on PubMed
    # ======================================================================
    print("STEP 1: Evaluating evidence classifier...")
    
    if mode == "llm":
        from llm_classifier import get_llm_classifier
        classifier = get_llm_classifier()
        eval_result = evaluate_on_pubmed_llm(classifier)
    else:
        eval_result = evaluate_on_pubmed(use_llm=False)
    
    if eval_result:
        print(f"\n[RESULT] Macro F1: {eval_result['macro_f1']:.4f}")
        print(f"[RESULT] Accuracy: {eval_result['accuracy']:.4f}")
    else:
        print("[SKIP] Evaluation not available.")
    
    # ======================================================================
    # STEP 2: Ablation
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Preparing ablation data")
    print("=" * 70)
    
    claims_ev = load_phase2_evidence()
    
    if mode == "llm":
        from llm_classifier import get_llm_classifier
        classifier = get_llm_classifier()
        # Pass the pre-created classifier to avoid re-loading
        weighted, unweighted = prepare_ablation_data(claims_ev, classifier=classifier)
    else:
        weighted, unweighted = prepare_ablation_data(claims_ev, use_llm=False)
    
    # ======================================================================
    # STEP 3: Summary Statistics
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Phase 3 Summary")
    print("=" * 70)
    
    if weighted:
        all_tiers = []
        for claim in weighted:
            for ev in claim['evidence']:
                all_tiers.append(ev['predicted_tier'])
        
        tier_counts = Counter(all_tiers)
        total = sum(tier_counts.values())
        
        print(f"\nEvidence tier distribution across {len(weighted)} claims:")
        
        tier_weights = {'A': '1.0', 'B': '0.7', 'C': '0.4', 'D': '0.2'}
        for tier in ['A', 'B', 'C', 'D']:
            count = tier_counts.get(tier, 0)
            pct = (count / total * 100) if total > 0 else 0
            weight_str = tier_weights.get(tier, '0.0')
            print(f"  Tier {tier} (weight={weight_str}): {count:4d} ({pct:5.1f}%)")
        
        print(f"\nTotal evidence pieces: {total}")
        print(f"Average evidence per claim: {total / len(weighted):.1f}")
    
    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print("Outputs generated:")
    print(f"  1. PubMed evaluation report: {OUTPUT_DIR}/pubmed_evaluation_report.txt")
    print(f"  2. Weighted evidence: {OUTPUT_DIR}/evidence_weighted.json")
    print(f"  3. Unweighted evidence: {OUTPUT_DIR}/evidence_unweighted.json")
    print("\nNext step: Train Phase 4 verifier using weighted vs. unweighted evidence.")
    print("Run ablation: Train Phase 4 with BOTH files and compare F1 scores.")


if __name__ == "__main__":
    main()