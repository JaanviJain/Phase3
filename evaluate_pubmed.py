"""
Evaluate Phase 3 classifier on PubMed evaluation set.
REAL ground truth: PubMed publication type filters.
"""

import os
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from evidence_classifier import get_classifier
from config import OUTPUT_DIR

EVALUATION_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "pubmed_evaluation", "phase3_evaluation_set.json")
PUBMED_REPORT_PATH = os.path.join(OUTPUT_DIR, "pubmed_evaluation_report.txt")


def load_evaluation_set():
    """Load PubMed evaluation set."""
    if not os.path.exists(EVALUATION_SET_PATH):
        print(f"[ERROR] Evaluation set not found at: {EVALUATION_SET_PATH}")
        print("[INFO] Run: python build_evaluation_set.py")
        return None
    
    with open(EVALUATION_SET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_on_pubmed(use_llm=False):
    """Evaluate classifier on real PubMed abstracts."""
    print("=" * 70)
    print("PHASE 3 EVALUATION: PUBMED CONTROLLED SET")
    print("=" * 70)
    print("Ground truth: PubMed publication type filters")
    print("A = RCT, B = Cohort, C = Case Report, D = Editorial")
    print("=" * 70)
    
    data = load_evaluation_set()
    if not data:
        return None
    
    print(f"[INFO] Evaluating {len(data)} articles...")
    
    classifier = get_classifier(use_llm=use_llm)
    
    y_true = []
    y_pred = []
    
    for article in data:
        true_tier = article['true_tier']
        
        # FIX: Handle None values + use FULL text
        title = article.get('title', '') or ''
        abstract = article.get('abstract', '') or ''
        
        # Pass full title and full abstract — NO TRUNCATION
        pred_tier, weight, reason = classifier.classify(title, abstract)
        
        y_true.append(true_tier)
        y_pred.append(pred_tier)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=['A', 'B', 'C', 'D'], average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=['A', 'B', 'C', 'D'], zero_division=0
    )
    
    # Report
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Total evaluated: {len(data)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")
    print(f"\nPer-Class Metrics:")
    for i, tier in enumerate(['A', 'B', 'C', 'D']):
        p, r, f, s = per_class[0][i], per_class[1][i], per_class[2][i], per_class[3][i]
        print(f"  Tier {tier}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=['A', 'B', 'C', 'D'], zero_division=0))
    
    print(f"Confusion Matrix (A/B/C/D):")
    print(confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C', 'D']))
    
    # Save report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PHASE 3: PUBMED EVALUATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Ground truth: PubMed publication type filters")
    report_lines.append(f"Total articles: {len(data)}")
    report_lines.append(f"Accuracy:  {accuracy:.4f}")
    report_lines.append(f"Macro P:   {precision:.4f}")
    report_lines.append(f"Macro R:   {recall:.4f}")
    report_lines.append(f"Macro F1:  {f1:.4f}")
    report_lines.append("")
    report_lines.append("Classification Report:")
    report_lines.append(classification_report(y_true, y_pred, labels=['A', 'B', 'C', 'D'], zero_division=0))
    report_lines.append("Confusion Matrix:")
    report_lines.append(str(confusion_matrix(y_true, y_pred, labels=['A', 'B', 'C', 'D'])))
    
    report_text = "\n".join(report_lines)
    
    with open(PUBMED_REPORT_PATH, 'w') as f:
        f.write(report_text)
    print(f"\n[SAVE] Report: {PUBMED_REPORT_PATH}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': f1,
        'per_class_precision': per_class[0].tolist(),
        'per_class_recall': per_class[1].tolist(),
        'per_class_f1': per_class[2].tolist(),
        'per_class_support': per_class[3].tolist()
    }

def evaluate_on_pubmed_llm(classifier):
    """Evaluate using LLM classifier (slower but more accurate)."""
    print("=" * 70)
    print("PHASE 3 EVALUATION: PUBMED CONTROLLED SET (LLM MODE)")
    print("=" * 70)
    
    data = load_evaluation_set()
    if not data:
        return None
    
    print(f"[INFO] Evaluating {len(data)} articles with LLM...")
    print(f"[INFO] This will take ~{len(data) * 3 // 60} minutes...")
    
    y_true = []
    y_pred = []
    
    for i, article in enumerate(data):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(data)}...")
        
        true_tier = article['true_tier']
        title = article.get('title', '') or ''
        abstract = article.get('abstract', '') or ''
        
        pred_tier, weight, reason = classifier.classify(title, abstract)
        
        y_true.append(true_tier)
        y_pred.append(pred_tier)
    
    # Same metrics as before
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=['A','B','C','D'], average='macro', zero_division=0
    )
    
    print(f"\n[RESULT] LLM Accuracy: {accuracy:.4f}")
    print(f"[RESULT] LLM Macro F1: {f1:.4f}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "pubmed_evaluation_llm_report.txt")
    # ... (save same format as rule-based)
    
    return {'accuracy': accuracy, 'macro_f1': f1}


if __name__ == "__main__":
    evaluate_on_pubmed(use_llm=False)