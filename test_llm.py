"""
Balanced test: 12-13 samples from each tier (A/B/C/D) = ~50 total
This gives a FAIR F1 score.
"""

from llm_classifier import get_llm_classifier
from evaluate_pubmed import load_evaluation_set
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from collections import Counter

def test_llm_balanced():
    print("=" * 70)
    print("TESTING LLM ON BALANCED 50 SAMPLES (12-13 per tier)")
    print("=" * 70)
    
    data = load_evaluation_set()
    if not data:
        print("[ERROR] No evaluation set found.")
        return
    
    # Group by true tier
    by_tier = {'A': [], 'B': [], 'C': [], 'D': []}
    for article in data:
        tier = article['true_tier']
        if tier in by_tier:
            by_tier[tier].append(article)
    
    # Take first 13 from each tier (or all available if less)
    test_data = []
    for tier in ['A', 'B', 'C', 'D']:
        samples = by_tier[tier][:13]
        test_data.extend(samples)
        print(f"[INFO] Tier {tier}: {len(samples)} samples available")
    
    print(f"[INFO] Total balanced test set: {len(test_data)} samples")
    print(f"[INFO] Distribution: {Counter(a['true_tier'] for a in test_data)}")
    print()
    
    classifier = get_llm_classifier()
    
    y_true = []
    y_pred = []
    
    for i, article in enumerate(test_data):
        true_tier = article['true_tier']
        title = article.get('title', '') or ''
        abstract = article.get('abstract', '') or ''
        
        pred_tier, weight, reason = classifier.classify(title, abstract)
        
        # Show first 5 for inspection
        if i < 5:
            print(f"  Sample {i+1} (True: {true_tier}):")
            print(f"    Title: {title[:50]}...")
            print(f"    Predicted: {pred_tier} | Reason: {reason}")
            print()
        
        y_true.append(true_tier)
        y_pred.append(pred_tier)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=['A','B','C','D'], average='macro', zero_division=0
    )
    
    # Per-class
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=['A','B','C','D'], zero_division=0
    )
    
    print(f"{'='*70}")
    print(f"RESULTS ON BALANCED SET:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    print(f"  Macro P:   {precision:.4f}")
    print(f"  Macro R:   {recall:.4f}")
    print(f"\nPer-Class:")
    for i, tier in enumerate(['A','B','C','D']):
        p, r, f, s = per_class[0][i], per_class[1][i], per_class[2][i], per_class[3][i]
        print(f"  Tier {tier}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, Support={s}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=['A','B','C','D'], zero_division=0))
    
    print(f"Confusion Matrix (A/B/C/D):")
    print(confusion_matrix(y_true, y_pred, labels=['A','B','C','D']))
    
    print(f"{'='*70}")
    
    if f1 >= 0.80:
        print("[STATUS] ✅ EXCELLENT! Run full 1,200 with LLM!")
    elif f1 >= 0.70:
        print("[STATUS] 🟡 GOOD. LLM beats rule-based (77%). Run full 1,200.")
    elif f1 >= 0.60:
        print("[STATUS] 🟡 DECENT. Comparable to rule-based. Your choice.")
    else:
        print("[STATUS] 🔴 FAILED. Abandon LLM. Use rule-based 77%.")
    
    return f1

if __name__ == "__main__":
    test_llm_balanced()