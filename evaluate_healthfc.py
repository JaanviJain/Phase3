"""
Evaluate evidence tier classifier against HealthFC ground truth.
Maps T1/T2/T3/T4 citation types to A/B/C/D quality tiers for evaluation.
"""

import os
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from config import HEALTHFC_DIR, HEALTHFC_CSV, HEALTHFC_URL, EVALUATION_REPORT_PATH
from evidence_classifier import get_classifier


def download_healthfc():
    """Attempt to download HealthFC dataset."""
    os.makedirs(HEALTHFC_DIR, exist_ok=True)
    if os.path.exists(HEALTHFC_CSV):
        print(f"[OK] HealthFC found: {HEALTHFC_CSV}")
        return True
    print("[INFO] Attempting to download HealthFC...")
    try:
        urls = [
            HEALTHFC_URL,
            "https://raw.githubusercontent.com/lingbozhang/healthfc/main/healthfc.csv",
            "https://raw.githubusercontent.com/lingbozhang/healthfc/master/data/healthfc.csv"
        ]
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    with open(HEALTHFC_CSV, 'wb') as f:
                        f.write(r.content)
                    print(f"[OK] Downloaded HealthFC")
                    return True
            except:
                continue
    except Exception as e:
        print(f"[FAIL] {e}")
    print("\n[MANUAL] Download from https://github.com/lingbozhang/healthfc")
    return False


def citation_to_tier(citation_type):
    """
    Map HealthFC citation types to evidence quality tiers.
    This is a reasoned mapping based on typical evidence quality.
    """
    if pd.isna(citation_type):
        return 'B'
    
    s = str(citation_type).upper().strip()
    
    # Primary research and guidelines are typically higher quality
    if 'T1' in s or 'PRIMARY' in s:
        # Primary could be RCT (A) or observational (B) - average to B
        return 'B'
    
    # Guidelines are usually evidence-based but not always RCT
    if 'T2' in s or 'GUIDELINE' in s:
        return 'B'
    
    # References are mixed quality
    if 'T3' in s or 'REFERENCE' in s:
        return 'C'
    
    # Supplementary is typically lowest quality
    if 'T4' in s or 'SUPPLEMENTARY' in s:
        return 'D'
    
    return 'B'


def evaluate_classifier(use_llm: bool = False):
    """
    Evaluate classifier using T1-T4 mapped to A/B/C/D as ground truth.
    Also classifies evidence_text directly for comparison.
    """
    print("=" * 70)
    print("HEALTHFC EVIDENCE TIER EVALUATION")
    print("=" * 70)
    
    if not download_healthfc():
        print("[SKIP] HealthFC not available.")
        return None
    
    df = pd.read_csv(HEALTHFC_CSV)
    print(f"[INFO] Loaded HealthFC: {len(df)} rows")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    # Check required columns
    if 'evidence_tier' not in df.columns or 'evidence_text' not in df.columns:
        print("[ERROR] Required columns missing!")
        return None
    
    # Show raw distribution
    print(f"\n[INFO] Raw citation types in CSV:")
    print(df['evidence_tier'].value_counts())
    
    # Create ground truth by mapping T1-T4 to A/B/C/D
    df['true_tier'] = df['evidence_tier'].apply(citation_to_tier)
    print(f"\n[INFO] Mapped tier distribution (ground truth):")
    print(df['true_tier'].value_counts().sort_index())
    
    # Classify using evidence_text
    classifier = get_classifier(use_llm=use_llm)
    predictions = []
    
    print(f"\n[INFO] Classifying {len(df)} evidence pieces...")
    for idx, row in df.iterrows():
        text = str(row.get('evidence_text', ''))
        if text == 'nan' or len(text.strip()) < 10:
            text = str(row.get('claim_text', ''))
        
        title = text[:300]
        abstract = text[300:]
        tier, weight, reason = classifier.classify(title, abstract)
        predictions.append(tier)
    
    df['predicted_tier'] = predictions
    
    print(f"\n[INFO] Predicted tier distribution:")
    print(pd.Series(predictions).value_counts().sort_index())
    
    # REAL METRICS
    y_true = df['true_tier'].tolist()
    y_pred = df['predicted_tier'].tolist()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=['A', 'B', 'C', 'D'], zero_division=0
    )
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Macro P:   {precision:.4f}")
    print(f"Macro R:   {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print(f"\nPer-Class Metrics:")
    for i, tier in enumerate(['A', 'B', 'C', 'D']):
        p, r, f, s = per_class[0][i], per_class[1][i], per_class[2][i], per_class[3][i]
        print(f"  Tier {tier}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, Support={s}")
    
    # Report
    output = []
    output.append("=" * 70)
    output.append("PHASE 3: EVIDENCE HIERARCHY EVALUATION REPORT")
    output.append("=" * 70)
    output.append(f"NOTE: Ground truth created by mapping citation types:")
    output.append(f"      T1_Primary -> B, T2_Guideline -> B, T3_Reference -> C, T4_Supplementary -> D")
    output.append(f"")
    output.append(f"Total samples: {len(df)}")
    output.append(f"Accuracy:  {accuracy:.4f}")
    output.append(f"Macro P:   {precision:.4f}")
    output.append(f"Macro R:   {recall:.4f}")
    output.append(f"Macro F1:  {f1:.4f}")
    output.append(f"")
    output.append("Classification Report:")
    output.append(classification_report(y_true, y_pred, labels=['A','B','C','D'], zero_division=0))
    output.append("Confusion Matrix (A/B/C/D):")
    output.append(str(confusion_matrix(y_true, y_pred, labels=['A','B','C','D'])))
    output.append("True Tier Distribution (from T1-T4 mapping):")
    output.append(str(df['true_tier'].value_counts().sort_index()))
    output.append("Predicted Tier Distribution (from evidence_text):")
    output.append(str(df['predicted_tier'].value_counts().sort_index()))
    
    report_text = "\n".join(output)
    print("\n" + report_text)
    
    with open(EVALUATION_REPORT_PATH, 'w') as f:
        f.write(report_text)
    print(f"\n[SAVE] Report: {EVALUATION_REPORT_PATH}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': f1,
        'predictions': df
    }


if __name__ == "__main__":
    evaluate_classifier(use_llm=False)