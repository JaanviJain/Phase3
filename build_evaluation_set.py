"""
Build Phase 3 Evaluation Set from PubMed
Downloads real abstracts with known study types (A/B/C/D).
"""

import os
import json
import time
import requests
from xml.etree import ElementTree as ET
from config import DATA_DIR

PUBMED_EVAL_DIR = os.path.join(DATA_DIR, "pubmed_evaluation")
os.makedirs(PUBMED_EVAL_DIR, exist_ok=True)

EVALUATION_SET_PATH = os.path.join(PUBMED_EVAL_DIR, "phase3_evaluation_set.json")

# PubMed E-utilities
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pmids(query, retmax=300):
    """Search PubMed and return PMIDs."""
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': retmax,
        'retmode': 'json'
    }
    try:
        r = requests.get(ESEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data['esearchresult']['idlist']
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        return []


def fetch_abstracts(pmids, batch_size=100):
    """Fetch title and abstract for PMIDs."""
    all_articles = []
    
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        print(f"  Fetching batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1} ({len(batch)} articles)...")
        
        params = {
            'db': 'pubmed',
            'id': ','.join(batch),
            'retmode': 'xml'
        }
        
        try:
            r = requests.get(EFETCH_URL, params=params, timeout=60)
            r.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(r.content)
            
            for article in root.findall('.//PubmedArticle'):
                pmid_elem = article.find('.//PMID')
                pmid = pmid_elem.text if pmid_elem is not None else 'unknown'
                
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else ''
                
                # Get abstract text
                abstract_texts = []
                abstract_elem = article.find('.//Abstract')
                if abstract_elem is not None:
                    for text_elem in abstract_elem.findall('.//AbstractText'):
                        if text_elem.text:
                            abstract_texts.append(text_elem.text)
                
                abstract = ' '.join(abstract_texts)
                
                if title or abstract:
                    all_articles.append({
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract,
                        'text': f"{title} {abstract}".strip()
                    })
            
            # Rate limit: 3 requests per second max
            time.sleep(0.34)
            
        except Exception as e:
            print(f"[ERROR] Fetch failed for batch: {e}")
            time.sleep(1)
    
    return all_articles


def build_evaluation_set():
    """Download and save evaluation set."""
    print("=" * 70)
    print("BUILDING PHASE 3 EVALUATION SET FROM PUBMED")
    print("=" * 70)
    
    queries = {
        'A': '"randomized controlled trial"[pt] AND 2020:2024[dp]',
        'B': '"cohort studies"[mh] AND 2020:2024[dp]',
        'C': '"case reports"[pt] AND 2020:2024[dp]',
        'D': '"editorial"[pt] AND 2020:2024[dp]'
    }
    
    evaluation_set = []
    
    for tier, query in queries.items():
        print(f"\n[Tier {tier}] Searching: {query}")
        pmids = search_pmids(query, retmax=300)
        print(f"  Found {len(pmids)} PMIDs")
        
        if pmids:
            articles = fetch_abstracts(pmids[:300])  # Max 300
            print(f"  Fetched {len(articles)} articles with text")
            
            for article in articles:
                article['true_tier'] = tier
                evaluation_set.append(article)
    
    # Save
    with open(EVALUATION_SET_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluation_set, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"✅ SAVED: {EVALUATION_SET_PATH}")
    print(f"   Total articles: {len(evaluation_set)}")
    
    # Distribution
    from collections import Counter
    dist = Counter(a['true_tier'] for a in evaluation_set)
    for tier in ['A', 'B', 'C', 'D']:
        print(f"   Tier {tier}: {dist.get(tier, 0)} articles")
    print(f"{'='*70}")
    
    return evaluation_set


def load_evaluation_set():
    """Load existing evaluation set."""
    if not os.path.exists(EVALUATION_SET_PATH):
        print(f"[INFO] Evaluation set not found. Building...")
        return build_evaluation_set()
    
    with open(EVALUATION_SET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[OK] Loaded evaluation set: {len(data)} articles")
    return data


if __name__ == "__main__":
    build_evaluation_set()