import json
import os
from datetime import datetime
from urllib.parse import urlparse

def load_sources(sources_file):
    with open(sources_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['sites']

def save_json_file(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_analysis_id(url):
    domain = urlparse(url).netloc
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"PG_{timestamp}_{domain.replace('.', '_')}"

def identify_primary_focus(insights):
    category_scores = {}
    for category, terms in insights.items():
        if terms:
            category_scores[category.replace('_terms', '')] = sum(term['relevance_score'] for term in terms)
    
    return max(category_scores, key=category_scores.get) if category_scores else "general"

def identify_audience_intent(insights):
    if insights['timing_terms']:
        return "planning_phase"
    elif insights['activity_terms']:
        return "research_phase"
    elif insights['price_terms']:
        return "decision_phase"
    else:
        return "awareness_phase"

def calculate_ad_potential(terms, sentiment):
    high_relevance_terms = [t for t in terms if t['relevance_score'] > 5.0]
    
    if len(high_relevance_terms) > 10 and sentiment > 0.55:
        return "high"
    elif len(high_relevance_terms) > 5 and sentiment > 0.4:
        return "medium"
    else:
        return "low"