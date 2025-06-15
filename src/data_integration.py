import os
import glob
import json
import pandas as pd
import csv
from term_sentiment import analyze_term_sentiments

def get_latest_analysis_detailed(results_dir='results'):
    """Retorna o caminho do arquivo analysis_detailed_*.json mais recente."""
    files = glob.glob(os.path.join(results_dir, 'analysis_detailed_*.json'))
    if not files:
        raise FileNotFoundError("Nenhum arquivo analysis_detailed_*.json encontrado em results/")
    latest = max(files, key=os.path.getctime)
    return latest

def update_sites_csv_from_source(source_json='config/source.json', csv_path='power_bi_data/sites.csv'):
    """Atualiza o sites.csv a partir do config/source.json."""
    with open(source_json, encoding='utf-8') as f:
        sources = json.load(f)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['site_name', 'site_number'])
        for site in sources['sites']:
            writer.writerow([site['name'], site['id']])
    print(f"Arquivo {csv_path} atualizado com sucesso!")

def generate_powerbi_csvs_from_analysis(analysis_json, sites_csv='power_bi_data/sites.csv', output_dir='power_bi_data'):
    """Gera os arquivos CSV para o Power BI a partir do arquivo de análise detalhada."""
    with open(analysis_json, encoding='utf-8') as f:
        data = json.load(f)
    df_site_number_mapping = pd.read_csv(sites_csv)

    # 1. terms_analysis.csv
    terms_data = []
    site_texts = {site['site_info']['name']: site['site_info'].get('content', '') for site in data['results']}
    for site in data['results']:
        site_name = site['site_info']['name']
        site_text = site_texts.get(site_name, '')
        # Coletar todos os termos únicos deste site
        site_terms = list(set([term_info['term'] for category_type, terms in site['categorized_terms'].items() for term_info in terms]))
        # Calcular sentimento médio dos termos neste site
        term_sentiments = analyze_term_sentiments(site_text, site_terms) if site_text else {}
        for category_type, terms in site['categorized_terms'].items():
            for term_info in terms:
                terms_data.append({
                    'site_name': site_name,
                    'term': term_info['term'],
                    'type': term_info['type'],
                    'frequency': term_info['frequency'],
                    'relevance_score': term_info['relevance_score'],
                    'category': term_info['category'],
                    'detailed_category': term_info['detailed_category'],
                    'term_sentiment': term_sentiments.get(term_info['term'])
                })
    df_terms_analysis = pd.DataFrame(terms_data)
    df_terms_analysis = pd.merge(df_terms_analysis, df_site_number_mapping, on='site_name', how='left')
    df_terms_analysis = df_terms_analysis[['site_number', 'term', 'type', 'frequency', 'relevance_score', 'category', 'detailed_category', 'term_sentiment']]
    df_terms_analysis.to_csv(os.path.join(output_dir, 'terms_analysis.csv'), index=False)

    # 2. site_metrics.csv
    site_metrics_data = []
    for site in data['results']:
        site_name = site['site_info']['name']
        total_terms_frequency = 0
        total_relevance_score = 0
        num_terms = 0
        for category_type, terms in site['categorized_terms'].items():
            for term_info in terms:
                total_terms_frequency += term_info['frequency']
                total_relevance_score += term_info['relevance_score']
                num_terms += 1
        avg_relevance_score = total_relevance_score / num_terms if num_terms > 0 else 0
        site_metrics_data.append({
            'site_name': site_name,
            'total_terms_frequency': total_terms_frequency,
            'avg_relevance_score': avg_relevance_score
        })
    df_site_metrics = pd.DataFrame(site_metrics_data)
    df_site_metrics = pd.merge(df_site_metrics, df_site_number_mapping, on='site_name', how='left')
    df_site_metrics = df_site_metrics[['site_number', 'total_terms_frequency', 'avg_relevance_score']]
    df_site_metrics.to_csv(os.path.join(output_dir, 'site_metrics.csv'), index=False)

    # 3. category_metrics.csv
    category_metrics_data = []
    for site in data['results']:
        site_name = site['site_info']['name']
        for category, metrics in site['category_summary'].items():
            category_metrics_data.append({
                'site_name': site_name,
                'category': category.replace('_terms', ''),
                'category_frequency': metrics['total_terms'],
                'category_relevance': metrics['avg_relevance_score']
            })
    df_category_metrics = pd.DataFrame(category_metrics_data)
    df_category_metrics = pd.merge(df_category_metrics, df_site_number_mapping, on='site_name', how='left')
    df_category_metrics = df_category_metrics[['site_number', 'category', 'category_frequency', 'category_relevance']]
    df_category_metrics.to_csv(os.path.join(output_dir, 'category_metrics.csv'), index=False)

    # 4. travel_terms.csv
    travel_related_categories = ['destination', 'activity', 'accommodation', 'planning', 'timing', 'price', 'sentiment']
    travel_terms_data = []
    for site in data['results']:
        site_name = site['site_info']['name']
        for category_type, terms in site['categorized_terms'].items():
            cleaned_category = category_type.replace('_terms', '')
            if cleaned_category in travel_related_categories:
                for term_info in terms:
                    travel_terms_data.append({
                        'site_name': site_name,
                        'term': term_info['term'],
                        'type': term_info['type'],
                        'frequency': term_info['frequency'],
                        'relevance_score': term_info['relevance_score'],
                        'category': term_info['category'],
                        'detailed_category': term_info['detailed_category']
                    })
    df_travel_terms = pd.DataFrame(travel_terms_data)
    df_travel_terms = pd.merge(df_travel_terms, df_site_number_mapping, on='site_name', how='left')
    df_travel_terms = df_travel_terms[['site_number', 'term', 'type', 'frequency', 'relevance_score', 'category', 'detailed_category']]
    df_travel_terms.to_csv(os.path.join(output_dir, 'travel_terms.csv'), index=False)

    print("Arquivos para Power BI atualizados com sucesso!")

def check_terms_sentiment_filled(terms_csv_path='power_bi_data/terms_analysis.csv'):
    """Verifica se a coluna term_sentiment está totalmente preenchida no terms_analysis.csv."""
    import pandas as pd
    df = pd.read_csv(terms_csv_path)
    missing = df['term_sentiment'].isnull() | (df['term_sentiment'] == '')
    if missing.any():
        missing_count = missing.sum()
        print(f"⚠️ Atenção: {missing_count} termos estão sem sentimento calculado em '{terms_csv_path}'.")
        return False
    print(f"✅ Todos os termos possuem sentimento calculado em '{terms_csv_path}'.")
    return True
