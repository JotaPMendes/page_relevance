import requests
import json
from bs4 import BeautifulSoup
import nltk
from collections import Counter
import re
from datetime import datetime
from urllib.parse import urlparse
import time
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class SiteAnalyzer:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('portuguese'))
        self.stop_words.update(['que', 'para', 'com', 'uma', 'mais', 'muito', 'pode', 'ser', 'tem', 'vai', 'seu', 'sua', 'isso', 'essa', 'este'])
        
    def categorize_keyword(self, term, frequency):
        destination_keywords = ['porto', 'galinhas', 'pernambuco', 'recife', 'nordeste', 'brasil']
        timing_keywords = ['época', 'quando', 'mês', 'temporada', 'período', 'clima']
        activity_keywords = ['mergulho', 'praia', 'passeio', 'turismo', 'viagem', 'hotel', 'pousada']
        price_keywords = ['preço', 'custo', 'valor', 'barato', 'caro', 'orçamento']
        
        if any(keyword in term.lower() for keyword in destination_keywords):
            return 'destination'
        elif any(keyword in term.lower() for keyword in timing_keywords):
            return 'timing'
        elif any(keyword in term.lower() for keyword in activity_keywords):
            return 'activity'
        elif any(keyword in term.lower() for keyword in price_keywords):
            return 'price'
        else:
            return 'general'

    def calculate_relevance_score(self, term, frequency, total_words, category):
        base_score = (frequency / total_words) * 1000
        
        category_multipliers = {
            'destination': 2.0,
            'timing': 1.8,
            'activity': 1.5,
            'price': 1.7,
            'general': 1.0
        }
        
        multiplier = category_multipliers.get(category, 1.0)
        relevance_score = base_score * multiplier
        
        return round(min(relevance_score, 10.0), 2)

    def extract_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            print(f"Erro ao extrair conteúdo de {url}: {e}")
            return None

    def process_text(self, text):
        original_text = text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        words = [word for word in text.split() if len(word) > 2]
        filtered_words = [word for word in words if word not in self.stop_words]
        
        unigrams = Counter(filtered_words)
        
        bigrams = []
        for i in range(len(filtered_words) - 1):
            bigram = f"{filtered_words[i]} {filtered_words[i+1]}"
            bigrams.append(bigram)
        bigrams_count = Counter(bigrams)
        
        trigrams = []
        for i in range(len(filtered_words) - 2):
            trigram = f"{filtered_words[i]} {filtered_words[i+1]} {filtered_words[i+2]}"
            trigrams.append(trigram)
        trigrams_count = Counter(trigrams)
        
        return original_text, unigrams, bigrams_count, trigrams_count, len(filtered_words)

    def calculate_sentiment(self, text):
        positive_words = ['melhor', 'bom', 'boa', 'excelente', 'incrível', 'lindo', 'maravilhoso']
        negative_words = ['ruim', 'péssimo', 'caro', 'problema', 'difícil', 'complicado']
        
        positive_count = sum(text.lower().count(word) for word in positive_words)
        negative_count = sum(text.lower().count(word) for word in negative_words)
        
        if positive_count + negative_count > 0:
            return positive_count / (positive_count + negative_count)
        return 0.5

    def analyze_site(self, site_data):
        url = site_data['url']
        site_name = site_data['name']
        site_type = site_data['type']
        
        print(f"Analisando: {site_name}")
        
        content = self.extract_content(url)
        if not content:
            return None
        
        original_text, unigrams, bigrams_count, trigrams_count, total_relevant_words = self.process_text(content)
        
        domain = urlparse(url).netloc
        analysis_id = f"PG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{domain.replace('.', '_')}"
        
        keyword_insights = {
            'destination_terms': [],
            'timing_terms': [],
            'activity_terms': [],
            'price_terms': [],
            'general_terms': []
        }
        
        all_terms = []
        
        for term, freq in unigrams.most_common(30):
            category = self.categorize_keyword(term, freq)
            relevance_score = self.calculate_relevance_score(term, freq, total_relevant_words, category)
            
            term_data = {
                'term': term,
                'type': 'unigram',
                'frequency': freq,
                'relevance_score': relevance_score,
                'category': category
            }
            all_terms.append(term_data)
            keyword_insights[f'{category}_terms'].append(term_data)
        
        for term, freq in bigrams_count.most_common(20):
            category = self.categorize_keyword(term, freq)
            relevance_score = self.calculate_relevance_score(term, freq, total_relevant_words, category)
            
            term_data = {
                'term': term,
                'type': 'bigram',
                'frequency': freq,
                'relevance_score': relevance_score,
                'category': category
            }
            all_terms.append(term_data)
            keyword_insights[f'{category}_terms'].append(term_data)
        
        for term, freq in trigrams_count.most_common(15):
            category = self.categorize_keyword(term, freq)
            relevance_score = self.calculate_relevance_score(term, freq, total_relevant_words, category)
            
            term_data = {
                'term': term,
                'type': 'trigram',
                'frequency': freq,
                'relevance_score': relevance_score,
                'category': category
            }
            all_terms.append(term_data)
            keyword_insights[f'{category}_terms'].append(term_data)
        
        sentiment_score = self.calculate_sentiment(original_text)
        
        results = {
            "analysis_id": analysis_id,
            "source": {
                "site_name": site_name,
                "url": url,
                "domain": domain,
                "site_category": site_type,
                "scraping_date": datetime.now().isoformat()
            },
            "content_analysis": {
                "total_words": len(original_text.split()),
                "relevant_words": total_relevant_words,
                "unique_words": len(unigrams),
                "sentiment_score": round(sentiment_score, 2),
                "porto_galinhas_mentions": original_text.lower().count("porto de galinhas")
            },
            "keyword_insights": keyword_insights,
            "top_keywords_by_relevance": sorted(all_terms, key=lambda x: x['relevance_score'], reverse=True)[:15],
            "marketing_insights": {
                "primary_focus": self.identify_primary_focus(keyword_insights),
                "target_audience_intent": self.identify_audience_intent(keyword_insights),
                "content_type": site_type,
                "ad_targeting_potential": self.calculate_ad_potential(all_terms, sentiment_score)
            }
        }
        
        return results

    def identify_primary_focus(self, insights):
        category_scores = {}
        for category, terms in insights.items():
            if terms:
                category_scores[category.replace('_terms', '')] = sum(term['relevance_score'] for term in terms)
        
        return max(category_scores, key=category_scores.get) if category_scores else "general"

    def identify_audience_intent(self, insights):
        if insights['timing_terms']:
            return "planning_phase"
        elif insights['activity_terms']:
            return "research_phase"
        elif insights['price_terms']:
            return "decision_phase"
        else:
            return "awareness_phase"

    def calculate_ad_potential(self, terms, sentiment):
        high_relevance_terms = [t for t in terms if t['relevance_score'] > 5.0]
        
        if len(high_relevance_terms) > 10 and sentiment > 0.6:
            return "high"
        elif len(high_relevance_terms) > 5 and sentiment > 0.4:
            return "medium"
        else:
            return "low"

class MultiSiteScraper:
    def __init__(self, sources_file):
        self.analyzer = SiteAnalyzer()
        self.sources_file = sources_file
        self.results = []

    def load_sources(self):
        with open(self.sources_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['sites']

    def scrape_all_sites(self):
        sites = self.load_sources()
        
        print(f"Iniciando análise de {len(sites)} sites...")
        
        for i, site in enumerate(sites, 1):
            print(f"\n[{i}/{len(sites)}] Processando: {site['name']}")
            
            result = self.analyzer.analyze_site(site)
            if result:
                self.results.append(result)
                self.save_individual_result(result)
            
            time.sleep(2)
        
        self.save_consolidated_results()
        self.generate_comparison_report()

    def save_individual_result(self, result):
        os.makedirs("page_relevance/data/raw", exist_ok=True)
        
        filename = f"page_relevance/data/raw/{result['analysis_id']}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def save_consolidated_results(self):
        consolidated = {
            "scraping_session": {
                "timestamp": datetime.now().isoformat(),
                "total_sites_analyzed": len(self.results),
                "sites_data": self.results
            }
        }
        
        filename = f"page_relevance/data/raw/consolidated_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, ensure_ascii=False, indent=2)
        
        print(f"\nResultados consolidados salvos em: {filename}")

    def generate_comparison_report(self):
        if not self.results:
            return
        
        comparison = {
            "summary": {
                "total_sites": len(self.results),
                "average_sentiment": sum(r['content_analysis']['sentiment_score'] for r in self.results) / len(self.results),
                "total_porto_galinhas_mentions": sum(r['content_analysis']['porto_galinhas_mentions'] for r in self.results)
            },
            "site_rankings": {
                "by_relevance": sorted(self.results, key=lambda x: len(x['top_keywords_by_relevance']), reverse=True),
                "by_sentiment": sorted(self.results, key=lambda x: x['content_analysis']['sentiment_score'], reverse=True),
                "by_porto_mentions": sorted(self.results, key=lambda x: x['content_analysis']['porto_galinhas_mentions'], reverse=True)
            },
            "recommendations": self.generate_recommendations()
        }
        
        filename = f"page_relevance/data/raw/comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        self.print_summary(comparison)

    def generate_recommendations(self):
        high_potential = [r for r in self.results if r['marketing_insights']['ad_targeting_potential'] == 'high']
        
        return {
            "best_sites_for_ads": [r['source']['site_name'] for r in high_potential],
            "primary_keywords": self.get_most_common_keywords(),
            "content_strategy": "Focus on timing and activity-based content"
        }

    def get_most_common_keywords(self):
        all_keywords = {}
        for result in self.results:
            for keyword in result['top_keywords_by_relevance'][:5]:
                term = keyword['term']
                if term in all_keywords:
                    all_keywords[term] += keyword['relevance_score']
                else:
                    all_keywords[term] = keyword['relevance_score']
        
        return sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:10]

    def print_summary(self, comparison):
        print("\n" + "="*60)
        print("RELATÓRIO DE ANÁLISE CONSOLIDADA")
        print("="*60)
        
        print(f"Sites analisados: {comparison['summary']['total_sites']}")
        print(f"Sentimento médio: {comparison['summary']['average_sentiment']:.2f}")
        print(f"Total menções 'Porto de Galinhas': {comparison['summary']['total_porto_galinhas_mentions']}")
        
        print("\nTOP 3 SITES POR RELEVÂNCIA:")
        for i, site in enumerate(comparison['site_rankings']['by_relevance'][:3], 1):
            print(f"{i}. {site['source']['site_name']} - {len(site['top_keywords_by_relevance'])} keywords relevantes")
        
        print("\nTOP 5 KEYWORDS GLOBAIS:")
        for i, (keyword, score) in enumerate(comparison['recommendations']['primary_keywords'][:5], 1):
            print(f"{i}. {keyword} - Score: {score:.2f}")

def main():
    scraper = MultiSiteScraper("page_relevance/config/sources.json")
    scraper.scrape_all_sites()

if __name__ == "__main__":
    main()