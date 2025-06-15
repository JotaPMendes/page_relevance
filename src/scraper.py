import requests
from bs4 import BeautifulSoup
import re
import nltk
import json
import time
import os
import torch
from multiprocessing import Pool, cpu_count
from analyzer import TextAnalyzer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def check_gpu():
    """Verifica disponibilidade e status da GPU"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        memory_cached = torch.cuda.memory_reserved(0) / (1024**2)
        
        print(f"\n🎮 GPU Disponível: {gpu_name}")
        print(f"   - Memória Total: {total_memory:.0f}MB")
        print(f"   - Memória Em Uso: {memory_allocated:.0f}MB")
        print(f"   - Memória Cache: {memory_cached:.0f}MB")
        return True
    else:
        print("\n💻 GPU não encontrada, usando CPU")
        return False

# Get the absolute path to the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class WebScraper:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('portuguese'))
        self.stop_words.update(['que', 'para', 'com', 'uma', 'mais', 'muito', 'pode', 'ser', 'tem', 'vai', 'seu', 'sua', 'isso', 'essa', 'este'])
        
        # Verificar GPU antes de inicializar
        self.has_gpu = check_gpu()
        
        # Inicializar o analyzer com transformers
        print("🤖 Inicializando analyzer com transformers...")
        self.analyzer = TextAnalyzer()
        
        if self.has_gpu:
            torch.cuda.empty_cache()  # Limpa memória GPU
        
        print("✅ Analyzer carregado!")

    def load_urls_from_json(self, json_file_path):
        """Carrega URLs do arquivo JSON"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return data.get('sites', [])
        except Exception as e:
            print(f"❌ Erro ao carregar JSON: {e}")
            return []

    def extract_content(self, url):
        """Extrai conteúdo de uma URL com headers personalizados"""
        try:
            print(f"📥 Extraindo: {url}")
            
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "pt-BR,pt;q=0.9",
                "Referer": "https://www.google.com"
            }
            
            session = requests.Session()
            response = session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            time.sleep(3)  # Espera 3 segundos entre requisições
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            
            print(f"✅ Extraído: {len(text)} caracteres")
            return text

        except Exception as e:
            print(f"❌ Erro ao extrair conteúdo de {url}: {e}")
            return None

    def process_text(self, text):
        """Processa texto extraído"""
        original_text = text
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        words = [word for word in clean_text.split() if len(word) > 2]
        filtered_words = [word for word in words if word not in self.stop_words]
        
        return original_text, filtered_words, len(filtered_words)

    def analyze_single_site(self, site_data):
        """Analisa um único site com suporte a GPU"""
        if self.has_gpu:
            torch.cuda.empty_cache()  # Limpa cache antes de cada análise
            
        url = site_data.get('url')
        site_name = site_data.get('name', 'Site desconhecido')
        
        print(f"\n🔍 Analisando: {site_name}")
        print(f"URL: {url}")
        
        # 1. Extrair conteúdo
        content = self.extract_content(url)
        if not content:
            return None
        
        # 2. Processar texto
        original_text, filtered_words, total_words = self.process_text(content)
        
        if total_words < 10:
            print(f"⚠️ Muito pouco conteúdo relevante ({total_words} palavras)")
            return None
        
        print(f"📝 Processado: {total_words} palavras relevantes")
        
        # 3. Análise de sentimento
        print("💭 Analisando sentimento...")
        sentiment_result = self.analyzer.analyze_sentiment_transformer(original_text)
        
        # 4. Análise de n-gramas com contexto
        print("🔤 Analisando termos e categorias...")
        all_terms, unigrams = self.analyzer.analyze_ngrams(filtered_words, total_words, original_text)
        
        # 5. Categorizar termos
        categorized_terms = self.analyzer.categorize_terms(all_terms)
        
        # 6. Compilar resultados
        result = {
            'site_info': {
                'name': site_name,
                'url': url,
                'content_length': len(content),
                'relevant_words': total_words,
                'content': original_text  # Adiciona o texto do site para análise de sentimento contextual
            },
            'sentiment_analysis': sentiment_result,
            'top_terms': all_terms[:15],  # Top 15 termos
            'categorized_terms': categorized_terms,
            'category_summary': self._generate_category_summary(categorized_terms),
            'ai_usage': {
                'total_terms': len(all_terms),
                'contextual_classifications': len([t for t in all_terms if t['classification_method'] == 'contextual']),
                'keyword_fallbacks': len([t for t in all_terms if t['classification_method'] == 'keyword'])
            }
        }
        
        print(f"✅ Análise concluída para {site_name}")
        return result

    def _generate_category_summary(self, categorized_terms):
        """Gera resumo das categorias"""
        summary = {}
        
        for category, terms in categorized_terms.items():
            if terms:
                avg_score = sum(term['relevance_score'] for term in terms) / len(terms)
                top_terms = sorted(terms, key=lambda x: x['relevance_score'], reverse=True)[:3]
                
                summary[category] = {
                    'total_terms': len(terms),
                    'avg_relevance_score': round(avg_score, 2),
                    'top_terms': [term['term'] for term in top_terms]
                }
        
        return summary

    def scrape_all_sites(self, json_file_path):
        """Scraper principal - analisa todos os sites do JSON"""
        print("🚀 Iniciando scraping com analyzer_bart...")
        
        # Converter para caminho absoluto se necessário
        if not os.path.isabs(json_file_path):
            json_file_path = os.path.join(PROJECT_ROOT, json_file_path)
        
        sites = self.load_urls_from_json(json_file_path)
        if not sites:
            print("❌ Nenhum site encontrado no JSON")
            return []
        
        print(f"📋 {len(sites)} sites carregados do JSON")
        
        # Análise sequencial com analyzer_bart
        results = []
        for i, site in enumerate(sites, 1):
            print(f"[{i}/{len(sites)}] Analisando: {site.get('name', 'Site Desconhecido')}")
            
            try:
                result = self.analyze_single_site(site)
                if result:
                    results.append(result)
                    sentiment = result['sentiment_analysis']['overall_sentiment']
                    method = result['sentiment_analysis'].get('method', 'unknown')
                    print(f"  ✅ Sentiment: {sentiment:.3f} (método: {method})")
                else:
                    print(f"  ❌ Falha na análise")
            except Exception as e:
                print(f"  ❌ Erro na análise: {e}")
            
            # Pausa entre análises
            time.sleep(2)
        
        if self.has_gpu:
            torch.cuda.empty_cache()  # Limpa cache GPU ao finalizar
        
        print(f"\n{'='*60}")
        print(f"🎯 ANÁLISE CONCLUÍDA!")
        print(f"✅ {len(results)} sites analisados com sucesso")
        print(f"❌ {len(sites) - len(results)} sites falharam")
        
        if results:
            # Mostrar distribuição de sentimentos
            sentiments = [r['sentiment_analysis']['overall_sentiment'] for r in results]
            positive = len([s for s in sentiments if s > 0.55])
            neutral = len([s for s in sentiments if 0.4 <= s <= 0.55])
            negative = len([s for s in sentiments if s < 0.4])
            
            print(f"\n📊 DISTRIBUIÇÃO DE SENTIMENTOS:")
            print(f"  😊 Positivos: {positive} ({positive/len(results)*100:.1f}%)")
            print(f"  😐 Neutros: {neutral} ({neutral/len(results)*100:.1f}%)")
            print(f"  😞 Negativos: {negative} ({negative/len(results)*100:.1f}%)")
        
        return results

    def generate_summary_report(self, results):
        """Gera relatório resumo de todos os sites"""
        if not results:
            return "Nenhum resultado para gerar relatório"
        
        report = []
        report.append("📊 RELATÓRIO RESUMO - ANÁLISE DE SITES")
        report.append("="*50)
        
        total_sites = len(results)
        avg_sentiment = sum(r['sentiment_analysis']['overall_sentiment'] for r in results) / total_sites
        total_ai_classifications = sum(r['ai_usage']['contextual_classifications'] for r in results)
        total_fallbacks = sum(r['ai_usage']['keyword_fallbacks'] for r in results)
        
        report.append(f"\n🔢 ESTATÍSTICAS GERAIS:")
        report.append(f"  Sites analisados: {total_sites}")
        report.append(f"  Sentimento médio: {avg_sentiment:.3f}")
        report.append(f"  Classificações com IA: {total_ai_classifications}")
        report.append(f"  Fallbacks por palavra-chave: {total_fallbacks}")
        report.append(f"  Taxa de uso da IA: {(total_ai_classifications/(total_ai_classifications+total_fallbacks)*100):.1f}%")
        
        report.append(f"\n📋 ANÁLISE POR SITE:")
        for result in results:
            site_name = result['site_info']['name']
            sentiment = result['sentiment_analysis']['overall_sentiment']
            sentiment_label = "Positivo" if sentiment > 0.55 else "Neutro" if sentiment > 0.4 else "Negativo"
            
            report.append(f"\n  🌐 {site_name}")
            report.append(f"    Sentimento: {sentiment:.3f} ({sentiment_label})")
            report.append(f"    Palavras relevantes: {result['site_info']['relevant_words']}")
            
            categories = result['category_summary']
            top_categories = sorted(categories.items(), key=lambda x: x[1]['total_terms'], reverse=True)[:3]
            
            report.append(f"    Top categorias:")
            for cat_name, cat_data in top_categories:
                report.append(f"      - {cat_name}: {cat_data['total_terms']} termos")
        
        return "\n".join(report)

    def save_results(self, results, output_file='analysis_results.json'):
        """Salva resultados em JSON"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Resultados salvos em {output_file}")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")
