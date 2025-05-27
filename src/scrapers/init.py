from analyzer import TextAnalyzer
from scraper import WebScraper
import os
import json
from datetime import datetime

def generate_analysis_id():
    """Gera ID único para a análise"""
    return f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def run_complete_analysis(json_path=None, output_dir=None):
    """
    Executa análise completa de todos os sites do JSON com salvamento progressivo
    
    Args:
        json_path: Caminho para o arquivo JSON com os sites
        output_dir: Diretório para salvar os resultados
    """
    
    print("🚀 INICIANDO ANÁLISE COMPLETA DE SITES")
    print("="*60)
    
    # Gerar timestamp e ID únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = generate_analysis_id()
    
    # Definir caminhos padrão se não fornecidos
    if json_path is None:
        # Obter diretório raiz do projeto (page_relevance)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        
        # Navegar até encontrar a pasta page_relevance
        while project_root and not project_root.endswith('page_relevance'):
            parent = os.path.dirname(project_root)
            if parent == project_root:  # Chegou na raiz do sistema
                break
            project_root = parent
        
        # Caminhos possíveis para sources.json
        possible_paths = [
            os.path.join(project_root, "config", "sources.json"),
            os.path.join(os.path.dirname(current_dir), "config", "sources.json"),
            os.path.join(os.path.dirname(os.path.dirname(current_dir)), "config", "sources.json"),
            "../../config/sources.json",
            "../config/sources.json",
            "config/sources.json",
            "sources.json"
        ]
        
        json_path = None
        for path in possible_paths:
            if os.path.exists(path):
                json_path = path
                print(f"✅ JSON encontrado em: {path}")
                break
        
        if json_path is None:
            print("❌ Arquivo sources.json não encontrado!")
            print("Caminhos verificados:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Criando arquivo de exemplo...")
            json_path = create_example_sites_json()
    
    if output_dir is None:
        output_dir = "results"
    
    # Criar diretório de resultados
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 JSON encontrado: {json_path}")
    print(f"💾 Resultados serão salvos em: {output_dir}")
    
    # Carregar e exibir sites
    sites = load_all_sites(json_path)
    if not sites:
        print("❌ Nenhum site válido encontrado!")
        return None
    
    print(f"\n📋 {len(sites)} SITES CARREGADOS:")
    for i, site in enumerate(sites, 1):
        print(f"  {i}. {site.get('name', 'Site sem nome')} - {site.get('url', 'URL inválida')}")
    
    # Inicializar scraper com transformers
    print(f"\n🤖 Inicializando scraper com IA...")
    scraper = WebScraper()
    
    # =============================
    # ANÁLISE COM SALVAMENTO PROGRESSIVO
    # =============================
    print(f"\n🔍 INICIANDO ANÁLISE COM SALVAMENTO PROGRESSIVO...")
    
    results = []
    failed_sites = []
    
    # Arquivo de backup progressivo
    backup_file = os.path.join(output_dir, f"backup_progress_{timestamp}.json")
    
    for i, site in enumerate(sites, 1):
        print(f"\n{'='*50}")
        print(f"📊 PROCESSANDO SITE {i}/{len(sites)}")
        print(f"🌐 {site.get('name', 'Site desconhecido')}")
        
        try:
            # Analisar site individual
            result = scraper.analyze_single_site(site)
            
            if result:
                results.append(result)
                print(f"✅ Site {i} processado com sucesso!")
                
                # SALVAR IMEDIATAMENTE após cada site
                individual_file = os.path.join(output_dir, f"site_{i}_{timestamp}.json")
                with open(individual_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 Resultado individual salvo: {individual_file}")
                
                # BACKUP PROGRESSIVO a cada 2 sites
                if len(results) % 2 == 0 or i == len(sites):
                    backup_data = {
                        'analysis_id': analysis_id,
                        'timestamp': timestamp,
                        'processed_sites': len(results),
                        'total_sites': len(sites),
                        'results': results
                    }
                    
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(backup_data, f, ensure_ascii=False, indent=2)
                    print(f"🔄 Backup atualizado: {len(results)} sites processados")
                
            else:
                failed_sites.append(site)
                print(f"❌ Falha no site {i}")
                
        except Exception as e:
            print(f"❌ Erro crítico no site {i}: {e}")
            failed_sites.append(site)
            continue
    
    # =============================
    # RELATÓRIOS FINAIS
    # =============================
    
    if not results:
        print("\n❌ NENHUM SITE FOI ANALISADO COM SUCESSO!")
        return None
    
    print(f"\n🎯 PROCESSAMENTO CONCLUÍDO!")
    print(f"✅ {len(results)} sites analisados com sucesso")
    print(f"❌ {len(failed_sites)} sites falharam")
    
    # Salvar resultados detalhados FINAIS
    detailed_file = os.path.join(output_dir, f"analysis_detailed_{timestamp}.json")
    final_data = {
        'analysis_metadata': {
            'id': analysis_id,
            'timestamp': timestamp,
            'total_sites': len(sites),
            'successful_sites': len(results),
            'failed_sites': len(failed_sites)
        },
        'results': results,
        'failed_sites': failed_sites
    }
    
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    # Gerar e salvar relatório resumo
    report = generate_summary_report(results, failed_sites)
    report_file = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Gerar relatório consolidado
    consolidated_report = generate_consolidated_report(results, analysis_id)
    consolidated_file = os.path.join(output_dir, f"analysis_consolidated_{timestamp}.json")
    
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_report, f, ensure_ascii=False, indent=2)
    
    # Exibir resultados finais
    print("\n" + "="*60)
    print("🎯 ANÁLISE COMPLETA FINALIZADA!")
    print("="*60)
    print(report)
    print("\n📁 ARQUIVOS GERADOS:")
    print(f"  📊 Relatório detalhado: {detailed_file}")
    print(f"  📋 Relatório resumo: {report_file}")
    print(f"  📈 Relatório consolidado: {consolidated_file}")
    print(f"  🔄 Backup progressivo: {backup_file}")
    
    return {
        'analysis_id': analysis_id,
        'results': results,
        'failed_sites': failed_sites,
        'files': {
            'detailed': detailed_file,
            'report': report_file,
            'consolidated': consolidated_file,
            'backup': backup_file
        }
    }

def load_all_sites(json_path):
    """Carrega todos os sites do JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            sites = data.get('sites', [])
            
            # Validar sites
            valid_sites = []
            for site in sites:
                if isinstance(site, dict) and 'url' in site:
                    valid_sites.append(site)
                else:
                    print(f"⚠️ Site inválido ignorado: {site}")
            
            return valid_sites
            
    except Exception as e:
        print(f"❌ Erro ao carregar JSON: {e}")
        return []

def create_example_sites_json():
    """Cria arquivo JSON de exemplo com sites reais"""
    example_data = {
        "sites": [
            {
                "name": "Viagens Montreal - Porto de Galinhas",
                "url": "https://www.viagensmontreal.com/blog/quando-ir-a-porto-de-galinhas-descubra-a-melhor-epoca/",
                "category": "blog"
            },
            {
                "name": "Melhores Destinos - Porto de Galinhas", 
                "url": "https://www.melhoresdestinos.com.br/porto-de-galinhas.html",
                "category": "guia"
            },
            {
                "name": "Viagem e Turismo - Porto de Galinhas",
                "url": "https://viajeaqui.abril.com.br/cidades/br-pe-porto-de-galinhas",
                "category": "revista"
            },
            {
                "name": "TripAdvisor - Porto de Galinhas",
                "url": "https://www.tripadvisor.com.br/Tourism-g303404-Porto_de_Galinhas_Ipojuca_State_of_Pernambuco-Vacations.html",
                "category": "reviews"
            }
        ]
    }
    
    # Criar diretório data se não existir
    os.makedirs("../data", exist_ok=True)
    json_path = "../data/sites.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(example_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Arquivo de exemplo criado: {json_path}")
    return json_path

def generate_summary_report(results, failed_sites):
    """Gera relatório resumo"""
    if not results:
        return "Nenhum resultado para gerar relatório"
    
    report = []
    report.append("📊 RELATÓRIO RESUMO - ANÁLISE DE SITES")
    report.append("="*50)
    
    # Estatísticas gerais
    total_sites = len(results) + len(failed_sites)
    avg_sentiment = sum(r['sentiment_analysis']['overall_sentiment'] for r in results) / len(results)
    total_ai_classifications = sum(r['ai_usage']['contextual_classifications'] for r in results)
    total_fallbacks = sum(r['ai_usage']['keyword_fallbacks'] for r in results)
    
    report.append(f"\n🔢 ESTATÍSTICAS GERAIS:")
    report.append(f"  Sites processados: {total_sites}")
    report.append(f"  Sites analisados: {len(results)}")
    report.append(f"  Sites falharam: {len(failed_sites)}")
    report.append(f"  Taxa de sucesso: {(len(results)/total_sites)*100:.1f}%")
    report.append(f"  Sentimento médio: {avg_sentiment:.3f}")
    report.append(f"  Classificações com IA: {total_ai_classifications}")
    report.append(f"  Fallbacks por palavra-chave: {total_fallbacks}")
    if total_ai_classifications + total_fallbacks > 0:
        report.append(f"  Taxa de uso da IA: {(total_ai_classifications/(total_ai_classifications+total_fallbacks)*100):.1f}%")
    
    # Análise por site
    report.append(f"\n📋 ANÁLISE POR SITE:")
    for result in results:
        site_name = result['site_info']['name']
        sentiment = result['sentiment_analysis']['overall_sentiment']
        sentiment_label = "Positivo" if sentiment > 0.6 else "Neutro" if sentiment > 0.4 else "Negativo"
        
        report.append(f"\n  🌐 {site_name}")
        report.append(f"    Sentimento: {sentiment:.3f} ({sentiment_label})")
        report.append(f"    Palavras relevantes: {result['site_info']['relevant_words']}")
    
    # Sites que falharam
    if failed_sites:
        report.append(f"\n❌ SITES QUE FALHARAM:")
        for site in failed_sites:
            report.append(f"  - {site.get('name', 'Site sem nome')}")
    
    return "\n".join(report)

def generate_consolidated_report(results, analysis_id):
    """Gera relatório consolidado com insights"""
    if not results:
        return {}
    
    # Estatísticas gerais
    total_sites = len(results)
    sentiments = [r['sentiment_analysis']['overall_sentiment'] for r in results]
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    # Análise de categorias consolidada
    all_categories = {}
    for result in results:
        for category, terms in result['categorized_terms'].items():
            if category not in all_categories:
                all_categories[category] = []
            all_categories[category].extend(terms)
    
    # Top termos globais por categoria
    category_insights = {}
    for category, terms in all_categories.items():
        if terms:
            # Contar frequência total de cada termo
            term_frequency = {}
            for term in terms:
                term_text = term['term']
                if term_text not in term_frequency:
                    term_frequency[term_text] = {
                        'count': 0,
                        'total_score': 0
                    }
                term_frequency[term_text]['count'] += term['frequency']
                term_frequency[term_text]['total_score'] += term['relevance_score']
            
            # Top 10 termos da categoria
            top_terms = sorted(
                term_frequency.items(),
                key=lambda x: x[1]['total_score'],
                reverse=True
            )[:10]
            
            category_insights[category] = {
                'total_terms': len(terms),
                'unique_terms': len(term_frequency),
                'top_terms': [
                    {
                        'term': term,
                        'total_frequency': data['count'],
                        'total_score': round(data['total_score'], 2)
                    }
                    for term, data in top_terms
                ]
            }
    
    return {
        'analysis_metadata': {
            'id': analysis_id,
            'timestamp': datetime.now().isoformat(),
            'total_sites_analyzed': total_sites
        },
        'overall_insights': {
            'average_sentiment': round(avg_sentiment, 3),
            'sentiment_distribution': {
                'positive': len([s for s in sentiments if s > 0.6]),
                'neutral': len([s for s in sentiments if 0.4 <= s <= 0.6]),
                'negative': len([s for s in sentiments if s < 0.4])
            }
        },
        'category_insights': category_insights,
        'site_rankings': sorted(
            [
                {
                    'name': r['site_info']['name'],
                    'sentiment': r['sentiment_analysis']['overall_sentiment'],
                    'total_terms': len(r['top_terms'])
                }
                for r in results
            ],
            key=lambda x: x['sentiment'],
            reverse=True
        )
    }