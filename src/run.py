from init import run_complete_analysis
from data_integration import get_latest_analysis_detailed, update_sites_csv_from_source, generate_powerbi_csvs_from_analysis, check_terms_sentiment_filled

if __name__ == "__main__":
    print("🌟 PROJETO SAVINO - ANÁLISE AUTOMÁTICA DE SITES")
    print("="*60)
    
    # Executar análise completa
    result = run_complete_analysis()
    
    if result:
        print(f"\n🎉 SUCESSO! Análise ID: {result['analysis_id']}")
        print(f"📊 {len(result['results'])} sites analisados")
        print("\n🔍 Para ver os resultados detalhados, abra os arquivos gerados!")
        # Integração extra: atualizar CSVs do Power BI
        try:
            update_sites_csv_from_source()
            latest_json = get_latest_analysis_detailed()
            generate_powerbi_csvs_from_analysis(latest_json)
            # Verificação automática do preenchimento da coluna term_sentiment
            check_terms_sentiment_filled()
        except Exception as e:
            print(f"⚠️ Erro ao atualizar arquivos para Power BI: {e}")
    else:
        print("\n❌ Falha na análise. Verifique os logs acima.")