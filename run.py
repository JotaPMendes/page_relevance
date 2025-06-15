from init import run_complete_analysis

if __name__ == "__main__":
    print("🌟 PROJETO SAVINO - ANÁLISE AUTOMÁTICA DE SITES")
    print("="*60)
    
    # Executar análise completa
    result = run_complete_analysis()
    
    if result:
        print(f"\n🎉 SUCESSO! Análise ID: {result['analysis_id']}")
        print(f"📊 {len(result['results'])} sites analisados")
        print("\n🔍 Para ver os resultados detalhados, abra os arquivos gerados!")
    else:
        print("\n❌ Falha na análise. Verifique os logs acima.")