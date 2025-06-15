import time
from init import run_complete_analysis

if __name__ == "__main__":
    print("🌟 PROJETO SAVINO - ANÁLISE AUTOMÁTICA DE SITES")
    print("="*60)
    
    # Marcar início
    start_time = time.time()
    
    # Executar análise completa
    result = run_complete_analysis()
    
    # Calcular tempo total
    end_time = time.time()
    execution_time = end_time - start_time
    
    if result:
        print(f"\n🎉 SUCESSO! Análise ID: {result['analysis_id']}")
        print(f"📊 {len(result['results'])} sites analisados")
        print(f"⏱️ Tempo total: {execution_time:.2f} segundos ({execution_time/60:.1f} minutos)")
        print("\n🔍 Para ver os resultados detalhados, abra os arquivos gerados!")
    else:
        print(f"\n❌ Falha na análise após {execution_time:.2f} segundos")