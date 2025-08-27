# -*- coding: utf-8 -*-
"""
ROTEIRO APRESENTAÇÃO DEEP LEARNING - League of Legends
Projeto: Predição de Vitória do Time Azul (blueWins) - Versão Melhorada e Simplificada

Este script implementa um projeto educacional de Deep Learning seguindo critérios específicos:
- Estrutura clara e modular para fins didáticos
- Comparação entre modelo baseline e modelo complexo
- Análise completa dos resultados com visualizações
- Código simplificado e bem documentado

Melhorias implementadas:
1. Código mais limpo e modular
2. Documentação aprimorada
3. Redução de redundâncias
4. Foco nas métricas mais importantes
5. Visualizações mais claras
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, classification_report
)
from sklearn.feature_selection import mutual_info_classif

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
##BASE_DIR = "/content/"
DATA_FILE = os.path.join("high_diamond_ranked_10min.csv")
OUTPUT_DIR = os.path.join("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure matplotlib para melhor qualidade
plt.style.use('default')
sns.set_palette("husl")

# Membros do Grupo (preencher conforme necessário)
MEMBERS = ["Membro 1", "Membro 2", "Membro 3"]

# ============================================================================
# CLASSES E FUNÇÕES UTILITÁRIAS
# ============================================================================

class LeagueMLPresentation:
    """
    Classe principal para organizar a apresentação de Machine Learning
    do projeto League of Legends de forma didática e estruturada.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_cols = None
        self.results = {}

    def print_section(self, title: str, content: str = ""):
        """Imprime seção formatada"""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        if content:
            print(content)

    def load_and_describe_data(self):
        """
        SEÇÃO 1: Problema de Negócio e Base de Dados
        """
        self.print_section("1. MEMBROS DO GRUPO E PROBLEMA DE NEGÓCIO")

        # Carregar dados
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset não encontrado: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        n_rows, n_cols = self.df.shape

        # Informações do projeto
        info = f"""
Membros do Grupo:
{chr(10).join([f"  • {member}" for member in MEMBERS])}

Problema de Negócio:
  • Tipo: CLASSIFICAÇÃO BINÁRIA
  • Objetivo: Predizer se o time Azul vencerá a partida
  • Variável alvo: blueWins (0=derrota, 1=vitória)

Base de Dados:
  • Tamanho da amostra: {n_rows:,} partidas
  • Número de features: {n_cols-1} (excluindo target)
  • Dataset: League of Legends - Ranked Games (10 min)
  • Fonte: Dados de partidas ranqueadas em alto elo
        """
        print(info)

        # Identificar features numéricas, excluindo variáveis do time red
        self.feature_cols = [col for col in self.df.columns
                            if col not in ['gameId', 'blueWins'] and
                            'red' not in col.lower() and
                            pd.api.types.is_numeric_dtype(self.df[col])]

        print(f"Features selecionadas para análise: {len(self.feature_cols)} variáveis")
        return self.df

    def exploratory_analysis(self):
        """
        SEÇÃO 2: Análise Exploratória
        """
        self.print_section("2. ANÁLISE EXPLORATÓRIA")

        target = 'blueWins'

        # 2.1 Distribuição da variável alvo
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        counts = self.df[target].value_counts()
        plt.pie(counts.values, labels=['Derrota (0)', 'Vitória (1)'],
               autopct='%1.1f%%', startangle=90)
        plt.title('Distribuição da Variável Alvo')

        plt.subplot(1, 2, 2)
        sns.countplot(data=self.df, x=target)
        plt.title('Contagem por Classe')
        plt.xlabel('Blue Team Wins')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'distribuicao_target.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2.2 Análise de relevância das features
        X = self.df[self.feature_cols].fillna(self.df[self.feature_cols].median())
        y = self.df[target].values

        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)

        # Top 10 features mais importantes
        top_features = feature_importance.head(10)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title('Top 10 Features por Mutual Information')
        plt.xlabel('Importância (Mutual Information)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Principais insights da análise exploratória:")
        print(f"  • Classes balanceadas: {counts[0]} derrotas vs {counts[1]} vitórias")
        print(f"  • Feature mais relevante: {top_features.iloc[0]['feature']}")
        print(f"  • Foco na análise: goldDiff e expDiff como variáveis primordiais")
        print(f"  • Variáveis do time red foram eliminadas da análise")

        # Priorizar goldDiff e expDiff como variáveis primordiais
        primary_features = ['blueGoldDiff', 'blueExperienceDiff']

        # Selecionar top 3 outras features por importância (excluindo as primordiais)
        other_features = feature_importance[~feature_importance['feature'].isin(primary_features)]
        top_3_others = other_features.head(3)['feature'].values

        # Combinar features primordiais com as top 3 por importância
        self.selected_features = np.concatenate([primary_features, top_3_others])

        print(f"\nVariáveis primordiais selecionadas: {', '.join(primary_features)}")
        print(f"Top 3 outras variáveis por importância: {', '.join(top_3_others)}")
        print(f"Total de features para análise: {len(self.selected_features)}")
        return feature_importance

    def prepare_data(self):
        """
        SEÇÃO 3: Preparação dos Dados
        """
        self.print_section("3. PREPARAÇÃO DOS DADOS")

        # Usar features selecionadas por relevância
        X = self.df[self.selected_features].copy()
        y = self.df['blueWins'].copy()

        # Tratamento de valores ausentes (se houver)
        X = X.fillna(X.median())

        # Split estratificado
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Dados de treino: {self.X_train.shape[0]} amostras")
        print(f"Dados de teste: {self.X_test.shape[0]} amostras")
        print(f"Features utilizadas: {len(self.selected_features)}")

    def build_and_train_models(self):
        """
        SEÇÃO 4: Treinamento dos Modelos
        """
        self.print_section("4. TREINAMENTO DOS MODELOS")

        # Modelo Baseline: Regressão Logística
        baseline_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])

        # Modelo Complexo: MLP (Multi-Layer Perceptron)
        mlp_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(random_state=42, max_iter=500))
        ])

        # Hiperparâmetros para otimização do MLP
        param_grid = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate_init': [0.001, 0.01]
        }

        print("Estratégia de otimização:")
        print("  • Método: GridSearchCV com 5-fold cross-validation")
        print("  • Métrica de seleção: F1-score")
        print("  • Topologia testada: 1-2 camadas ocultas")
        print("  • Referência: MLPClassifier do scikit-learn com ReLU")

        # Grid Search para MLP
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            mlp_model, param_grid, cv=cv, scoring='f1',
            n_jobs=-1, verbose=0
        )

        # Treinamento
        print("\nTreinando modelos...")
        baseline_model.fit(self.X_train, self.y_train)
        grid_search.fit(self.X_train, self.y_train)

        # Melhor modelo MLP
        best_mlp = grid_search.best_estimator_

        print(f"Melhor configuração MLP: {grid_search.best_params_}")
        print(f"Melhor score CV: {grid_search.best_score_:.4f}")

        self.models = {
            'baseline': baseline_model,
            'mlp': best_mlp,
            'mlp_params': grid_search.best_params_
        }

        return self.models

    def evaluate_models(self):
        """
        SEÇÃO 5: Validação e Comparação
        """
        self.print_section("5. VALIDAÇÃO E COMPARAÇÃO DOS MODELOS")

        results = {}

        for name, model in [('Baseline (LogReg)', self.models['baseline']),
                           ('MLP Otimizado', self.models['mlp'])]:

            # Predições
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # Métricas
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_prob)
            }

            results[name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_prob
            }

        # Comparação das métricas
        comparison_df = pd.DataFrame({
            model: data['metrics']
            for model, data in results.items()
        }).round(4)

        print("COMPARAÇÃO DE PERFORMANCE:")
        print(comparison_df.to_string())

        # Determinar melhor modelo
        baseline_f1 = results['Baseline (LogReg)']['metrics']['f1']
        mlp_f1 = results['MLP Otimizado']['metrics']['f1']

        if mlp_f1 > baseline_f1:
            best_model_name = 'MLP Otimizado'
            improvement = ((mlp_f1 - baseline_f1) / baseline_f1) * 100
        else:
            best_model_name = 'Baseline (LogReg)'
            improvement = ((baseline_f1 - mlp_f1) / mlp_f1) * 100

        print(f"\nMELHOR MODELO: {best_model_name}")
        print(f"Melhoria de performance: {improvement:.2f}%")

        self.results = results
        return results, comparison_df

    def generate_visualizations(self):
        """
        SEÇÃO 6: Gráficos para Análise de Resultados
        """
        self.print_section("6. VISUALIZAÇÕES DOS RESULTADOS")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        models = [('Baseline', 'Baseline (LogReg)'), ('MLP', 'MLP Otimizado')]

        for i, (short_name, full_name) in enumerate(models):
            y_pred = self.results[full_name]['predictions']
            y_prob = self.results[full_name]['probabilities']

            # Matriz de Confusão
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, 0])
            axes[i, 0].set_title(f'Matriz de Confusão - {short_name}')
            axes[i, 0].set_ylabel('Real')
            axes[i, 0].set_xlabel('Predito')

            # Curva ROC
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc = self.results[full_name]['metrics']['roc_auc']
            axes[i, 1].plot(fpr, tpr, label=f'{short_name} (AUC = {auc:.3f})')
            axes[i, 1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
            axes[i, 1].set_xlabel('Taxa de Falsos Positivos')
            axes[i, 1].set_ylabel('Taxa de Verdadeiros Positivos')
            axes[i, 1].set_title(f'Curva ROC - {short_name}')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

            # Distribuição de Probabilidades
            axes[i, 2].hist(y_prob[self.y_test == 0], bins=30, alpha=0.7,
                           label='Classe 0', density=True)
            axes[i, 2].hist(y_prob[self.y_test == 1], bins=30, alpha=0.7,
                           label='Classe 1', density=True)
            axes[i, 2].set_xlabel('Probabilidade Predita')
            axes[i, 2].set_ylabel('Densidade')
            axes[i, 2].set_title(f'Distribuição de Probabilidades - {short_name}')
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'analise_completa_modelos.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Gráfico de convergência para MLP (se disponível)
        mlp_classifier = self.models['mlp'].named_steps['classifier']
        if hasattr(mlp_classifier, 'loss_curve_'):
            plt.figure(figsize=(10, 6))
            plt.plot(mlp_classifier.loss_curve_, linewidth=2)
            plt.title('Convergência do MLP - Curva de Loss')
            plt.xlabel('Épocas')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(OUTPUT_DIR, 'convergencia_mlp.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        print("Gráficos salvos:")
        print("  • analise_completa_modelos.png - Comparação geral")
        print("  • convergencia_mlp.png - Convergência do treinamento")

    def analyze_errors(self):
        """
        SEÇÃO 7: Análise de Erros
        """
        self.print_section("7. ANÁLISE DE ERROS")

        # Analisar erros do melhor modelo
        best_model_name = max(self.results.keys(),
                            key=lambda x: self.results[x]['metrics']['f1'])

        y_pred = self.results[best_model_name]['predictions']
        y_prob = self.results[best_model_name]['probabilities']

        # Identificar erros
        errors = self.y_test != y_pred
        error_indices = np.where(errors)[0]

        # Erros por confiança
        confidence = np.abs(y_prob - 0.5) * 2  # Converter para 0-1
        high_confidence_errors = error_indices[confidence[error_indices] > 0.8]

        print(f"Análise de erros do modelo: {best_model_name}")
        print(f"Total de erros: {sum(errors)} ({sum(errors)/len(self.y_test)*100:.1f}%)")
        print(f"Erros de alta confiança: {len(high_confidence_errors)}")

        # Mostrar exemplos de erros
        if len(high_confidence_errors) > 0:
            print("\nExemplos de erros de alta confiança:")
            for i, idx in enumerate(high_confidence_errors[:3]):
                real_idx = self.X_test.index[idx]
                print(f"  Exemplo {i+1}:")
                print(f"    Real: {self.y_test.iloc[idx]}, Predito: {y_pred[idx]}")
                print(f"    Confiança: {confidence[idx]:.3f}")
                print(f"    Probabilidade: {y_prob[idx]:.3f}")

        return len(high_confidence_errors), sum(errors)

    def conclusions_and_next_steps(self):
        """
        SEÇÃO 8: Conclusões e Próximos Passos
        """
        self.print_section("8. CONCLUSÕES E PRÓXIMOS PASSOS")

        # Resumo dos resultados
        baseline_f1 = self.results['Baseline (LogReg)']['metrics']['f1']
        mlp_f1 = self.results['MLP Otimizado']['metrics']['f1']

        conclusions = f"""
PRINCIPAIS DESCOBERTAS:

1. Performance dos Modelos:
   • Baseline (Regressão Logística): F1 = {baseline_f1:.4f}
   • MLP Otimizado: F1 = {mlp_f1:.4f}
   • Diferença: {abs(mlp_f1 - baseline_f1):.4f}

2. Insights sobre o Problema:
   • Dataset balanceado facilita o aprendizado
   • Features de gold e experiência são altamente preditivas
   • Diferenças entre times (goldDiff, expDiff) são cruciais

3. Modelo Recomendado:
   • {'MLP' if mlp_f1 > baseline_f1 else 'Baseline'} apresentou melhor performance
   • ROC-AUC indica boa capacidade discriminativa

PRÓXIMOS PASSOS:

1. Melhorias no Modelo:
   • Testar arquiteturas mais profundas (3+ camadas)
   • Implementar regularização avançada (Dropout)
   • Experimentar outros algoritmos (Random Forest, XGBoost)

2. Engenharia de Features:
   • Criar ratios entre estatísticas dos times
   • Features de interação temporal
   • Normalização por posição/role dos jogadores

3. Deployment:
   • Implementar API REST com Flask/FastAPI
   • Sistema de monitoramento de performance
   • Interface web para predições em tempo real
        """

        print(conclusions)

        # Salvar resumo executivo
        summary = {
            'project_info': {
                'members': MEMBERS,
                'problem_type': 'Binary Classification',
                'target': 'blueWins',
                'dataset_size': len(self.df)
            },
            'best_model': max(self.results.keys(),
                            key=lambda x: self.results[x]['metrics']['f1']),
            'performance_summary': {
                model: data['metrics']
                for model, data in self.results.items()
            },
            'mlp_best_params': self.models['mlp_params']
        }

        with open(os.path.join(OUTPUT_DIR, 'resumo_executivo.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nResumo executivo salvo em: {os.path.join(OUTPUT_DIR, 'resumo_executivo.json')}")

    def run_complete_analysis(self):
        """
        Executa análise completa seguindo o roteiro da apresentação
        """
        print("🚀 INICIANDO ANÁLISE COMPLETA - LEAGUE OF LEGENDS ML")
        print("="*60)

        try:
            # Executar todas as etapas
            self.load_and_describe_data()
            self.exploratory_analysis()
            self.prepare_data()
            self.build_and_train_models()
            self.evaluate_models()
            self.generate_visualizations()
            self.analyze_errors()
            self.conclusions_and_next_steps()

            dump(self.models['mlp'], os.path.join(OUTPUT_DIR, 'melhor_modelo.joblib'))
            print("✅ Modelo salvo com sucesso para deployment!")

            self.print_section("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
            print(f"Todos os arquivos foram salvos em: {OUTPUT_DIR}")


        except Exception as e:
            print(f"❌ Erro durante a execução: {str(e)}")
            raise

# ============================================================================
# DEPLOYMENT (OPCIONAL)
# ============================================================================

def create_simple_deployment_example():
    """
    Cria um exemplo simples de código de deployment com Flask,
    e salva como um arquivo Python.
    """
    deployment_code = '''\
from joblib import load
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar modelo treinado
model = load('melhor_modelo.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predições"""
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(probability),
            'team_win_probability': {
                'blue': float(model.predict_proba(features)[0][1]),
                'red': float(model.predict_proba(features)[0][0])
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

    filepath = os.path.join(OUTPUT_DIR, 'exemplo_deployment.py')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(deployment_code)

    print(f"Exemplo de deployment salvo em: {filepath}")

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================


def main():
    """Função principal para executar toda a apresentação"""

    # Inicializar e executar análise
    presentation = LeagueMLPresentation(DATA_FILE)
    presentation.run_complete_analysis()

    # Criar exemplo de deployment
    print("\n" + "="*60)
    print("DEPLOYMENT (OPCIONAL)")
    print("="*60)
    create_simple_deployment_example()
    print("Exemplo básico de deployment criado!")

if __name__ == "__main__":
    main()