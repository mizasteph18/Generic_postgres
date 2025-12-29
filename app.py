# app.py - Structure générique pour serveur JSON-driven
import os
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import io
import base64

# Configuration
CONFIG_DIR = "page_configs"
DATA_SOURCE = "postgresql"  # ou "mysql", "api", "csv", etc.

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# ============================================
# 1. GESTION DES CONFIGURATIONS
# ============================================
class ConfigurationManager:
    """Gère le chargement et la validation des configurations"""
    
    @staticmethod
    def load_config(config_type):
        """Charge une configuration par type"""
        config_path = f"configs/{config_type}.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def load_page_config(page_id):
        """Charge la configuration d'une page spécifique"""
        page_path = f"{CONFIG_DIR}/{page_id}"
        if os.path.exists(page_path):
            with open(page_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def save_page_config(page_id, config_data):
        """Sauvegarde une configuration de page"""
        os.makedirs(CONFIG_DIR, exist_ok=True)
        page_path = f"{CONFIG_DIR}/{page_id}"
        with open(page_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        return True

# ============================================
# 2. EXÉCUTEUR DE REQUÊTES GÉNÉRIQUE
# ============================================
class QueryExecutor:
    """Exécute des requêtes basées sur la configuration"""
    
    def __init__(self, data_source=DATA_SOURCE):
        self.data_source = data_source
        
    def execute(self, query_config, params=None):
        """Exécute une requête basée sur la configuration"""
        query_type = query_config.get('type', 'sql')
        
        if query_type == 'sql':
            return self._execute_sql(query_config, params)
        elif query_type == 'api':
            return self._execute_api(query_config, params)
        elif query_type == 'file':
            return self._execute_file(query_config, params)
        elif query_type == 'function':
            return self._execute_function(query_config, params)
        else:
            raise ValueError(f"Type de requête non supporté: {query_type}")
    
    def _execute_sql(self, query_config, params):
        """Exécute une requête SQL"""
        # Implémentation générique pour PostgreSQL/MySQL
        # À adapter selon votre configuration
        import psycopg2
        from sqlalchemy import create_engine
        
        query = query_config.get('query')
        db_config = query_config.get('connection', 'default')
        
        # Remplacer les paramètres dans la requête
        if params:
            for key, value in params.items():
                query = query.replace(f'{{{key}}}', str(value))
        
        # Exécution (exemple avec PostgreSQL)
        try:
            conn = psycopg2.connect(**self._get_db_config(db_config))
            df = pd.read_sql(query, conn)
            conn.close()
            return df
        except Exception as e:
            # Fallback: données simulées pour le développement
            return self._get_sample_data(query_config)
    
    def _execute_api(self, query_config, params):
        """Exécute un appel API"""
        import requests
        
        url = query_config.get('url')
        method = query_config.get('method', 'GET')
        headers = query_config.get('headers', {})
        
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    
    def _execute_file(self, query_config, params):
        """Charge des données depuis un fichier"""
        file_path = query_config.get('path')
        file_type = query_config.get('file_type', 'csv')
        
        if file_type == 'csv':
            return pd.read_csv(file_path)
        elif file_type == 'json':
            return pd.read_json(file_path)
        elif file_type == 'excel':
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Type de fichier non supporté: {file_type}")
    
    def _execute_function(self, query_config, params):
        """Exécute une fonction Python"""
        function_name = query_config.get('function')
        module_name = query_config.get('module', 'data_functions')
        
        # Import dynamique du module
        module = __import__(module_name)
        function = getattr(module, function_name)
        
        return function(**params) if params else function()
    
    def _get_db_config(self, config_name):
        """Récupère la configuration de base de données"""
        # À adapter avec vos configurations réelles
        configs = {
            'default': {
                'host': 'localhost',
                'database': 'app_db',
                'user': 'user',
                'password': 'password',
                'port': 5432
            }
        }
        return configs.get(config_name, configs['default'])
    
    def _get_sample_data(self, query_config):
        """Retourne des données d'exemple pour le développement"""
        # Génère des données de démo basées sur la configuration
        chart_type = query_config.get('chart_type', 'bar')
        
        if chart_type == 'line':
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            values = [100 + i * 2 + (i % 7) * 10 for i in range(30)]
            return pd.DataFrame({'date': dates, 'value': values})
        
        elif chart_type == 'bar':
            categories = ['A', 'B', 'C', 'D', 'E']
            values = [25, 40, 30, 35, 20]
            return pd.DataFrame({'category': categories, 'value': values})
        
        else:
            # Données tabulaires par défaut
            return pd.DataFrame({
                'id': range(1, 11),
                'name': [f'Item {i}' for i in range(1, 11)],
                'value': [i * 10 for i in range(1, 11)],
                'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
            })

# ============================================
# 3. GÉNÉRATEUR DE VISUALISATIONS
# ============================================
class VisualizationGenerator:
    """Génère des visualisations basées sur les données"""
    
    @staticmethod
    def generate_chart(data, chart_config):
        """Génère un graphique Plotly"""
        chart_type = chart_config.get('type', 'line')
        
        if isinstance(data, pd.DataFrame):
            if chart_type == 'line':
                fig = go.Figure()
                for column in data.columns[1:]:  # Première colonne = axe X
                    fig.add_trace(go.Scatter(
                        x=data.iloc[:, 0],
                        y=data[column],
                        mode='lines+markers',
                        name=column
                    ))
            
            elif chart_type == 'bar':
                fig = go.Figure()
                for i, column in enumerate(data.columns[1:]):
                    fig.add_trace(go.Bar(
                        x=data.iloc[:, 0],
                        y=data[column],
                        name=column
                    ))
            
            elif chart_type == 'pie':
                fig = go.Figure(data=[go.Pie(
                    labels=data.iloc[:, 0],
                    values=data.iloc[:, 1]
                )])
            
            elif chart_type == 'scatter':
                fig = go.Figure(data=go.Scatter(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    mode='markers'
                ))
            
            else:
                # Graphique par défaut
                fig = go.Figure(data=[go.Scatter(
                    x=list(range(len(data))),
                    y=data.iloc[:, 1] if len(data.columns) > 1 else data.iloc[:, 0],
                    mode='lines'
                )])
            
            # Appliquer la configuration du layout
            layout_config = chart_config.get('layout', {})
            fig.update_layout(
                title=layout_config.get('title', 'Chart'),
                xaxis_title=layout_config.get('xaxis_title', ''),
                yaxis_title=layout_config.get('yaxis_title', ''),
                template=layout_config.get('template', 'plotly_white')
            )
            
            return fig.to_json()
        
        return None
    
    @staticmethod
    def generate_table(data, table_config):
        """Formate les données pour un tableau"""
        if isinstance(data, pd.DataFrame):
            return {
                'columns': data.columns.tolist(),
                'rows': data.values.tolist(),
                'total': len(data)
            }
        elif isinstance(data, list):
            return {
                'columns': list(data[0].keys()) if data else [],
                'rows': [[row.get(col) for col in row.keys()] for row in data],
                'total': len(data)
            }
        else:
            return {
                'columns': [],
                'rows': [],
                'total': 0
            }
    
    @staticmethod
    def generate_metrics(data, metrics_config):
        """Génère des cartes de métriques"""
        if isinstance(data, pd.DataFrame) and not data.empty:
            return {
                'value': data.iloc[0, 0] if len(data.columns) > 0 else 0,
                'change': data.iloc[0, 1] if len(data.columns) > 1 else None,
                'trend': 'up' if (len(data.columns) > 1 and data.iloc[0, 1] > 0) else 'down'
            }
        return {'value': 0, 'change': None, 'trend': 'neutral'}

# ============================================
# 4. ROUTES API PRINCIPALES
# ============================================
@app.route('/')
def index():
    """Serveur de l'application front-end"""
    return send_from_directory('static', 'index.html')

@app.route('/api/configurations', methods=['GET'])
def get_configurations():
    """Récupère toutes les configurations disponibles"""
    config_types = ['analytics', 'userData', 'systemMetrics', 'salesData']
    configs = {}
    
    for config_type in config_types:
        config = ConfigurationManager.load_config(config_type)
        if config:
            configs[config_type] = config
    
    return jsonify(configs)

@app.route('/api/page-configs', methods=['GET'])
def get_page_configs():
    """Récupère les configurations de pages"""
    page_configs = {}
    
    # Charger toutes les configurations de pages
    if os.path.exists(CONFIG_DIR):
        for file in os.listdir(CONFIG_DIR):
            if file.endswith('.json'):
                with open(os.path.join(CONFIG_DIR, file), 'r') as f:
                    page_configs[file] = json.load(f)
    
    return jsonify(page_configs)

@app.route('/api/generate-page', methods=['POST'])
def generate_page():
    """Génère une page complète basée sur la configuration"""
    try:
        page_request = request.json
        
        # 1. Valider la configuration
        if not page_request.get('config'):
            return jsonify({'error': 'Configuration manquante'}), 400
        
        # 2. Extraire la configuration
        config = page_request['config']
        selections = page_request.get('selections', {})
        
        # 3. Initialiser les exécuteurs
        query_executor = QueryExecutor()
        viz_generator = VisualizationGenerator()
        
        # 4. Traiter chaque composant
        components = []
        
        if 'components' in config:
            for component_config in config['components']:
                component_result = process_component(
                    component_config, 
                    selections, 
                    query_executor, 
                    viz_generator
                )
                components.append(component_result)
        
        # 5. Créer la réponse
        page_id = str(uuid.uuid4())
        response = {
            'page_id': page_id,
            'title': config.get('title', 'Dashboard'),
            'timestamp': datetime.now().isoformat(),
            'components': components,
            'metadata': {
                'config_type': config.get('type'),
                'selections': selections
            }
        }
        
        # 6. Sauvegarder la page générée (optionnel)
        if page_request.get('save', True):
            save_path = f"generated_pages/{page_id}.json"
            os.makedirs('generated_pages', exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(response, f, indent=2)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_component(component_config, selections, query_executor, viz_generator):
    """Traite un composant individuel"""
    component_type = component_config.get('type', 'table')
    
    # Préparer les paramètres
    params = {}
    if 'params' in component_config:
        for param_name, param_value in component_config['params'].items():
            # Remplacer les sélections
            if isinstance(param_value, str) and param_value.startswith('${'):
                selection_key = param_value[2:-1]
                params[param_name] = selections.get(selection_key, param_value)
            else:
                params[param_name] = param_value
    
    # Récupérer les données
    data = None
    if 'data' in component_config:
        data = query_executor.execute(component_config['data'], params)
    
    # Générer la visualisation
    result = {
        'id': component_config.get('id', str(uuid.uuid4())),
        'type': component_type,
        'title': component_config.get('title', ''),
        'config': component_config
    }
    
    if data is not None:
        if component_type == 'chart':
            result['chart'] = viz_generator.generate_chart(data, component_config)
        elif component_type == 'table':
            result['table'] = viz_generator.generate_table(data, component_config)
        elif component_type == 'metric':
            result['metric'] = viz_generator.generate_metrics(data, component_config)
        elif component_type == 'custom':
            result['custom'] = data.to_dict('records') if isinstance(data, pd.DataFrame) else data
    
    return result

@app.route('/api/export/<format_type>', methods=['POST'])
def export_data(format_type):
    """Exporte des données dans différents formats"""
    try:
        data = request.json.get('data')
        
        if format_type == 'csv':
            df = pd.DataFrame(data)
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return jsonify({
                'content': output.getvalue(),
                'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            })
        
        elif format_type == 'json':
            return jsonify({
                'content': json.dumps(data, indent=2),
                'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            })
        
        elif format_type == 'excel':
            df = pd.DataFrame(data)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            output.seek(0)
            
            # Encoder en base64 pour le transfert
            encoded = base64.b64encode(output.getvalue()).decode('utf-8')
            
            return jsonify({
                'content': encoded,
                'filename': f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                'encoded': True
            })
        
        else:
            return jsonify({'error': f'Format non supporté: {format_type}'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# ============================================
# 5. ROUTES D'ADMINISTRATION
# ============================================
@app.route('/admin/configs', methods=['GET'])
def list_configs():
    """Liste toutes les configurations disponibles"""
    configs = []
    
    if os.path.exists('configs'):
        for file in os.listdir('configs'):
            if file.endswith('.json'):
                configs.append({
                    'name': file.replace('.json', ''),
                    'path': f'configs/{file}',
                    'size': os.path.getsize(f'configs/{file}')
                })
    
    return jsonify(configs)

@app.route('/admin/configs/<config_type>', methods=['PUT'])
def update_config(config_type):
    """Met à jour une configuration"""
    try:
        config_data = request.json
        
        config_path = f"configs/{config_type}.json"
        os.makedirs('configs', exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return jsonify({'success': True, 'message': f'Configuration {config_type} mise à jour'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# 6. STRUCTURE DE DOSSIERS
# ============================================
"""
L'application devrait avoir cette structure :

app/
├── app.py                    # Ce fichier
├── configs/                  # Configurations globales
│   ├── analytics.json
│   ├── userData.json
│   └── ...
├── page_configs/            # Configurations de pages
│   ├── analytics_performance_summary.json
│   └── ...
├── data_functions.py        # Fonctions Python personnalisées
├── generated_pages/         # Pages générées (optionnel)
├── static/                  # Front-end
│   └── index.html          # Version modifiée du front-end
├── templates/               # Templates HTML (optionnel)
└── requirements.txt
"""

# ============================================
# EXEMPLE DE CONFIGURATION (analytics.json)
# ============================================
"""
{
  "type": "analytics",
  "title": "Performance Analysis",
  "description": "Configure performance analysis settings",
  "fontSize": "15px",
  "levels": {
    "level1": {
      "title": "Analysis Type",
      "type": "dropdown",
      "id": "analysisType",
      "label": "Select analysis type",
      "values": ["Web Performance", "User Engagement"],
      "mandatory": true
    }
  },
  "components": [
    {
      "type": "chart",
      "title": "Performance Chart",
      "data": {
        "type": "sql",
        "query": "SELECT date, value FROM metrics WHERE type = '${analysisType}'",
        "connection": "default"
      },
      "chart_type": "line",
      "layout": {
        "title": "Performance Over Time",
        "xaxis_title": "Date",
        "yaxis_title": "Value"
      }
    }
  ]
}
"""

if __name__ == '__main__':
    # Créer les répertoires nécessaires
    os.makedirs('configs', exist_ok=True)
    os.makedirs('page_configs', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🚀 Serveur JSON-Driven démarré !")
    print("📊 Accédez à l'interface: http://localhost:5000")
    print("🔧 API Health: http://localhost:5000/api/health")
    print("📋 Configurations: http://localhost:5000/api/configurations")
    
    app.run(debug=True, port=5000)