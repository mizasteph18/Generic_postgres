import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration
CONFIG_FILE = "config.json"
PAGE_CONFIGS_DIR = "page_configs"
GENERATED_PAGES_DIR = "generated_pages"

# Configuration de la base de données
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'dashboard_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': os.getenv('DB_PORT', 5432)
}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

# ==================== CLASSES ====================

class DatabaseManager:
    """Gestionnaire de connexion PostgreSQL"""
    
    @staticmethod
    def get_connection():
        """Obtenir une connexion à PostgreSQL"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            logger.error(f"Erreur connexion DB: {e}")
            return None
    
    @staticmethod
    def execute_query(query, params=None):
        """Exécuter une requête SQL et retourner un DataFrame"""
        conn = DatabaseManager.get_connection()
        if conn:
            try:
                df = pd.read_sql(query, conn, params=params)
                conn.close()
                return df
            except Exception as e:
                logger.error(f"Erreur exécution requête: {e}")
                conn.close()
                return pd.DataFrame()
        return pd.DataFrame()

class QueryExecutor:
    """Exécute des requêtes basées sur la configuration"""
    
    @staticmethod
    def execute(query_config, params=None):
        """Exécuter une requête basée sur la configuration"""
        query_type = query_config.get('type', 'sql')
        
        try:
            if query_type == 'sql':
                return QueryExecutor._execute_sql(query_config, params)
            elif query_type == 'static':
                return QueryExecutor._execute_static(query_config, params)
            elif query_type == 'api':
                return QueryExecutor._execute_api(query_config, params)
            elif query_type == 'function':
                return QueryExecutor._execute_function(query_config, params)
            else:
                logger.warning(f"Type de requête non supporté: {query_type}")
                return QueryExecutor._get_sample_data(query_config)
        except Exception as e:
            logger.error(f"Exécution requête échouée: {e}")
            return QueryExecutor._get_sample_data(query_config)
    
    @staticmethod
    def _execute_sql(query_config, params):
        """Exécuter une requête SQL"""
        query = query_config.get('query', '')
        
        # Remplacer les paramètres dans la requête
        if params:
            for key, value in params.items():
                if isinstance(value, str):
                    query = query.replace(f'{{{key}}}', f"'{value}'")
                else:
                    query = query.replace(f'{{{key}}}', str(value))
        
        # Remplacer aussi les placeholders de style ${...}
        if params:
            for key, value in params.items():
                placeholder = f"${{{key}}}"
                if isinstance(value, str):
                    query = query.replace(placeholder, f"'{value}'")
                else:
                    query = query.replace(placeholder, str(value))
        
        logger.debug(f"Exécution SQL: {query[:200]}...")
        return DatabaseManager.execute_query(query)
    
    @staticmethod
    def _execute_static(query_config, params):
        """Exécuter une requête statique"""
        data = query_config.get('data', {})
        
        if 'columns' in data and 'rows' in data:
            return pd.DataFrame(data['rows'], columns=data['columns'])
        elif 'labels' in data and 'values' in data:
            return pd.DataFrame({
                'label': data['labels'],
                'value': data['values']
            })
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    
    @staticmethod
    def _execute_api(query_config, params):
        """Exécuter un appel API"""
        try:
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
                return pd.DataFrame(response.json())
            else:
                logger.error(f"Erreur API: {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erreur appel API: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _execute_function(query_config, params):
        """Exécuter une fonction Python"""
        try:
            module_name = query_config.get('module', 'data_functions')
            function_name = query_config.get('function')
            
            # Import dynamique
            module = __import__(module_name)
            function = getattr(module, function_name)
            
            result = function(**params) if params else function()
            
            # Convertir en DataFrame si possible
            if isinstance(result, dict):
                return pd.DataFrame([result])
            elif isinstance(result, list):
                return pd.DataFrame(result)
            else:
                return pd.DataFrame({'value': [result]})
        except Exception as e:
            logger.error(f"Erreur fonction: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def _get_sample_data(query_config):
        """Générer des données d'exemple"""
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
            return pd.DataFrame({
                'id': range(1, 11),
                'name': [f'Item {i}' for i in range(1, 11)],
                'value': [i * 10 for i in range(1, 11)]
            })

class VisualizationGenerator:
    """Génère des visualisations dynamiques à partir des données"""
    
    @staticmethod
    def generate_chart(data, chart_config):
        """Générer un graphique Plotly dynamique"""
        if data.empty:
            return json.dumps({
                'data': [],
                'layout': {
                    'title': 'No Data Available',
                    'plot_bgcolor': 'white',
                    'paper_bgcolor': 'white'
                }
            })
        
        chart_type = chart_config.get('type', 'line')
        
        try:
            # Obtenir les colonnes de configuration ou utiliser les premières colonnes
            x_column = chart_config.get('x_column', data.columns[0] if len(data.columns) > 0 else 'index')
            y_column = chart_config.get('y_column', data.columns[1] if len(data.columns) > 1 else data.columns[0])
            
            if x_column == 'index':
                x_data = list(range(len(data)))
            else:
                x_data = data[x_column].tolist()
            
            if chart_type == 'line':
                fig = go.Figure()
                
                # Si y_column est une liste, créer plusieurs lignes
                if isinstance(y_column, list):
                    for col in y_column:
                        if col in data.columns:
                            fig.add_trace(go.Scatter(
                                x=x_data,
                                y=data[col].tolist(),
                                mode='lines+markers',
                                name=col,
                                line=dict(width=3)
                            ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=data[y_column].tolist(),
                        mode='lines+markers',
                        name=y_column,
                        line=dict(width=3, color='#3498db')
                    ))
            
            elif chart_type == 'bar':
                fig = go.Figure()
                
                if isinstance(y_column, list):
                    for col in y_column:
                        if col in data.columns:
                            fig.add_trace(go.Bar(
                                x=x_data,
                                y=data[col].tolist(),
                                name=col
                            ))
                else:
                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=data[y_column].tolist(),
                        name=y_column,
                        marker_color='#2ecc71'
                    ))
            
            elif chart_type == 'pie':
                labels_col = chart_config.get('labels_column', data.columns[0])
                values_col = chart_config.get('values_column', data.columns[1] if len(data.columns) > 1 else data.columns[0])
                
                fig = go.Figure(data=[go.Pie(
                    labels=data[labels_col].tolist(),
                    values=data[values_col].tolist(),
                    hole=0.3,
                    marker=dict(colors=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
                )])
            
            elif chart_type == 'scatter':
                fig = go.Figure(data=go.Scatter(
                    x=data[x_column].tolist(),
                    y=data[y_column].tolist(),
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=data[y_column].tolist() if len(data.columns) > 2 else '#3498db',
                        colorscale='Viridis',
                        showscale=True
                    )
                ))
            
            else:
                # Graphique par défaut
                fig = go.Figure(data=[go.Scatter(
                    x=list(range(len(data))),
                    y=data.iloc[:, 0].tolist(),
                    mode='lines'
                )])
            
            # Appliquer la configuration du layout
            layout_config = chart_config.get('layout', {})
            
            fig.update_layout(
                title=layout_config.get('title', 'Chart'),
                xaxis_title=layout_config.get('xaxis_title', x_column),
                yaxis_title=layout_config.get('yaxis_title', y_column if isinstance(y_column, str) else 'Value'),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Arial', size=12),
                hovermode='closest',
                showlegend=True
            )
            
            # Ajouter des options de configuration supplémentaires
            if 'template' in layout_config:
                fig.update_layout(template=layout_config['template'])
            
            if 'height' in layout_config:
                fig.update_layout(height=layout_config['height'])
            
            if 'width' in layout_config:
                fig.update_layout(width=layout_config['width'])
            
            return fig.to_json()
            
        except Exception as e:
            logger.error(f"Erreur génération chart: {e}")
            return json.dumps({'error': str(e)})
    
    @staticmethod
    def generate_table(data, table_config):
        """Générer un tableau dynamique"""
        if data.empty:
            return {'columns': [], 'rows': [], 'total': 0}
        
        # Appliquer les transformations si spécifiées
        if 'transform' in table_config:
            data = VisualizationGenerator._apply_transforms(data, table_config['transform'])
        
        # Sélectionner les colonnes spécifiques si configuré
        columns = table_config.get('columns', data.columns.tolist())
        
        # Filtrer les colonnes disponibles
        available_columns = [col for col in columns if col in data.columns]
        
        if not available_columns:
            available_columns = data.columns.tolist()
        
        # Formater les valeurs si nécessaire
        rows = []
        for _, row in data.iterrows():
            formatted_row = []
            for col in available_columns:
                value = row[col]
                # Appliquer le formatage si spécifié
                if 'format' in table_config:
                    format_config = table_config['format']
                    if col in format_config:
                        if format_config[col] == 'currency':
                            value = f"${value:,.2f}"
                        elif format_config[col] == 'percentage':
                            value = f"{value}%"
                        elif format_config[col] == 'number':
                            value = f"{value:,.0f}"
                formatted_row.append(str(value))
            rows.append(formatted_row)
        
        return {
            'columns': available_columns,
            'rows': rows,
            'total': len(data)
        }
    
    @staticmethod
    def _apply_transforms(data, transforms):
        """Appliquer des transformations aux données"""
        result = data.copy()
        
        for transform in transforms:
            transform_type = transform.get('type')
            
            if transform_type == 'sort':
                columns = transform.get('columns', [])
                ascending = transform.get('ascending', True)
                if columns:
                    result = result.sort_values(by=columns, ascending=ascending)
            
            elif transform_type == 'filter':
                column = transform.get('column')
                operator = transform.get('operator', 'eq')
                value = transform.get('value')
                
                if column in result.columns:
                    if operator == 'eq':
                        result = result[result[column] == value]
                    elif operator == 'gt':
                        result = result[result[column] > value]
                    elif operator == 'lt':
                        result = result[result[column] < value]
                    elif operator == 'contains':
                        result = result[result[column].astype(str).str.contains(value, na=False)]
            
            elif transform_type == 'aggregate':
                group_by = transform.get('group_by', [])
                aggregations = transform.get('aggregations', {})
                
                if group_by and aggregations:
                    result = result.groupby(group_by).agg(aggregations).reset_index()
            
            elif transform_type == 'rename':
                rename_map = transform.get('mapping', {})
                result = result.rename(columns=rename_map)
        
        return result
    
    @staticmethod
    def generate_metric(data, metric_config):
        """Générer une métrique/KPI"""
        if data.empty:
            return {
                'value': 'N/A',
                'change': None,
                'trend': 'neutral',
                'status': 'unknown',
                'target': None
            }
        
        try:
            value_col = metric_config.get('value_column', data.columns[0])
            change_col = metric_config.get('change_column')
            target_col = metric_config.get('target_column')
            
            # Obtenir la valeur
            if value_col in data.columns:
                value = float(data.iloc[0][value_col])
            else:
                value = float(data.iloc[0, 0])
            
            # Calculer le changement
            change = None
            trend = 'neutral'
            if change_col and change_col in data.columns:
                change_val = float(data.iloc[0][change_col])
                change = f"{'+' if change_val > 0 else ''}{change_val:.1f}%"
                
                if change_val > 0:
                    trend = 'up'
                elif change_val < 0:
                    trend = 'down'
            
            # Vérifier par rapport à la cible
            status = 'neutral'
            target = None
            if target_col and target_col in data.columns:
                target = float(data.iloc[0][target_col])
                if value >= target:
                    status = 'good'
                elif value >= target * 0.8:
                    status = 'warning'
                else:
                    status = 'critical'
            
            # Formater la valeur
            format_type = metric_config.get('format', 'number')
            if format_type == 'currency':
                formatted_value = f"${value:,.2f}"
            elif format_type == 'percentage':
                formatted_value = f"{value:.1f}%"
            elif format_type == 'number':
                formatted_value = f"{value:,.0f}"
            else:
                formatted_value = str(value)
            
            return {
                'value': formatted_value,
                'raw_value': value,
                'change': change,
                'trend': trend,
                'status': status,
                'target': target
            }
            
        except Exception as e:
            logger.error(f"Erreur génération métrique: {e}")
            return {
                'value': 'N/A',
                'change': None,
                'trend': 'neutral',
                'status': 'error',
                'target': None
            }

# ==================== GESTION DES CONFIGURATIONS ====================

def load_config():
    """Charger la configuration globale"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erreur chargement config: {e}")
        return {}

def load_page_config(page_config_file):
    """Charger la configuration d'une page spécifique"""
    try:
        page_path = os.path.join(PAGE_CONFIGS_DIR, page_config_file)
        if os.path.exists(page_path):
            with open(page_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Erreur chargement page config {page_config_file}: {e}")
        return None

def save_generated_page(page_data):
    """Sauvegarder une page générée"""
    try:
        os.makedirs(GENERATED_PAGES_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        page_id = page_data.get('id', str(uuid.uuid4()))
        filename = f"{timestamp}_{page_id}.json"
        filepath = os.path.join(GENERATED_PAGES_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        return filename
    except Exception as e:
        logger.error(f"Erreur sauvegarde page: {e}")
        return None

def list_generated_pages():
    """Lister toutes les pages générées"""
    try:
        if not os.path.exists(GENERATED_PAGES_DIR):
            return []
        
        pages = []
        for filename in os.listdir(GENERATED_PAGES_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(GENERATED_PAGES_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        page_data = json.load(f)
                    
                    pages.append({
                        'id': page_data.get('id', filename.replace('.json', '')),
                        'filename': filename,
                        'name': page_data.get('name', page_data.get('title', 'Unnamed Page')),
                        'type': page_data.get('type', 'dashboard'),
                        'source_config': page_data.get('metadata', {}).get('config_type', 'unknown'),
                        'generated_at': page_data.get('generated_at', ''),
                        'components_count': len(page_data.get('components', []))
                    })
                except Exception as e:
                    logger.error(f"Erreur lecture page {filename}: {e}")
        
        pages.sort(key=lambda x: x.get('generated_at', ''), reverse=True)
        return pages
    except Exception as e:
        logger.error(f"Erreur listing pages: {e}")
        return []

def check_page_requirements(page_info, selections):
    """Vérifier si les sélections satisfont les prérequis de la page - VERSION AMÉLIORÉE"""
    try:
        requirements = page_info.get('requirements', {})
        
        # Pour chaque niveau requis
        for level, required_value in requirements.items():
            if level not in selections:
                return False
            
            level_selection = selections[level]
            
            # Si la sélection correspond à la valeur requise
            if level_selection != required_value:
                return False
        
        return True
    except Exception as e:
        logger.error(f"Erreur vérification prérequis: {e}")
        return False

def create_default_page_config(page_info):
    """Créer une configuration de page par défaut si le fichier n'existe pas"""
    report_name = page_info.get('metadata', {}).get('reportName', '')
    
    return {
        "type": "dashboard",
        "title": page_info.get('name', 'Default Dashboard'),
        "description": f"Generated dashboard for {report_name}",
        "visualizations": [
            {
                "type": "metric",
                "title": "Sample Performance Metric",
                "data": {
                    "type": "static",
                    "data": {
                        "value": 125.5,
                        "change": "+12.5",
                        "target": 120
                    }
                },
                "config": {
                    "value_column": "value",
                    "change_column": "change",
                    "target_column": "target",
                    "format": "number"
                }
            },
            {
                "type": "table",
                "title": "Sample Data Table",
                "data": {
                    "type": "static",
                    "data": {
                        "columns": ["Category", "Value", "Change"],
                        "rows": [
                            ["Category A", "100", "+5%"],
                            ["Category B", "150", "+10%"],
                            ["Category C", "75", "-2%"]
                        ]
                    }
                },
                "config": {
                    "format": {
                        "Value": "number",
                        "Change": "percentage"
                    }
                }
            }
        ]
    }

def process_visualization(viz_config, all_params):
    """Traiter une visualisation individuelle avec tous les paramètres"""
    viz_type = viz_config.get('type', 'table')
    
    # Récupérer les données
    data = pd.DataFrame()
    if 'data' in viz_config:
        data = QueryExecutor.execute(viz_config['data'], all_params)
    
    # Générer la visualisation
    result = {
        'id': str(uuid.uuid4()),
        'type': viz_type,
        'title': viz_config.get('title', 'Visualization'),
        'config': viz_config.get('config', {})
    }
    
    if not data.empty:
        if viz_type == 'chart':
            result['chart'] = VisualizationGenerator.generate_chart(data, viz_config.get('config', {}))
        elif viz_type == 'table':
            result['table'] = VisualizationGenerator.generate_table(data, viz_config.get('config', {}))
        elif viz_type == 'metric':
            result['metric'] = VisualizationGenerator.generate_metric(data, viz_config.get('config', {}))
        elif viz_type == 'custom':
            result['data'] = data.to_dict('records')
    else:
        # Données d'exemple si vide
        if viz_type == 'chart':
            sample_data = QueryExecutor._get_sample_data({'chart_type': 'bar'})
            result['chart'] = VisualizationGenerator.generate_chart(sample_data, viz_config.get('config', {}))
        elif viz_type == 'table':
            sample_data = QueryExecutor._get_sample_data({})
            result['table'] = VisualizationGenerator.generate_table(sample_data, viz_config.get('config', {}))
        elif viz_type == 'metric':
            result['metric'] = {
                'value': 'N/A',
                'change': None,
                'trend': 'neutral',
                'status': 'unknown',
                'target': None
            }
    
    return result

# ==================== ROUTES API ====================

@app.route('/')
def index():
    """Servir l'application front-end"""
    return send_from_directory('static', 'index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Obtenir la configuration globale - FORMAT COMPATIBLE AVEC HTML"""
    config = load_config()
    
    # Formater la configuration pour être compatible avec le frontend
    formatted_config = {}
    
    for key, value in config.items():
        formatted_config[key] = {
            "title": value.get("title", key),
            "description": value.get("description", ""),
            "form": value.get("form", {"type": "dropdown", "label": "Select Category:"}),
            "pageConfigs": value.get("pageConfigs", [])
        }
    
    return jsonify(formatted_config)

@app.route('/api/pages/<config_type>', methods=['GET'])
def get_pages_for_config(config_type):
    """Obtenir les pages disponibles pour un type de configuration"""
    config = load_config()
    
    if config_type not in config:
        return jsonify({"error": f"Configuration type '{config_type}' not found"}), 404
    
    pages = config[config_type].get('pageConfigs', [])
    return jsonify(pages)

@app.route('/api/generate', methods=['POST'])
def generate_pages():
    """Générer les pages sélectionnées avec données dynamiques - VERSION AMÉLIORÉE"""
    try:
        data = request.json
        config_type = data.get('configType')
        selections = data.get('selections', {})
        selected_page_ids = data.get('selectedPages', [])
        form_data = data.get('formData', {})
        
        if not config_type:
            return jsonify({'error': 'Type de configuration requis'}), 400
        
        config = load_config()
        if config_type not in config:
            return jsonify({'error': 'Type de configuration invalide'}), 400
        
        generated_pages = []
        
        # Obtenir toutes les configurations de pages pour ce type
        page_configs = config[config_type].get('pageConfigs', [])
        
        # Filtrer les pages sélectionnées
        selected_pages = [p for p in page_configs if p.get('id') in selected_page_ids]
        
        for page_info in selected_pages:
            page_id = page_info.get('id')
            
            # Vérifier les prérequis
            if not check_page_requirements(page_info, selections):
                logger.warning(f"Prérequis non satisfaits pour {page_id}")
                continue
            
            # Charger la configuration de la page
            page_config_file = page_info.get('pageConfig')
            page_config = load_page_config(page_config_file)
            
            if not page_config:
                logger.warning(f"Configuration page {page_config_file} non trouvée")
                # Créer une page de base si le fichier n'existe pas
                page_config = create_default_page_config(page_info)
            
            # Combiner tous les paramètres
            all_params = {**selections, **form_data}
            
            # Traiter les visualisations
            components = []
            if 'visualizations' in page_config:
                for viz_config in page_config['visualizations']:
                    component = process_visualization(viz_config, all_params)
                    components.append(component)
            
            # Créer la réponse de la page
            page_response = {
                'id': str(uuid.uuid4()),
                'name': page_info.get('name', page_config.get('title', 'Generated Page')),
                'type': page_config.get('type', 'dashboard'),
                'title': page_info.get('name', page_config.get('title', 'Generated Page')),
                'template': config_type,
                'config_id': page_id,
                'generated_at': datetime.now().isoformat(),
                'selections': selections,
                'form_data': form_data,
                'components': components,
                'metadata': {
                    'config_type': config_type,
                    'page_config': page_config_file,
                    'report_name': page_info.get('metadata', {}).get('reportName', ''),
                    'generation_time': datetime.now().isoformat(),
                    'description': page_info.get('metadata', {}).get('description', '')
                }
            }
            
            # Sauvegarder la page générée
            filename = save_generated_page(page_response)
            if filename:
                page_response['filename'] = filename
            
            generated_pages.append({
                'id': page_response['id'],
                'name': page_info.get('name', 'Unnamed Page'),
                'generated_page': page_response
            })
        
        return jsonify({
            'success': True,
            'generated_count': len(generated_pages),
            'pages': generated_pages
        })
        
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generated-pages', methods=['GET'])
def get_generated_pages():
    """Obtenir la liste de toutes les pages générées"""
    pages = list_generated_pages()
    return jsonify({'pages': pages})

@app.route('/api/generated-pages/<page_id>', methods=['GET'])
def get_generated_page(page_id):
    """Obtenir une page générée spécifique"""
    try:
        for filename in os.listdir(GENERATED_PAGES_DIR):
            if filename.endswith('.json') and page_id in filename:
                filepath = os.path.join(GENERATED_PAGES_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                return jsonify(page_data)
        
        # Chercher par nom aussi
        for filename in os.listdir(GENERATED_PAGES_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(GENERATED_PAGES_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    page_data = json.load(f)
                if page_data.get('id') == page_id or page_data.get('config_id') == page_id:
                    return jsonify(page_data)
        
        return jsonify({'error': 'Page non trouvée'}), 404
    except Exception as e:
        logger.error(f"Erreur obtention page {page_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de santé de l'API"""
    try:
        # Tester la connexion à la base de données
        conn = DatabaseManager.get_connection()
        db_status = 'connected' if conn else 'disconnected'
        if conn:
            conn.close()
        
        # Charger la configuration
        config = load_config()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '3.0.0',
            'database': db_status,
            'config_loaded': len(config) > 0,
            'config_categories': list(config.keys()),
            'page_configs_count': len(os.listdir(PAGE_CONFIGS_DIR)) if os.path.exists(PAGE_CONFIGS_DIR) else 0,
            'generated_pages_count': len(os.listdir(GENERATED_PAGES_DIR)) if os.path.exists(GENERATED_PAGES_DIR) else 0
        })
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/test-query', methods=['POST'])
def test_query():
    """Tester une requête SQL"""
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Requête vide'}), 400
        
        df = DatabaseManager.execute_query(query)
        
        return jsonify({
            'success': True,
            'row_count': len(df),
            'columns': df.columns.tolist(),
            'sample_data': df.head(10).to_dict('records'),
            'query_executed': query
        })
    except Exception as e:
        logger.error(f"Erreur test requête: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== INITIALISATION ====================

def initialize_directories():
    """Créer les répertoires nécessaires"""
    directories = [PAGE_CONFIGS_DIR, GENERATED_PAGES_DIR, 'static']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == '__main__':
    initialize_directories()
    
    print("=" * 70)
    print("🚀 JSON-Driven Dashboard Generator avec PostgreSQL")
    print("=" * 70)
    print(f"📁 Configuration: {CONFIG_FILE}")
    print(f"📄 Pages Configs: {PAGE_CONFIGS_DIR}/")
    print(f"💾 Pages Générées: {GENERATED_PAGES_DIR}/")
    print(f"🗄️  Base de données: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"🌐 Interface: http://localhost:8080")
    print(f"🔧 API Health: http://localhost:8080/api/health")
    print("=" * 70)
    print("📋 Endpoints API:")
    print("  • GET  /api/config              - Configuration globale")
    print("  • GET  /api/pages/<type>        - Pages par type de configuration")
    print("  • POST /api/generate            - Générer pages avec données DB")
    print("  • GET  /api/generated-pages     - Pages générées")
    print("  • GET  /api/generated-pages/<id> - Page générée spécifique")
    print("  • POST /api/test-query          - Tester une requête SQL")
    print("=" * 70)
    
    app.run(debug=True, port=8080, host='0.0.0.0')