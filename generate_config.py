# generate_config_verified.py
import pandas as pd
import json
import re
import os
from typing import Dict, List, Any, Optional

class ConfigGenerator:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.data = {}
        self.errors = []
        self.warnings = []
        self.config = {}
        
    def load_excel(self) -> bool:
        """Charge l'Excel et vérifie les feuilles"""
        print("📊 Chargement du fichier Excel...")
        
        if not os.path.exists(self.excel_path):
            self.errors.append(f"Fichier Excel non trouvé: {self.excel_path}")
            return False
        
        try:
            xls = pd.ExcelFile(self.excel_path)
            required_sheets = ['Level1', 'Level2', 'Level3', 'Inputs']
            
            # Vérifier les feuilles requises
            missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
            if missing_sheets:
                self.errors.append(f"Feuilles manquantes: {', '.join(missing_sheets)}")
                return False
            
            # Charger chaque feuille
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = df.where(pd.notnull(df), None)
                self.data[sheet_name] = df
            
            print(f"✅ Excel chargé: {len(self.data)} feuilles")
            return True
            
        except Exception as e:
            self.errors.append(f"Erreur chargement Excel: {str(e)}")
            return False
    
    def validate_level1(self) -> bool:
        """Valide la feuille Level1"""
        print("🔍 Validation Level1...")
        df = self.data['Level1']
        valid = True
        
        # Vérifier colonnes requises
        required_cols = ['Id', 'Title', 'Tag', 'value', 'Description', 'Form', 'label', 'Action']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.errors.append(f"Level1 - Colonnes manquantes: {', '.join(missing_cols)}")
            valid = False
        
        # Vérifier chaque ligne
        for idx, row in df.iterrows():
            row_num = idx + 2  # +2 car Excel commence à 1 + header
            
            # Vérifier Id unique
            if not row['Id']:
                self.errors.append(f"Level1 Ligne {row_num}: Id manquant")
                valid = False
            
            # Vérifier value unique
            if not row['value']:
                self.errors.append(f"Level1 Ligne {row_num}: value manquante")
                valid = False
            
            # Vérifier Action
            if not row['Action']:
                self.errors.append(f"Level1 Ligne {row_num}: Action manquante")
                valid = False
            else:
                action = str(row['Action'])
                if '.json' in action and not row['InputId']:
                    self.warnings.append(f"Level1 Ligne {row_num}: Action .json mais InputId vide")
                elif 'NextLevel' in action and row['InputId']:
                    self.warnings.append(f"Level1 Ligne {row_num}: Action NextLevel mais InputId rempli")
        
        # Vérifier valeurs uniques
        if 'value' in df.columns:
            duplicates = df[df['value'].duplicated()]['value'].tolist()
            if duplicates:
                self.errors.append(f"Level1 - Valeurs dupliquées: {', '.join(map(str, duplicates))}")
                valid = False
        
        if valid:
            print(f"✅ Level1 validé: {len(df)} lignes")
        return valid
    
    def validate_level2(self) -> bool:
        """Valide la feuille Level2"""
        print("🔍 Validation Level2...")
        df = self.data['Level2']
        valid = True
        
        # Vérifier colonnes requises
        required_cols = ['PreviousLevel', 'Id', 'Title', 'Tag', 'value', 'Description', 'Form', 'label', 'Action']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.errors.append(f"Level2 - Colonnes manquantes: {', '.join(missing_cols)}")
            valid = False
        
        # Vérifier chaque ligne
        for idx, row in df.iterrows():
            row_num = idx + 2
            
            # Vérifier PreviousLevel existe dans Level1
            if row['PreviousLevel']:
                level1_ids = self.data['Level1']['Id'].tolist()
                if row['PreviousLevel'] not in level1_ids:
                    self.errors.append(f"Level2 Ligne {row_num}: PreviousLevel {row['PreviousLevel']} non trouvé dans Level1")
                    valid = False
            
            # Vérifier Action
            if not row['Action']:
                self.errors.append(f"Level2 Ligne {row_num}: Action manquante")
                valid = False
            else:
                action = str(row['Action'])
                if '.json' in action:
                    # Vérifier InputId si Action contient .json
                    if not row['InputId']:
                        self.errors.append(f"Level2 Ligne {row_num}: Action .json mais InputId vide")
                        valid = False
                    
                    # Vérifier le format du fichier
                    files = [f.strip() for f in action.split(',') if '.json' in f]
                    for file in files:
                        if not file.endswith('.json'):
                            self.errors.append(f"Level2 Ligne {row_num}: Fichier invalide '{file}' - doit finir par .json")
                            valid = False
        
        if valid:
            print(f"✅ Level2 validé: {len(df)} lignes")
        return valid
    
    def validate_level3(self) -> bool:
        """Valide la feuille Level3"""
        print("🔍 Validation Level3...")
        df = self.data['Level3']
        valid = True
        
        # Vérifier colonnes requises
        required_cols = ['PreviousLevel', 'Id', 'Title', 'Tag', 'value', 'Description', 'Form', 'label', 'Action']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.errors.append(f"Level3 - Colonnes manquantes: {', '.join(missing_cols)}")
            valid = False
        
        # Vérifier chaque ligne
        for idx, row in df.iterrows():
            row_num = idx + 2
            
            # Vérifier PreviousLevel existe dans Level2
            if row['PreviousLevel']:
                level2_ids = self.data['Level2']['Id'].tolist()
                if row['PreviousLevel'] not in level2_ids:
                    self.errors.append(f"Level3 Ligne {row_num}: PreviousLevel {row['PreviousLevel']} non trouvé dans Level2")
                    valid = False
            
            # Vérifier Action et InputId
            if row['Action']:
                action = str(row['Action'])
                if '.json' in action:
                    # RÈGLE CRITIQUE: InputId DOIT être rempli quand Action contient .json
                    if not row['InputId']:
                        self.errors.append(f"Level3 Ligne {row_num}: Action .json mais InputId VIDE - DOIT être rempli!")
                        valid = False
                    
                    # Vérifier InputId existe dans Inputs
                    if row['InputId']:
                        input_ids = self.data['Inputs']['InputId'].tolist()
                        if row['InputId'] not in input_ids:
                            self.errors.append(f"Level3 Ligne {row_num}: InputId '{row['InputId']}' non trouvé dans Inputs")
                            valid = False
        
        if valid:
            print(f"✅ Level3 validé: {len(df)} lignes")
        return valid
    
    def validate_inputs(self) -> bool:
        """Valide la feuille Inputs"""
        print("🔍 Validation Inputs...")
        df = self.data['Inputs']
        valid = True
        
        # Vérifier colonnes requises
        required_cols = ['InputId', 'FieldOrder', 'FieldName', 'FieldType', 'FieldLabel', 'Required']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.errors.append(f"Inputs - Colonnes manquantes: {', '.join(missing_cols)}")
            valid = False
        
        # Vérifier chaque ligne
        for idx, row in df.iterrows():
            row_num = idx + 2
            
            # Vérifier InputId
            if not row['InputId']:
                self.errors.append(f"Inputs Ligne {row_num}: InputId manquant")
                valid = False
            
            # Vérifier FieldOrder numérique
            if row['FieldOrder'] is not None:
                try:
                    int(row['FieldOrder'])
                except (ValueError, TypeError):
                    self.errors.append(f"Inputs Ligne {row_num}: FieldOrder invalide '{row['FieldOrder']}' - doit être numérique")
                    valid = False
            
            # Vérifier FieldType valide
            valid_types = ['dropdown', 'checklist', 'text', 'number', 'date', 'range']
            if row['FieldType'] and row['FieldType'] not in valid_types:
                self.warnings.append(f"Inputs Ligne {row_num}: FieldType '{row['FieldType']}' non standard")
            
            # Vérifier Tag booléen
            if 'Tag' in df.columns and row['Tag'] is not None:
                tag_val = str(row['Tag']).lower()
                if tag_val not in ['true', 'false', 'vrai', 'faux', '1', '0', '']:
                    self.warnings.append(f"Inputs Ligne {row_num}: Tag '{row['Tag']}' devrait être true/false")
        
        # Vérifier les InputId référencés existent dans les autres feuilles
        referenced_input_ids = []
        for sheet in ['Level1', 'Level2', 'Level3']:
            if sheet in self.data and 'InputId' in self.data[sheet].columns:
                referenced_input_ids.extend([id for id in self.data[sheet]['InputId'].tolist() if id])
        
        unique_referenced = set(referenced_input_ids)
        existing_input_ids = set(df['InputId'].tolist())
        
        # InputIds référencés mais non définis
        missing_in_inputs = unique_referenced - existing_input_ids
        if missing_in_inputs:
            self.errors.append(f"Inputs - InputId référencés mais non définis: {', '.join(missing_inferenced)}")
            valid = False
        
        # InputIds définis mais jamais référencés
        unused_inputs = existing_input_ids - unique_referenced
        if unused_inputs:
            self.warnings.append(f"Inputs - InputId définis mais non utilisés: {', '.join(unused_inputs)}")
        
        if valid:
            print(f"✅ Inputs validé: {len(df)} lignes")
        return valid
    
    def clean_tag(self, tag: str) -> str:
        """Nettoie un tag pour le reportName"""
        if not tag or pd.isna(tag):
            return ""
        clean = re.sub(r'[\[\]]', '', str(tag))
        clean = ''.join([c for c in clean if c.isalpha() and c.isupper()])
        return clean
    
    def generate_report_name(self, tags_dict: Dict[str, str]) -> str:
        """Génère le reportName à partir des tags"""
        parts = []
        
        for level in ['Level1', 'Level2', 'Level3']:
            if level in tags_dict and tags_dict[level]:
                cleaned = self.clean_tag(tags_dict[level])
                if cleaned:
                    parts.append(cleaned)
        
        report_name = '_'.join(parts)
        return report_name if report_name else "REPORT"
    
    def get_form_fields(self, input_id: str) -> Optional[Dict]:
        """Récupère les champs d'un formulaire"""
        if not input_id:
            return None
        
        df = self.data['Inputs']
        fields_data = df[df['InputId'] == input_id]
        
        if fields_data.empty:
            self.warnings.append(f"InputId '{input_id}' non trouvé dans Inputs")
            return None
        
        fields = []
        for _, field in fields_data.sort_values('FieldOrder').iterrows():
            field_config = {
                "name": field['FieldName'] or f"field_{field['FieldOrder']}",
                "type": field['FieldType'] or 'text',
                "label": field['FieldLabel'] or field['FieldName'],
                "required": bool(field['Required']) if field['Required'] is not None else False,
                "tag": bool(field['Tag']) if 'Tag' in field and field['Tag'] is not None else False
            }
            
            if field['Options']:
                field_config["options"] = [opt.strip() for opt in str(field['Options']).split(',')]
            
            if field['DefaultValue'] is not None:
                if field['FieldType'] == 'checklist':
                    default_str = str(field['DefaultValue'])
                    field_config["default"] = [val.strip() for val in default_str.split(',')]
                else:
                    field_config["default"] = str(field['DefaultValue'])
            
            if field['Validation']:
                field_config["validation"] = field['Validation']
            
            if field['Placeholder']:
                field_config["placeholder"] = field['Placeholder']
            
            if field['Description']:
                field_config["description"] = field['Description']
            
            fields.append(field_config)
        
        return {
            "title": f"Parameters Configuration",
            "fields": fields
        }
    
    def generate_config(self) -> bool:
        """Génère la configuration complète"""
        print("🔨 Génération de la configuration...")
        
        # Valider toutes les feuilles
        validations = [
            self.validate_level1(),
            self.validate_level2(),
            self.validate_level3(),
            self.validate_inputs()
        ]
        
        if not all(validations):
            print("❌ Validation échouée")
            return False
        
        # Générer la configuration
        self.config = {}
        
        for _, level1 in self.data['Level1'].iterrows():
            config_key = level1['value']
            
            self.config[config_key] = {
                "title": level1['Title'],
                "description": level1['Description'],
                "form": {
                    "type": level1['Form'],
                    "label": level1['label']
                },
                "pageConfigs": []
            }
            
            # Traiter les pages
            page_configs = self.generate_pages_for_level1(level1)
            self.config[config_key]['pageConfigs'] = page_configs
        
        print("✅ Configuration générée")
        return True
    
    def generate_pages_for_level1(self, level1_row) -> List[Dict]:
        """Génère les pages pour un Level1"""
        page_configs = []
        page_counter = 1
        
        # Pages directes depuis Level1
        if level1_row['Action'] and '.json' in str(level1_row['Action']):
            form_data = self.get_form_fields(level1_row['InputId'])
            tags_dict = {'Level1': level1_row['Tag']}
            report_name = self.generate_report_name(tags_dict)
            
            page_configs.append({
                "id": f"{level1_row['value']}_{page_counter}",
                "name": level1_row['Title'],
                "pageConfig": str(level1_row['Action']).strip(),
                "requirements": {
                    "Level1": level1_row['value']
                },
                "metadata": self.create_metadata(level1_row, form_data, tags_dict, report_name)
            })
            page_counter += 1
        
        # Pages depuis Level2
        level2_items = self.data['Level2'][self.data['Level2']['PreviousLevel'] == level1_row['Id']]
        
        for _, level2 in level2_items.iterrows():
            # Pages directes Level2
            if level2['Action'] and '.json' in str(level2['Action']):
                form_data = self.get_form_fields(level2['InputId'])
                tags_dict = {
                    'Level1': level1_row['Tag'],
                    'Level2': level2['Tag']
                }
                report_name = self.generate_report_name(tags_dict)
                
                page_configs.append({
                    "id": f"{level1_row['value']}_{level2['value'].upper()[:4]}_{page_counter}",
                    "name": level2['Title'],
                    "pageConfig": str(level2['Action']).strip(),
                    "requirements": {
                        "Level1": level1_row['value'],
                        "Level2": level2['value']
                    },
                    "metadata": self.create_metadata(level2, form_data, tags_dict, report_name)
                })
                page_counter += 1
            
            # Pages depuis Level3
            if level2['Action'] and 'NextLevel' in str(level2['Action']):
                level3_items = self.data['Level3'][self.data['Level3']['PreviousLevel'] == level2['Id']]
                
                for _, level3 in level3_items.iterrows():
                    form_data = self.get_form_fields(level3['InputId'])
                    tags_dict = {
                        'Level1': level1_row['Tag'],
                        'Level2': level2['Tag'],
                        'Level3': level3['Tag']
                    }
                    report_name = self.generate_report_name(tags_dict)
                    
                    page_configs.append({
                        "id": f"{level1_row['value']}_{level2['value'].upper()[:4]}_{level3['value'].upper()[:4]}_{page_counter}",
                        "name": level3['Title'],
                        "pageConfig": str(level3['Action']).strip(),
                        "requirements": {
                            "Level1": level1_row['value'],
                            "Level2": level2['value'],
                            "Level3": level3['value']
                        },
                        "metadata": self.create_metadata(level3, form_data, tags_dict, report_name)
                    })
                    page_counter += 1
        
        return page_configs
    
    def create_metadata(self, row, form_data, tags_dict, report_name) -> Dict:
        """Crée les métadonnées pour une page"""
        return {
            "tag": row['Tag'] or "",
            "description": row['Description'] or "",
            "reportName": report_name,
            "generateReport": True,
            "inputId": row['InputId'],
            "form": form_data
        }
    
    def save_config(self, output_file: str = 'config.json') -> bool:
        """Sauvegarde la configuration"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Configuration sauvegardée: {output_file}")
            return True
        except Exception as e:
            self.errors.append(f"Erreur sauvegarde: {str(e)}")
            return False
    
    def print_summary(self):
        """Affiche un résumé"""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA GÉNÉRATION")
        print("="*60)
        
        if self.errors:
            print("\n❌ ERREURS:")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print("\n⚠️  AVERTISSEMENTS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.config:
            print("\n✅ CONFIGURATION GÉNÉRÉE:")
            total_pages = 0
            total_forms = 0
            
            for key, value in self.config.items():
                pages = len(value['pageConfigs'])
                total_pages += pages
                forms = sum(1 for p in value['pageConfigs'] if p['metadata']['form'])
                total_forms += forms
                
                print(f"\n  {key}:")
                print(f"    - Pages: {pages}")
                print(f"    - Formulaires: {forms}")
                
                for page in value['pageConfigs']:
                    form_info = f" ({len(page['metadata']['form']['fields'])} champs)" if page['metadata']['form'] else ""
                    print(f"      • {page['name']}: {page['metadata']['reportName']}{form_info}")
            
            print(f"\n📈 TOTAL: {total_pages} pages, {total_forms} formulaires")
        
        print("\n" + "="*60)

def main():
    """Point d'entrée principal"""
    excel_path = "DashboardConfig.xlsx"
    
    print("🚀 Générateur de Configuration avec Vérifications")
    print("="*60)
    
    # Initialiser le générateur
    generator = ConfigGenerator(excel_path)
    
    # Charger l'Excel
    if not generator.load_excel():
        print("❌ Échec du chargement de l'Excel")
        return False
    
    # Générer la configuration
    if not generator.generate_config():
        print("❌ Échec de la génération de la configuration")
        return False
    
    # Sauvegarder
    if not generator.save_config():
        print("❌ Échec de la sauvegarde")
        return False
    
    # Afficher le résumé
    generator.print_summary()
    
    # Afficher les recommandations
    if generator.warnings:
        print("\n💡 RECOMMANDATIONS:")
        print("  1. Corriger les avertissements pour une configuration optimale")
        print("  2. Vérifier que tous les InputId référencés existent")
        print("  3. S'assurer que les fichiers .json existent dans le dossier page_configs/")
    
    return True

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    exit(exit_code)