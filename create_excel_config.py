# create_complete_excel.py
import pandas as pd

# Créer un writer Excel
with pd.ExcelWriter('DashboardConfig_Complete.xlsx', engine='openpyxl') as writer:
    
    # FEUILLE 1: Documentation_Levels
    doc_levels = {
        'Champ': ['Id', 'Title', 'Tag', 'value', 'Description', 'fontSize', 'Form', 'label', 'Action', 'InputId', 'PreviousLevel'],
        'Niveau': ['Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level1,2,3', 'Level2,3'],
        'Type': ['Numérique', 'Texte', 'Texte', 'Texte', 'Texte', 'Texte', 'Texte', 'Texte', 'Texte', 'Texte', 'Numérique'],
        'Requis': ['✅', '✅', '✅', '✅', '✅', '❌', '✅', '✅', '✅', '❌', '✅'],
        'Description': [
            'Identifiant unique dans le niveau',
            'Nom affiché à l\'utilisateur',
            'Code court pour référence',
            'Valeur technique (identifiant)',
            'Description de l\'option',
            'Taille de police CSS',
            'Type de contrôle',
            'Label du champ',
            'Comportement suivant',
            'ID du formulaire associé',
            'ID du niveau parent'
        ],
        'Exemple': ['1.0, 2.0', '"Sensitivity", "Equity"', '[SENSI], [EQ]', '"Sensitivity", "Equity"', '"Sensitivity reports"', '"15px"', '"dropdown", "checklist"', '"Select Category:"', '"NextLevel", "file.json"', '"EQ_INPUT"', '1.0, 2.0'],
        'Notes': [
            'Incrémental, unique par niveau',
            'Doit être clair et descriptif',
            'Utilisé pour générer reportName',
            'Utilisé dans requirements',
            'Affiché en tooltip',
            'Optionnel, 15px par défaut',
            'Voir feuille FormTypes',
            'Texte affiché au-dessus',
            '"NextLevel" ou fichier .json',
            'Obligatoire si Action contient .json',
            'Référence à Id du niveau supérieur'
        ]
    }
    pd.DataFrame(doc_levels).to_excel(writer, sheet_name='Documentation_Levels', index=False)
    
    # FEUILLE 2: Documentation_Inputs
    doc_inputs = {
        'Champ': ['InputId', 'FieldOrder', 'FieldName', 'FieldType', 'FieldLabel', 'Options', 'Required', 'DefaultValue', 'Validation', 'Placeholder', 'Description', 'Tag'],
        'Type': ['Texte', 'Numérique', 'Texte', 'Texte', 'Texte', 'Texte', 'Booléen', 'Texte', 'Texte', 'Texte', 'Texte', 'Booléen'],
        'Requis': ['✅', '✅', '✅', '✅', '✅', '❌', '✅', '❌', '❌', '❌', '❌', '✅'],
        'Description': [
            'Identifiant unique du formulaire',
            'Ordre d\'affichage des champs',
            'Nom technique du champ',
            'Type de champ',
            'Label affiché à l\'utilisateur',
            'Options pour dropdown/checklist',
            'Champ obligatoire',
            'Valeur par défaut',
            'Règles de validation',
            'Texte d\'aide',
            'Description du champ',
            'Tag pour classification'
        ],
        'Exemple': [
            '"EQ_INPUT", "CORR_INPUT"',
            '1, 2, 3',
            '"region", "period"',
            '"dropdown", "checklist", "text"',
            '"Select Region:", "Timeframe:"',
            '"US,Europe,Asia"',
            'true, false',
            '"US", "30", "Quarterly"',
            '"number,min=1", "min 1"',
            '"Enter days", "Select..."',
            '"Geographic region"',
            'true, false'
        ],
        'Notes': [
            'Doit être référencé dans Levels',
            'Définit l\'ordre dans le formulaire',
            'Utilisé dans le code',
            'Voir FormTypes pour les types',
            'Texte descriptif',
            'Obligatoire pour dropdown/checklist',
            'true = validation requise',
            'Format dépend de FieldType',
            'Voir ValidationTypes',
            'Affiché dans les champs vides',
            'Tooltip d\'aide',
            'true = champ tagué'
        ]
    }
    pd.DataFrame(doc_inputs).to_excel(writer, sheet_name='Documentation_Inputs', index=False)
    
    # FEUILLE 3: FormTypes
    form_types = {
        'FormType': ['dropdown', 'checklist', 'text', 'number', 'date', 'range', 'email', 'password', 'textarea', 'file'],
        'Description': [
            'Sélection unique',
            'Sélection multiple',
            'Texte libre',
            'Nombre',
            'Date',
            'Plage numérique',
            'Email',
            'Mot de passe',
            'Texte multiligne',
            'Fichier'
        ],
        'HTML Element': [
            '<select>',
            '<input type="checkbox">',
            '<input type="text">',
            '<input type="number">',
            '<input type="date">',
            '<input type="range">',
            '<input type="email">',
            '<input type="password">',
            '<textarea>',
            '<input type="file">'
        ],
        'Multiple': ['❌', '✅', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌'],
        'Options Requises': ['✅', '✅', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌'],
        'DefaultValue Format': ['Texte simple', 'Liste séparée par virgules', 'Texte simple', 'Nombre', 'Date ISO', 'Nombre', 'Email valide', 'Texte masqué', 'Texte multiligne', 'Chemin fichier'],
        'Usage Example': [
            'Select Region: [US|Europe|Asia]',
            'Select Regions: [✓US] [✓Europe] [ ]Asia',
            'Timeframe: [______]',
            'Quantity: [0-9]',
            'Date: [YYYY-MM-DD]',
            'Value: [0─────100]',
            'Email: [user@domain.com]',
            'Password: [•••••••]',
            'Comments: [______\n______]',
            'Upload: [Parcourir...]'
        ]
    }
    pd.DataFrame(form_types).to_excel(writer, sheet_name='FormTypes', index=False)
    
    # FEUILLE 4: ValidationTypes
    validation_types = {
        'Validation': ['required', 'min', 'max', 'min', 'max', 'step', 'pattern', 'maxlength', 'minlength', 'email', 'url'],
        'Applicable à': ['Tous', 'checklist', 'checklist', 'number/range', 'number/range', 'number/range', 'text', 'text/textarea', 'text/textarea', 'email', 'text'],
        'Description': [
            'Champ obligatoire',
            'Minimum d\'éléments',
            'Maximum d\'éléments',
            'Valeur minimale',
            'Valeur maximale',
            'Incrément',
            'Regex pattern',
            'Longueur max',
            'Longueur min',
            'Format email',
            'Format URL'
        ],
        'Format': ['"required"', '"min=X"', '"max=X"', '"min=X"', '"max=X"', '"step=X"', '"pattern=..."', '"maxlength=X"', '"minlength=X"', '"email"', '"url"'],
        'Exemple': ['true', '"min 1"', '"max 5"', '"min=0"', '"max=100"', '"step=5"', '"pattern=\\d{3}"', '"maxlength=255"', '"minlength=3"', 'true', 'true']
    }
    pd.DataFrame(validation_types).to_excel(writer, sheet_name='ValidationTypes', index=False)
    
    # FEUILLE 5: Level1
    level1_data = {
        'Id': [1, 2, 3],
        'Title': ['Sensitivity', 'Stress Test', 'Correlation'],
        'Tag': ['[SENSI]', '[ST]', '[CORR]'],
        'value': ['Sensitivity', 'StressTest', 'Correlation'],
        'Description': ['Sensitivity reports', 'Stress test reports', 'Correlation reports'],
        'fontSize': ['15px', '15px', '15px'],
        'Form': ['dropdown', 'dropdown', 'dropdown'],
        'label': ['Select Category:', 'Select Category:', 'Select Category:'],
        'Action': ['NextLevel', 'NextLevel', 'Correlation_report.json'],
        'InputId': [None, None, 'CORR_INPUT']
    }
    pd.DataFrame(level1_data).to_excel(writer, sheet_name='Level1', index=False)
    
    # FEUILLE 6: Level2
    level2_data = {
        'PreviousLevel': [1, 1, 2],
        'Id': [1, 2, 1],
        'Title': ['Equity', 'Credit', 'Activity1'],
        'Tag': ['[EQ]', '[CR]', '[ACT1]'],
        'value': ['Equity', 'Credit', 'Activity1'],
        'Description': ['Equity instruments', 'Credit instruments', 'Primary activity'],
        'fontSize': ['15px', '15px', '15px'],
        'Form': ['dropdown', 'dropdown', 'dropdown'],
        'label': ['Select Asset Class:', 'Select Asset Class:', 'Select Activity:'],
        'Action': ['EQ_dashboard.json', 'CR_dashboard.json', 'NextLevel'],
        'InputId': ['EQ_INPUT', 'CR_INPUT', None]
    }
    pd.DataFrame(level2_data).to_excel(writer, sheet_name='Level2', index=False)
    
    # FEUILLE 7: Level3
    level3_data = {
        'PreviousLevel': [1, 1],
        'Id': [1, 2],
        'Title': ['Sub Activity A', 'Sub Activity B'],
        'Tag': ['[]', '[SUB]'],
        'value': ['SubActivityA', 'SubActivityB'],
        'Description': ['Detailed sub-activity A', 'Detailed sub-activity B'],
        'fontSize': ['15px', '15px'],
        'Form': ['dropdown', 'checklist'],
        'label': ['Select Sub-activity:', 'Select Sub-activities:'],
        'Action': ['SubA_dashboard.json', 'SubB_dashboard.json'],
        'InputId': ['SUB_INPUT', 'SUB_INPUT_2']
    }
    pd.DataFrame(level3_data).to_excel(writer, sheet_name='Level3', index=False)
    
    # FEUILLE 8: Inputs
    inputs_data = {
        'InputId': ['CORR_INPUT', 'CORR_INPUT', 'EQ_INPUT', 'EQ_INPUT', 'CR_INPUT', 'CR_INPUT', 'SUB_INPUT', 'SUB_INPUT', 'SUB_INPUT_2'],
        'FieldOrder': [1, 2, 1, 2, 1, 2, 1, 2, 1],
        'FieldName': ['period', 'confidence', 'region', 'timeframe', 'rating', 'horizon', 'detail', 'format', 'activities'],
        'FieldType': ['dropdown', 'dropdown', 'checklist', 'text', 'dropdown', 'text', 'dropdown', 'dropdown', 'checklist'],
        'FieldLabel': ['Select Period:', 'Confidence Level:', 'Select Regions:', 'Timeframe (days):', 'Select Rating:', 'Time Horizon:', 'Detail Level:', 'Output Format:', 'Select Activities:'],
        'Options': ['Quarterly,Monthly,Yearly', '95%,99%,99.5%', 'US,Europe,Asia,EMEA', None, 'AAA,AA,A,BBB', None, 'High,Medium,Low', 'PDF,Excel,HTML', 'ActivityA,ActivityB,ActivityC'],
        'Required': [True, True, False, True, True, True, True, True, True],
        'DefaultValue': ['Quarterly', '99%', 'US,Europe', '30', 'A', '10', 'Medium', 'HTML', 'ActivityA'],
        'Validation': [None, None, 'min 1', 'number,min=1', None, 'number,min=1', None, None, 'min 1'],
        'Placeholder': [None, None, None, 'Enter days', None, 'Enter days', None, None, None],
        'Description': ['Correlation time period', 'Statistical confidence', 'Geographic regions', 'Analysis timeframe', 'Credit rating', 'Time horizon', 'Report detail level', 'Report format', 'Activities to include'],
        'Tag': [True, False, True, False, True, False, True, False, True]
    }
    pd.DataFrame(inputs_data).to_excel(writer, sheet_name='Inputs', index=False)

print("✅ Fichier Excel complet créé: DashboardConfig_Complete.xlsx")
print("\n📁 Feuilles incluses:")
print("  1. Documentation_Levels - Documentation des champs Levels")
print("  2. Documentation_Inputs - Documentation des champs Inputs")
print("  3. FormTypes - Tous les types de formulaires disponibles")
print("  4. ValidationTypes - Types de validation possibles")
print("  5. Level1 - Données Level1")
print("  6. Level2 - Données Level2")
print("  7. Level3 - Données Level3")
print("  8. Inputs - Données Inputs")