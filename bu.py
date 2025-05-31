import os
import json
import yaml
import pandas as pd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Union, Generator
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
import time
import shutil

# Constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_BATCH_SIZE = 1000
PROGRESS_FILE = "processing_progress.json"
GLOBAL_CONFIG_FILE = "global_loader_config.yaml"
RESERVED_COLUMNS = {"_loaded_timestamp", "_source_filename"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration class for processing parameters."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    max_connections: int = 5
    min_connections: int = 1
    connection_timeout: int = 30
    retry_attempts: int = 3
    enable_progress_tracking: bool = True
    enable_data_validation: bool = True
    enable_duplicate_handling: bool = False
    timestamp_tolerance_seconds: float = 1.0
    
@dataclass
class FileProcessingRule:
    """Defines rules for processing specific types of files."""
    base_name: str                  # e.g., 'sales' from 'sales_rule.yaml'
    directory: str                  # The source directory for this specific rule's files
    file_pattern: str               # Regular expression to match filenames
    date_format: Optional[str] = None # Format for date extraction from filename
    start_row: Optional[int] = None
    start_col: Optional[int] = None
    frequency: Optional[str] = None
    mode: Optional[str] = None      # "cancel_and_replace", "audit", or "insert"
    extract_date_col_name: Optional[str] = None # Column name for extracted date
    _compiled_pattern: Any = field(init=False, repr=False)

    def __post_init__(self):
        self._compiled_pattern = re.compile(self.file_pattern)

    def match(self, filename: str) -> Optional[re.Match]:
        """Attempts to match the filename against the rule's pattern."""
        return self._compiled_pattern.match(filename)
    
    @property
    def target_table(self) -> str:
        return self.base_name

    @property
    def mapping_file(self) -> str:
        return f"{self.base_name}_mapping.csv"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate rule configuration."""
        errors = []
        if not self.directory:
            errors.append("Directory is required")
        if not self.file_pattern:
            errors.append("File pattern is required")
        if self.mode and self.mode not in ["cancel_and_replace", "audit", "insert"]:
            errors.append(f"Invalid mode: {self.mode}")
        if self.extract_date_col_name and not self.date_format:
            errors.append("date_format is required when extract_date_col_name is specified")
        return len(errors) == 0, errors

@dataclass
class FileContext:
    """Holds all necessary information for processing a single file."""
    filepath: Path
    filename: str
    target_table: str
    mapping_filepath: Path
    extracted_timestamp_str: str
    file_modified_timestamp: datetime
    start_row: int
    start_col: int
    mode: str
    extract_date_col_name: Optional[str]

class DatabaseManager:
    """Handles database connections and operations."""
    
    def __init__(self, db_config: Dict[str, Any], config: ProcessingConfig):
        self.db_config = db_config
        self.config = config
        self.connection_pool = None
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                **self.db_config
            )
            logger.info(f"Database connection pool initialized with {self.config.max_connections} max connections")
        except psycopg2.Error as e:
            logger.critical(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with proper cleanup."""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_latest_timestamp_for_filename(self, target_table: str, source_filename: str) -> Optional[datetime]:
        """Get the most recent loaded_timestamp for a given filename"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = sql.SQL("""
                        SELECT MAX(_loaded_timestamp) 
                        FROM {} 
                        WHERE _source_filename = %s
                    """).format(sql.Identifier(target_table))
                    cursor.execute(query, (source_filename,))
                    result = cursor.fetchone()[0]
                    return result
        except psycopg2.errors.UndefinedTable:
            return None
        except Exception as e:
            logger.error(f"Error getting latest timestamp for {source_filename}: {e}")
            raise

    def delete_by_source_filename(self, conn, target_table: str, source_filename: str) -> int:
        """Delete all records with matching filename regardless of timestamp"""
        try:
            with conn.cursor() as cursor:
                query = sql.SQL("DELETE FROM {} WHERE _source_filename = %s").format(sql.Identifier(target_table))
                cursor.execute(query, (source_filename,))
                return cursor.rowcount
        except psycopg2.errors.UndefinedTable:
            return 0
        except Exception as e:
            logger.error(f"Error deleting records for {source_filename}: {e}")
            raise

    def file_exists_in_db(self, target_table: str, file_modified_timestamp: datetime, source_filename: str) -> bool:
        """Check if file with same timestamp and filename exists in DB"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = sql.SQL("""
                        SELECT 1 FROM {} 
                        WHERE _source_filename = %s 
                        AND ABS(EXTRACT(EPOCH FROM (_loaded_timestamp - %s))) <= %s
                        LIMIT 1
                    """).format(sql.Identifier(target_table))
                    cursor.execute(query, (
                        source_filename, 
                        file_modified_timestamp,
                        self.config.timestamp_tolerance_seconds
                    ))
                    return cursor.fetchone() is not None
        except psycopg2.errors.UndefinedTable:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence in DB for {source_filename}: {e}")
            return False

class DataValidator:
    """Handles data validation and quality checks."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def validate_dataframe(self, df: pd.DataFrame, mapping: pd.DataFrame, filename: str) -> Tuple[bool, List[str]]:
        """Validate DataFrame against mapping requirements."""
        errors = []
        
        if not self.config.enable_data_validation:
            return True, errors
        
        # Check for reserved column names
        reserved_cols = [col for col in RESERVED_COLUMNS if col in df.columns]
        if reserved_cols:
            errors.append(f"Critical: Reserved column names used: {', '.join(reserved_cols)}")
            return False, errors
        
        # Check for missing flagged columns (LoadFlag=1)
        required_load_cols = mapping[mapping['LoadFlag'] == 1]['TargetColumn'].values
        missing_required_cols = [col for col in required_load_cols if col not in df.columns]
        if missing_required_cols:
            errors.append(f"Critical: Columns marked for loading are missing: {', '.join(missing_required_cols)}")
            return False, errors
        
        # Check data types against mapping
        self._validate_data_types(df, mapping, errors, filename)
        
        return len(errors) == 0, errors
    
    def _validate_data_types(self, df: pd.DataFrame, mapping: pd.DataFrame, errors: List[str], filename: str) -> None:
        """Validate data types in DataFrame against mapping specifications."""
        type_mapping = {
            "INTEGER": ["int", "int32", "int64"],
            "NUMERIC": ["float", "float32", "float64"],
            "TIMESTAMP": ["datetime64", "datetime"],
            "BOOLEAN": ["bool"],
            "TEXT": ["object"]
        }
        
        for _, row in mapping.iterrows():
            if row['LoadFlag'] != 1 or row['TargetColumn'] not in df.columns:
                continue
                
            col = row['TargetColumn']
            expected_type = row['DataType'].upper()
            actual_type = str(df[col].dtype)
            
            # Check if actual type matches expected type category
            if expected_type in type_mapping:
                type_match = any(t in actual_type for t in type_mapping[expected_type])
                if not type_match:
                    errors.append(
                        f"Type mismatch in column '{col}': "
                        f"Expected {expected_type}, found {actual_type}"
                    )
            
            # Additional null checks for required columns
            if row.get('IndexColumn') == "Y":
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"Warning: {null_count} null values in index column '{col}'")

class ProgressTracker:
    """Tracks processing progress for resume capability using a local file."""
    
    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = progress_file
        self.processed_files = self._load_progress()
    
    def _load_progress(self) -> set:
        """Load previously processed files."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load progress file '{self.progress_file}': {e}")
        return set()
    
    def save_progress(self) -> None:
        """Save current progress."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
                logger.error(f"Could not save progress to '{self.progress_file}': {e}")
    
    def mark_processed(self, filepath: Path) -> None:
        """Mark file as processed."""
        self.processed_files.add(str(filepath))
        self.save_progress()
    
    def is_processed(self, filepath: Path) -> bool:
        """Check if file has been processed."""
        return str(filepath) in self.processed_files
    
    def get_remaining_files(self, all_file_contexts: List[FileContext]) -> List[FileContext]:
        """Get list of unprocessed files based on the tracking file."""
        return [fc for fc in all_file_contexts if not self.is_processed(fc.filepath)]
    
    def reset_progress(self) -> None:
        """Reset progress tracking."""
        self.processed_files = set()
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            logger.info(f"Progress file '{self.progress_file}' removed.")

class FileProcessor:
    """Handles file operations and data extraction."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def load_file(self, file_path: Path, start_row: int = 0, start_col: int = 0) -> pd.DataFrame:
        """Load file into DataFrame."""
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(
                    file_path,
                    skiprows=range(1, start_row + 1),
                    header=0
                ).iloc[:, start_col:]
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(
                    file_path,
                    skiprows=start_row,
                    header=0
                ).iloc[:, start_col:]
            else:
                raise ValueError(f"Unsupported file format: {file_ext} for {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise

class PostgresLoader:
    """Main loader class with automatic mapping file creation when missing."""
    
    def __init__(self, db_config: Dict[str, Any],
                 global_start_row: int = 0, global_start_col: int = 0,
                 delete_files: str = "N",
                 global_config_file: str = None,
                 rules_folder_path: str = "rules"):
        
        self.config = self._load_global_config(global_config_file)
        self.processing_rules = self._load_processing_rules(rules_folder_path)
        self.db_manager = DatabaseManager(db_config, self.config)
        self.validator = DataValidator(self.config)
        self.file_processor = FileProcessor(self.config)
        self.progress_tracker = ProgressTracker() if self.config.enable_progress_tracking else None
        
        self.global_start_row = global_start_row
        self.global_start_col = global_start_col
        self.delete_files = delete_files.upper() == "Y"
        
        self._validate_setup()
    
    def _load_global_config(self, config_file: Optional[str]) -> ProcessingConfig:
        """Load global configuration from file or use defaults."""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                return ProcessingConfig(**config_data)
            except Exception as e:
                logger.warning(f"Could not load global config file {config_file}: {e}. Using defaults.")
        
        return ProcessingConfig()
    
    def _load_processing_rules(self, rules_folder: str) -> List[FileProcessingRule]:
        """Loads file processing rules from a folder containing individual _rule.yaml files."""
        rules_folder_path = Path(rules_folder)
        if not rules_folder_path.exists() or not rules_folder_path.is_dir():
            logger.error(f"Rules folder not found or is not a directory: {rules_folder_path}")
            return []
        
        rules = []
        for rule_file_path in rules_folder_path.iterdir():
            if rule_file_path.is_file() and rule_file_path.name.endswith("_rule.yaml"):
                try:
                    base_name = rule_file_path.stem.replace("_rule", "")
                    if not base_name:
                        logger.warning(f"Skipping rule file '{rule_file_path.name}': Could not determine base name.")
                        continue

                    with open(rule_file_path, 'r') as f:
                        rule_data = yaml.safe_load(f)
                    
                    # Create rule instance
                    rule = FileProcessingRule(
                        base_name=base_name,
                        directory=rule_data.get('directory'),
                        file_pattern=rule_data.get('file_pattern'),
                        date_format=rule_data.get('date_format'),
                        start_row=rule_data.get('start_row'),
                        start_col=rule_data.get('start_col'),
                        frequency=rule_data.get('frequency'),
                        mode=rule_data.get('mode', 'insert'),
                        extract_date_col_name=rule_data.get('extract_date_col_name')
                    )
                    
                    # Validate rule
                    is_valid, errors = rule.validate()
                    if not is_valid:
                        logger.error(f"Invalid rule in {rule_file_path.name}: {', '.join(errors)}")
                        continue
                    
                    rules.append(rule)
                    logger.info(f"Loaded rule '{base_name}' from {rule_file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load rule from {rule_file_path.name}: {e}")
        
        return rules
    
    def _validate_setup(self) -> None:
        """Validate setup before processing and create missing mapping files."""
        logger.info("Validating setup...")
        
        for rule in self.processing_rules:
            rule_source_dir = Path(rule.directory)
            if not rule_source_dir.exists():
                rule_source_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {rule_source_dir}")
            
            mapping_filepath = rule_source_dir / rule.mapping_file
            
            # Create mapping file if it doesn't exist
            if not mapping_filepath.exists():
                logger.warning(f"Mapping file missing for rule '{rule.base_name}'. Creating new mapping file.")
                
                # Find a sample file to base mapping on
                sample_file = None
                for file_path in rule_source_dir.iterdir():
                    if file_path.is_file() and rule.match(file_path.name):
                        sample_file = file_path
                        break
                
                if sample_file:
                    generate_empty_mapping_file(
                        source_filepath=sample_file,
                        mapping_filepath=mapping_filepath,
                        start_row=rule.start_row if rule.start_row is not None else self.global_start_row,
                        start_col=rule.start_col if rule.start_col is not None else self.global_start_col,
                        sample_rows_for_type_inference=100
                    )
                    logger.info(f"Created new mapping file: {mapping_filepath}")
                    logger.warning("IMPORTANT: Please review and edit the mapping file before processing data!")
                else:
                    logger.error(f"Could not find sample file to generate mapping for rule '{rule.base_name}'. Please add a data file to {rule_source_dir}")

        # Test database connection
        if not self.db_manager.test_connection():
            raise ConnectionError("Database connection test failed")
        
        logger.info("Setup validation completed successfully")
    
    def get_files_to_process(self) -> List[FileContext]:
        """
        Scans each rule's specified directory, matches files, and prepares FileContext objects.
        Applies database-level audit check for 'audit' mode.
        """
        all_potential_file_contexts: List[FileContext] = []

        for rule in self.processing_rules:
            rule_source_dir = Path(rule.directory)
            if not rule_source_dir.exists() or not rule_source_dir.is_dir():
                logger.warning(f"Rule '{rule.base_name}': Source directory '{rule_source_dir}' does not exist or is not a directory. Skipping files for this rule.")
                continue

            mapping_filepath_for_rule = rule_source_dir / rule.mapping_file
            if not mapping_filepath_for_rule.exists():
                logger.warning(f"Rule '{rule.base_name}': Required mapping file '{mapping_filepath_for_rule}' not found in its specified directory. Skipping files for this rule.")
                continue

            for file_path in rule_source_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                    filename = file_path.name
                    match = rule.match(filename)
                    
                    if match:
                        extracted_timestamp_str = ""
                        if rule.date_format and match.groups():
                            try:
                                date_str_from_regex = "".join(match.groups())
                                datetime.strptime(date_str_from_regex, rule.date_format) 
                                extracted_timestamp_str = date_str_from_regex
                            except ValueError:
                                logger.warning(f"File '{filename}': Date format '{rule.date_format}' did not match captured group(s) '{date_str_from_regex}' for rule '{rule.base_name}'. Extracted date will be empty.")
                                extracted_timestamp_str = ""
                        elif rule.date_format and not match.groups():
                            logger.warning(f"File '{filename}': Rule '{rule.base_name}' has a date_format but no capturing groups in its regex. Extracted date will be empty.")
                        
                        file_modified_timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)

                        file_context = FileContext(
                            filepath=file_path,
                            filename=filename,
                            target_table=rule.target_table,
                            mapping_filepath=mapping_filepath_for_rule,
                            extracted_timestamp_str=extracted_timestamp_str,
                            file_modified_timestamp=file_modified_timestamp,
                            start_row=rule.start_row if rule.start_row is not None else self.global_start_row,
                            start_col=rule.start_col if rule.start_col is not None else self.global_start_col,
                            mode=rule.mode if rule.mode else "insert",
                            extract_date_col_name=rule.extract_date_col_name
                        )
                        all_potential_file_contexts.append(file_context)
        
        # Apply filtering based on progress tracker and audit mode
        files_to_process = []
        for fc in all_potential_file_contexts:
            # First, check progress tracker if enabled
            if self.progress_tracker and self.progress_tracker.is_processed(fc.filepath):
                logger.info(f"Skipping {fc.filename}: Already processed (tracked by {PROGRESS_FILE}).")
                continue
            
            # Then, apply audit mode check if specified in rule
            if fc.mode == "audit":
                try:
                    if self.db_manager.file_exists_in_db(fc.target_table, fc.file_modified_timestamp, fc.filename):
                        logger.info(f"Skipping {fc.filename}: Found in DB with same timestamp and filename (audit mode).")
                        # If found in DB, also mark as processed in progress_tracker to prevent future unnecessary DB checks
                        if self.progress_tracker:
                            self.progress_tracker.mark_processed(fc.filepath)
                        continue
                except Exception as e:
                    logger.error(f"Failed to check audit status for {fc.filename}: {e}. Will attempt to process.")
                    # If DB check fails, proceed with processing, log the error.
            
            files_to_process.append(fc)

        logger.info(f"Found {len(all_potential_file_contexts)} total potential files across all rule directories, {len(files_to_process)} selected for processing after checks.")
        return files_to_process
    
    def process_files(self) -> Dict[str, int]:
        """Process all files matching rules."""
        file_contexts = self.get_files_to_process()
        if not file_contexts:
            logger.info("No files to process.")
            return {"processed": 0, "failed": 0}
        
        processed_count = 0
        failed_count = 0
        
        for fc in file_contexts:
            success = self.process_file(fc)
            if success:
                processed_count += 1
            else:
                failed_count += 1
                
        return {"processed": processed_count, "failed": failed_count}
    
    def process_file(self, file_context: FileContext) -> bool:
        """Process a single file with comprehensive error handling."""
        try:
            logger.info(f"Processing: {file_context.filename} -> Table: {file_context.target_table}, Mode: {file_context.mode}")
            
            df = self.file_processor.load_file(
                file_context.filepath,
                file_context.start_row,
                file_context.start_col
            )
            
            if df.empty:
                logger.warning(f"File {file_context.filename} is empty or has no data after skipping rows/cols, skipping")
                if self.progress_tracker: 
                    self.progress_tracker.mark_processed(file_context.filepath)
                return True
            
            mapping = pd.read_csv(file_context.mapping_filepath)
            
            is_valid, errors = self.validator.validate_dataframe(df, mapping, file_context.filename)
            
            for error in errors:
                if "Critical:" in error:
                    logger.error(error)
                else:
                    logger.warning(error)
            
            if not is_valid:
                logger.error(f"Validation failed for {file_context.filename}. Skipping.")
                return False
            
            df = self._apply_transformations(df, mapping)

            # Add reserved timestamp and filename columns to DataFrame
            df['_loaded_timestamp'] = file_context.file_modified_timestamp
            df['_source_filename'] = file_context.filename

            # Add extracted date column if specified
            if file_context.extract_date_col_name and file_context.extracted_timestamp_str:
                try:
                    extracted_dt = datetime.strptime(
                        file_context.extracted_timestamp_str, 
                        next(rule.date_format for rule in self.processing_rules 
                             if rule.base_name == file_context.target_table)
                    )
                    df[file_context.extract_date_col_name] = extracted_dt
                except Exception as e:
                    logger.warning(f"Failed to add extracted date column '{file_context.extract_date_col_name}' for {file_context.filename}: {e}")
                    if file_context.extract_date_col_name not in df.columns:
                         df[file_context.extract_date_col_name] = None

            success = self._load_data_to_db(
                df,
                file_context.target_table,
                file_context.mode,
                file_context.file_modified_timestamp,
                file_context.filename
            )
            
            if success:
                if self.progress_tracker:
                    self.progress_tracker.mark_processed(file_context.filepath)
                
                if self.delete_files:
                    file_context.filepath.unlink()
                    logger.info(f"File deleted: {file_context.filename}")
                
                logger.info(f"Successfully processed: {file_context.filename}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing file {file_context.filename}: {e}", exc_info=True)
            return False
    
    def _apply_transformations(self, df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
        """Apply column mapping and transformations."""
        # Create mapping dictionary for columns to include
        column_mapping = {}
        for _, row in mapping.iterrows():
            if row['LoadFlag'] == 1 and row['RawColumn'] in df.columns:
                column_mapping[row['RawColumn']] = row['TargetColumn']
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Select only columns that are marked for loading
        columns_to_keep = [col for col in mapping[mapping['LoadFlag'] == 1]['TargetColumn'] if col in df.columns]
        return df[columns_to_keep]
    
    def _load_data_to_db(self, df: pd.DataFrame, target_table: str, mode: str, 
                        file_modified_timestamp: datetime, source_filename: str) -> bool:
        """
        Load data to database based on the specified mode.
        Handles 'cancel_and_replace' and 'insert' (audit mode handles skipping before this).
        """
        try:
            with self.db_manager.get_connection() as conn:
                if mode == "cancel_and_replace":
                    # Check if we have existing records for this filename
                    latest_db_ts = self.db_manager.get_latest_timestamp_for_filename(target_table, source_filename)
                    
                    if latest_db_ts is not None:
                        time_diff = abs((file_modified_timestamp - latest_db_ts).total_seconds())
                        if time_diff <= self.config.timestamp_tolerance_seconds:
                            logger.info(
                                f"Skipping {source_filename} - existing data is identical "
                                f"(DB: {latest_db_ts}, File: {file_modified_timestamp})"
                            )
                            return True  # Consider this successful skip
                        
                        # Delete existing records if file is newer
                        deleted_count = self.db_manager.delete_by_source_filename(conn, target_table, source_filename)
                        logger.info(f"Deleted {deleted_count} old records for {source_filename}")

                # Prepare columns to insert
                columns_to_insert = list(df.columns)
                values_to_insert = [tuple(row) for row in df.values]

                # Create safe SQL query
                columns_str = ", ".join([f'"{col}"' for col in columns_to_insert])
                query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                    sql.Identifier(target_table),
                    sql.SQL(columns_str)
                
                # Execute in batches
                with conn.cursor() as cursor:
                    execute_values(
                        cursor, 
                        query.as_string(conn), 
                        values_to_insert, 
                        page_size=self.config.batch_size
                    )
                    conn.commit()
                
                logger.info(f"Successfully loaded {len(df)} records to database '{target_table}' for file '{source_filename}'.")
            return True
        
        except psycopg2.errors.UniqueViolation as e:
            logger.error(f"Integrity error (UniqueViolation) during load for {source_filename} into {target_table}. Details: {e}. File rejected.")
            return False
        except Exception as e:
            logger.error(f"Database loading failed for table '{target_table}' (file: {source_filename}): {e}")
            return False

# --- UTILITY FUNCTION TO GENERATE MAPPING FILE ---
def generate_empty_mapping_file(
    source_filepath: Path, 
    mapping_filepath: Path,
    start_row: int = 0, 
    start_col: int = 0,
    sample_rows_for_type_inference: int = 100
) -> None:
    """
    Generates an empty mapping CSV file based on the columns found in a source data file.
    It infers basic data types and populates required mapping header fields.
    """
    try:
        if not source_filepath.exists():
            logger.warning(f"Source file for mapping generation not found: {source_filepath}. Creating empty mapping with default headers only.")
            columns = []
            dtypes = {}
        else:
            file_ext = source_filepath.suffix.lower()
            if file_ext == '.csv':
                df_sample = pd.read_csv(source_filepath, skiprows=range(1, start_row + 1), nrows=sample_rows_for_type_inference, header=0)
            elif file_ext in ['.xlsx', '.xls']:
                df_sample = pd.read_excel(source_filepath, skiprows=start_row, nrows=sample_rows_for_type_inference, header=0)
            else:
                logger.warning(f"Unsupported file format for mapping inference: {source_filepath.name}. Creating empty mapping with default headers only.")
                columns = []
                dtypes = {}
                df_sample = pd.DataFrame()
            
            if not df_sample.empty:
                df_sample = df_sample.iloc[:, start_col:]
            columns = df_sample.columns.tolist()
            dtypes = df_sample.dtypes.apply(str).to_dict()

        mapping_data = []
        for col_name in columns:
            pd_type = dtypes.get(col_name, 'object')
            sql_type = "TEXT"
            if "int" in pd_type:
                sql_type = "INTEGER"
            elif "float" in pd_type:
                sql_type = "NUMERIC"
            elif "datetime" in pd_type:
                sql_type = "TIMESTAMP"
            elif "bool" in pd_type:
                sql_type = "BOOLEAN"
            
            mapping_data.append({
                'RawColumn': col_name,
                'TargetColumn': col_name.lower().replace(' ', '_'),
                'DataType': sql_type,
                'LoadFlag': 1,
                'IndexColumn': 'N'
            })
        
        # Add reserved columns for tracking
        mapping_data.append({
            'RawColumn': '_loaded_timestamp',
            'TargetColumn': '_loaded_timestamp',
            'DataType': 'TIMESTAMP',
            'LoadFlag': 1,
            'IndexColumn': 'N'
        })
        mapping_data.append({
            'RawColumn': '_source_filename',
            'TargetColumn': '_source_filename',
            'DataType': 'TEXT',
            'LoadFlag': 1,
            'IndexColumn': 'N'
        })

        # Ensure the mapping file directory exists
        mapping_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create DataFrame and save
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv(mapping_filepath, index=False)
        logger.info(f"Generated empty mapping file: {mapping_filepath}")

    except Exception as e:
        logger.error(f"Error generating mapping file for {source_filepath}: {e}")
        # Fallback: create basic mapping file with just headers
        pd.DataFrame(columns=['RawColumn', 'TargetColumn', 'DataType', 'LoadFlag', 'IndexColumn']).to_csv(mapping_filepath, index=False)
        logger.info(f"Fallback: Created empty mapping file with only headers: {mapping_filepath}")

# --- SAMPLE CONFIGS AND DATA FOLDERS CREATION ---
def create_sample_configs_and_data_folders():
    """Create sample global config, rules config files, and data folders with dummy data."""
    # Clean up previous runs
    if os.path.exists("processing.log"):
        os.remove("processing.log")
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    
    # Ensure rules folder exists
    rules_folder = "rules"
    data_folders = ["./sales_data", "./inventory_data"]
    for path in data_folders + [rules_folder]:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        os.makedirs(path, exist_ok=True)

    # Sample Global Config
    sample_global_config = {
        "chunk_size": 10000,
        "batch_size": 1000,
        "max_connections": 5,
        "retry_attempts": 3,
        "enable_progress_tracking": True,
        "enable_data_validation": True,
        "timestamp_tolerance_seconds": 1.0
    }
    with open(GLOBAL_CONFIG_FILE, 'w') as f:
        yaml.dump(sample_global_config, f, default_flow_style=False)
    print(f"Sample global configuration created: {GLOBAL_CONFIG_FILE}")

    # Sample Sales Rule and Data
    sales_data_path = Path("./sales_data") / 'sales_report_20240101.csv'
    pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02'],
        'Product': ['Laptop', 'Mouse'],
        'Amount': [1200.50, 25.00]
    }).to_csv(sales_data_path, index=False)
    os.utime(sales_data_path, (time.time(), time.mktime(datetime(2024, 1, 1, 10, 0, 0).timetuple())))
    
    sales_rule_config = {
        "directory": "./sales_data",
        "file_pattern": "^sales_report_(\\d{8})\\.csv$",
        "date_format": "%Y%m%d",
        "mode": "cancel_and_replace",
        "extract_date_col_name": "SaleDate"
    }
    with open(Path(rules_folder) / "sales_rule.yaml", 'w') as f:
        yaml.dump(sales_rule_config, f, default_flow_style=False)
    print(f"Sample rule file created: {rules_folder}/sales_rule.yaml")

    # Sample Inventory Rule and Data
    inventory_data_path = Path("./inventory_data") / 'inventory_2024-01-01.xlsx'
    pd.DataFrame({
        'Item_ID': [101, 102],
        'Item_Name': ['Widget', 'Gadget'],
        'Stock': [500, 750]
    }).to_excel(inventory_data_path, index=False)
    os.utime(inventory_data_path, (time.time(), time.mktime(datetime(2024, 1, 1, 11, 0, 0).timetuple())))
    
    inventory_rule_config = {
        "directory": "./inventory_data",
        "file_pattern": "^inventory_(\\d{4}-\\d{2}-\\d{2})\\.xlsx$",
        "date_format": "%Y-%m-%d",
        "start_row": 0,
        "mode": "audit"
    }
    with open(Path(rules_folder) / "inventory_rule.yaml", 'w') as f:
        yaml.dump(inventory_rule_config, f, default_flow_style=False)
    print(f"Sample rule file created: {rules_folder}/inventory_rule.yaml")

if __name__ == "__main__":
    # Create sample configs and data folders for first-time users
    create_sample_configs_and_data_folders()
    
    # Configure your PostgreSQL database connection
    db_config = {
        "dbname": "your_database_name",
        "user": "your_username",
        "password": "your_password",
        "host": "localhost",
        "port": 5432
    }
    
    try:
        # Initialize the loader
        loader = PostgresLoader(
            db_config=db_config,
            global_start_row=0,
            global_start_col=0,
            delete_files="N",
            global_config_file=GLOBAL_CONFIG_FILE,
            rules_folder_path='rules'
        )
        
        # Process files
        result = loader.process_files()
        print(f"\nProcessing completed: {result['processed']} successful, {result['failed']} failed")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Check 'processing.log' for details")