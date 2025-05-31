import os
import json
import yaml
import pandas as pd
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import execute_values
import logging
import re
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
import shutil

# Constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_BATCH_SIZE = 1000
PROGRESS_FILE = "processing_progress.json"
GLOBAL_CONFIG_FILE = "global_loader_config.yaml"
RESERVED_COLUMNS = {
    "_loaded_timestamp", "_source_filename", 
    "_content_hash", "_operation"
}
HASH_EXCLUDE_COLS = {"_content_hash", "_operation"}

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
    change_detection_strategy: str = "content_hash"  # timestamp|file_size|content_hash
    hash_batch_size: int = 5000
    
@dataclass
class FileProcessingRule:
    """Defines rules for processing specific types of files."""
    base_name: str
    directory: str
    file_pattern: str
    date_format: Optional[str] = None
    start_row: Optional[int] = None
    start_col: Optional[int] = None
    frequency: Optional[str] = None
    mode: Optional[str] = None
    extract_date_col_name: Optional[str] = None
    change_detection: Optional[str] = None
    _compiled_pattern: Any = field(init=False, repr=False)

    def __post_init__(self):
        self._compiled_pattern = re.compile(self.file_pattern)
        self.change_detection = self.change_detection or "content_hash"

    def match(self, filename: str) -> Optional[re.Match]:
        return self._compiled_pattern.match(filename)
    
    @property
    def target_table(self) -> str:
        return self.base_name

    @property
    def mapping_file(self) -> str:
        return f"{self.base_name}_mapping.csv"
    
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.directory:
            errors.append("Directory is required")
        if not self.file_pattern:
            errors.append("File pattern is required")
        if self.mode and self.mode not in ["cancel_and_replace", "audit", "insert"]:
            errors.append(f"Invalid mode: {self.mode}")
        if self.extract_date_col_name and not self.date_format:
            errors.append("date_format is required when extract_date_col_name is specified")
        if self.change_detection not in ["timestamp", "file_size", "content_hash"]:
            errors.append(f"Invalid change detection: {self.change_detection}")
        return len(errors) == 0, errors

@dataclass
class FileContext:
    """Holds file processing context."""
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
    change_detection: str
    file_hash: Optional[str] = None
    file_size: Optional[int] = None

class DatabaseManager:
    """Handles database connections and operations."""
    
    def __init__(self, db_config: Dict[str, Any], config: ProcessingConfig):
        self.db_config = db_config
        self.config = config
        self.connection_pool = None
        self._initialize_pool()
        self._ensure_change_tracking_columns()
    
    def _initialize_pool(self) -> None:
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                self.config.min_connections,
                self.config.max_connections,
                **self.db_config
            )
            logger.info(f"Database connection pool initialized")
        except psycopg2.Error as e:
            logger.critical(f"Failed to initialize connection pool: {e}")
            raise
    
    def _ensure_change_tracking_columns(self) -> None:
        """Ensure all tables have change tracking columns."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(sql.SQL("""
                            ALTER TABLE {} 
                            ADD COLUMN IF NOT EXISTS _content_hash TEXT,
                            ADD COLUMN IF NOT EXISTS _operation CHAR(1)
                        """).format(sql.Identifier(table)))
                        conn.commit()
                        logger.debug(f"Added change tracking columns to {table}")
                    except Exception as e:
                        logger.warning(f"Could not add columns to {table}: {e}")
                        conn.rollback()
    
    @contextmanager
    def get_connection(self):
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
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_latest_timestamp_for_filename(self, target_table: str, source_filename: str) -> Optional[datetime]:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = sql.SQL("""
                        SELECT MAX(_loaded_timestamp) 
                        FROM {} 
                        WHERE _source_filename = %s
                    """).format(sql.Identifier(target_table))
                    cursor.execute(query, (source_filename,))
                    return cursor.fetchone()[0]
        except psycopg2.errors.UndefinedTable:
            return None
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            raise

    def delete_by_source_filename(self, conn, target_table: str, source_filename: str) -> int:
        try:
            with conn.cursor() as cursor:
                query = sql.SQL("DELETE FROM {} WHERE _source_filename = %s").format(sql.Identifier(target_table))
                cursor.execute(query, (source_filename,))
                return cursor.rowcount
        except psycopg2.errors.UndefinedTable:
            return 0
        except Exception as e:
            logger.error(f"Error deleting records: {e}")
            raise

    def file_exists_in_db(self, target_table: str, file_context: FileContext) -> bool:
        """Check if file with same content exists in DB."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Strategy 1: Timestamp comparison
                    if file_context.change_detection == "timestamp":
                        query = sql.SQL("""
                            SELECT 1 FROM {} 
                            WHERE _source_filename = %s 
                            AND ABS(EXTRACT(EPOCH FROM (_loaded_timestamp - %s))) <= %s
                            LIMIT 1
                        """).format(sql.Identifier(target_table))
                        cursor.execute(query, (
                            file_context.filename,
                            file_context.file_modified_timestamp,
                            self.config.timestamp_tolerance_seconds
                        ))
                        return cursor.fetchone() is not None
                    
                    # Strategy 2: File size comparison
                    elif file_context.change_detection == "file_size":
                        query = sql.SQL("""
                            SELECT 1 FROM {} 
                            WHERE _source_filename = %s 
                            AND _file_size = %s
                            LIMIT 1
                        """).format(sql.Identifier(target_table))
                        cursor.execute(query, (file_context.filename, file_context.file_size))
                        return cursor.fetchone() is not None
                    
                    # Strategy 3: Content hash comparison (default)
                    else:
                        query = sql.SQL("""
                            SELECT 1 FROM {} 
                            WHERE _source_filename = %s 
                            AND _content_hash = %s
                            LIMIT 1
                        """).format(sql.Identifier(target_table))
                        cursor.execute(query, (file_context.filename, file_context.file_hash))
                        return cursor.fetchone() is not None
        except psycopg2.errors.UndefinedTable:
            return False
        except Exception as e:
            logger.error(f"File existence check failed: {e}")
            return False

    def get_existing_hashes(self, target_table: str, source_filename: str) -> set:
        """Get existing content hashes for a file."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = sql.SQL("""
                        SELECT _content_hash FROM {}
                        WHERE _source_filename = %s
                    """).format(sql.Identifier(target_table))
                    cursor.execute(query, (source_filename,))
                    return {row[0] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Failed to get existing hashes: {e}")
            return set()

class DataValidator:
    """Handles data validation and quality checks."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def validate_dataframe(self, df: pd.DataFrame, mapping: pd.DataFrame, filename: str) -> Tuple[bool, List[str]]:
        errors = []
        
        if not self.config.enable_data_validation:
            return True, errors
        
        # Check for reserved column names
        reserved_cols = [col for col in RESERVED_COLUMNS if col in df.columns]
        if reserved_cols:
            errors.append(f"Critical: Reserved column names used: {', '.join(reserved_cols)}")
            return False, errors
        
        # Check for missing flagged columns
        required_load_cols = mapping[mapping['LoadFlag'] == 1]['TargetColumn'].values
        missing_required_cols = [col for col in required_load_cols if col not in df.columns]
        if missing_required_cols:
            errors.append(f"Critical: Missing columns: {', '.join(missing_required_cols)}")
            return False, errors
        
        return True, errors

class ProgressTracker:
    """Tracks processing progress for resume capability."""
    
    def __init__(self, progress_file: str = PROGRESS_FILE):
        self.progress_file = progress_file
        self.processed_files = self._load_progress()
    
    def _load_progress(self) -> set:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return set()
    
    def save_progress(self) -> None:
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
                logger.error(f"Could not save progress: {e}")
    
    def mark_processed(self, filepath: Path) -> None:
        self.processed_files.add(str(filepath))
        self.save_progress()
    
    def is_processed(self, filepath: Path) -> bool:
        return str(filepath) in self.processed_files
    
    def get_remaining_files(self, all_file_contexts: List[FileContext]) -> List[FileContext]:
        return [fc for fc in all_file_contexts if not self.is_processed(fc.filepath)]
    
    def reset_progress(self) -> None:
        self.processed_files = set()
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

class FileProcessor:
    """Handles file operations and data extraction."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def load_file(self, file_path: Path, start_row: int = 0, start_col: int = 0) -> pd.DataFrame:
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
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def calculate_row_hashes(self, df: pd.DataFrame) -> List[str]:
        """Calculate content hashes for each row."""
        hashes = []
        for _, row in df.iterrows():
            # Exclude metadata columns from hash calculation
            data = {k: v for k, v in row.items() if k not in HASH_EXCLUDE_COLS}
            hasher = hashlib.sha256()
            hasher.update(json.dumps(data, sort_keys=True).encode('utf-8'))
            hashes.append(hasher.hexdigest())
        return hashes

class PostgresLoader:
    """Main loader class with intelligent change detection."""
    
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
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                return ProcessingConfig(**config_data)
            except Exception as e:
                logger.warning(f"Could not load global config: {e}")
        
        return ProcessingConfig()
    
    def _load_processing_rules(self, rules_folder: str) -> List[FileProcessingRule]:
        rules_folder_path = Path(rules_folder)
        if not rules_folder_path.exists():
            logger.error(f"Rules folder not found: {rules_folder_path}")
            return []
        
        rules = []
        for rule_file_path in rules_folder_path.iterdir():
            if rule_file_path.is_file() and rule_file_path.name.endswith("_rule.yaml"):
                try:
                    base_name = rule_file_path.stem.replace("_rule", "")
                    with open(rule_file_path, 'r') as f:
                        rule_data = yaml.safe_load(f)
                    
                    rule = FileProcessingRule(
                        base_name=base_name,
                        directory=rule_data.get('directory'),
                        file_pattern=rule_data.get('file_pattern'),
                        date_format=rule_data.get('date_format'),
                        start_row=rule_data.get('start_row'),
                        start_col=rule_data.get('start_col'),
                        frequency=rule_data.get('frequency'),
                        mode=rule_data.get('mode', 'insert'),
                        extract_date_col_name=rule_data.get('extract_date_col_name'),
                        change_detection=rule_data.get('change_detection', 'content_hash')
                    )
                    
                    is_valid, errors = rule.validate()
                    if not is_valid:
                        logger.error(f"Invalid rule: {', '.join(errors)}")
                        continue
                    
                    rules.append(rule)
                    logger.info(f"Loaded rule '{base_name}'")
                except Exception as e:
                    logger.error(f"Failed to load rule: {e}")
        
        return rules
    
    def _validate_setup(self) -> None:
        logger.info("Validating setup...")
        
        for rule in self.processing_rules:
            rule_source_dir = Path(rule.directory)
            if not rule_source_dir.exists():
                rule_source_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {rule_source_dir}")
            
            mapping_filepath = rule_source_dir / rule.mapping_file
            
            if not mapping_filepath.exists():
                logger.warning(f"Creating mapping file: {mapping_filepath}")
                sample_file = next(rule_source_dir.glob("*.*"), None)
                
                if sample_file:
                    generate_empty_mapping_file(
                        source_filepath=sample_file,
                        mapping_filepath=mapping_filepath,
                        start_row=rule.start_row or self.global_start_row,
                        start_col=rule.start_col or self.global_start_col
                    )
                else:
                    logger.error(f"No sample file found for {rule.base_name}")

        if not self.db_manager.test_connection():
            raise ConnectionError("Database connection test failed")
        
        logger.info("Setup validation completed")
    
    def get_files_to_process(self) -> List[FileContext]:
        all_potential_file_contexts = []

        for rule in self.processing_rules:
            rule_source_dir = Path(rule.directory)
            if not rule_source_dir.exists():
                continue

            mapping_filepath = rule_source_dir / rule.mapping_file
            if not mapping_filepath.exists():
                continue

            for file_path in rule_source_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                    filename = file_path.name
                    match = rule.match(filename)
                    
                    if match:
                        extracted_timestamp = ""
                        if rule.date_format and match.groups():
                            try:
                                date_str = "".join(match.groups())
                                datetime.strptime(date_str, rule.date_format) 
                                extracted_timestamp = date_str
                            except ValueError:
                                pass
                        
                        file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                        file_size = file_path.stat().st_size
                        file_hash = self.file_processor.calculate_file_hash(file_path) \
                            if rule.change_detection == "content_hash" else None

                        file_context = FileContext(
                            filepath=file_path,
                            filename=filename,
                            target_table=rule.target_table,
                            mapping_filepath=mapping_filepath,
                            extracted_timestamp_str=extracted_timestamp,
                            file_modified_timestamp=file_modified,
                            start_row=rule.start_row or self.global_start_row,
                            start_col=rule.start_col or self.global_start_col,
                            mode=rule.mode or "insert",
                            extract_date_col_name=rule.extract_date_col_name,
                            change_detection=rule.change_detection,
                            file_hash=file_hash,
                            file_size=file_size
                        )
                        all_potential_file_contexts.append(file_context)
        
        files_to_process = []
        for fc in all_potential_file_contexts:
            if self.progress_tracker and self.progress_tracker.is_processed(fc.filepath):
                logger.info(f"Skipping {fc.filename}: Already processed")
                continue
            
            if fc.mode == "audit":
                try:
                    if self.db_manager.file_exists_in_db(fc.target_table, fc):
                        logger.info(f"Skipping {fc.filename}: Content unchanged")
                        if self.progress_tracker:
                            self.progress_tracker.mark_processed(fc.filepath)
                        continue
                except Exception as e:
                    logger.error(f"Audit check failed: {e}")
            
            files_to_process.append(fc)

        logger.info(f"Files to process: {len(files_to_process)}/{len(all_potential_file_contexts)}")
        return files_to_process
    
    def process_files(self) -> Dict[str, int]:
        file_contexts = self.get_files_to_process()
        processed, failed = 0, 0
        
        for fc in file_contexts:
            if self.process_file(fc):
                processed += 1
            else:
                failed += 1
                
        return {"processed": processed, "failed": failed}
    
    def process_file(self, file_context: FileContext) -> bool:
        try:
            logger.info(f"Processing: {file_context.filename} ({file_context.change_detection} strategy)")
            
            df = self.file_processor.load_file(
                file_context.filepath,
                file_context.start_row,
                file_context.start_col
            )
            
            if df.empty:
                logger.warning(f"File is empty: {file_context.filename}")
                if self.progress_tracker: 
                    self.progress_tracker.mark_processed(file_context.filepath)
                return True
            
            mapping = pd.read_csv(file_context.mapping_filepath)
            is_valid, errors = self.validator.validate_dataframe(df, mapping, file_context.filename)
            
            for error in errors:
                logger.error(error)
            if not is_valid:
                return False
            
            # Add metadata columns
            df['_loaded_timestamp'] = file_context.file_modified_timestamp
            df['_source_filename'] = file_context.filename
            df['_operation'] = 'I'  # Default to Insert
            
            # Add extracted date if specified
            if file_context.extract_date_col_name and file_context.extracted_timestamp_str:
                try:
                    extracted_dt = datetime.strptime(
                        file_context.extracted_timestamp_str, 
                        next(rule.date_format for rule in self.processing_rules 
                             if rule.base_name == file_context.target_table)
                    )
                    df[file_context.extract_date_col_name] = extracted_dt
                except Exception:
                    df[file_context.extract_date_col_name] = None
            
            # Calculate content hashes if needed
            if file_context.change_detection == "content_hash":
                df['_content_hash'] = self.file_processor.calculate_row_hashes(df)
            
            # Apply mode-specific processing
            if file_context.mode == "audit":
                success = self._process_audit_mode(df, file_context)
            elif file_context.mode == "cancel_and_replace":
                success = self._process_cancel_replace_mode(df, file_context)
            else:  # insert mode
                success = self._load_data_to_db(df, file_context.target_table)
            
            if success:
                if self.progress_tracker:
                    self.progress_tracker.mark_processed(file_context.filepath)
                
                if self.delete_files:
                    file_context.filepath.unlink()
                    logger.info(f"File deleted: {file_context.filename}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return False
    
    def _process_audit_mode(self, df: pd.DataFrame, file_context: FileContext) -> bool:
        """Process file in audit mode with change detection."""
        if file_context.change_detection == "content_hash":
            # Get existing hashes from DB
            existing_hashes = self.db_manager.get_existing_hashes(
                file_context.target_table, 
                file_context.filename
            )
            
            # Filter to only new/changed rows
            new_rows = df[~df['_content_hash'].isin(existing_hashes)]
            
            if new_rows.empty:
                logger.info(f"No changes detected: {file_context.filename}")
                return True
                
            logger.info(f"Loading {len(new_rows)} changed rows")
            return self._load_data_to_db(new_rows, file_context.target_table)
        else:
            return self._load_data_to_db(df, file_context.target_table)
    
    def _process_cancel_replace_mode(self, df: pd.DataFrame, file_context: FileContext) -> bool:
        """Process file in cancel-and-replace mode with change detection."""
        try:
            with self.db_manager.get_connection() as conn:
                if file_context.change_detection == "content_hash":
                    # Get existing hashes
                    existing_hashes = self.db_manager.get_existing_hashes(
                        file_context.target_table, 
                        file_context.filename
                    )
                    
                    # Identify changes
                    new_hashes = set(df['_content_hash'])
                    to_delete = existing_hashes - new_hashes
                    to_insert = new_hashes - existing_hashes
                    
                    # Delete removed rows
                    if to_delete:
                        with conn.cursor() as cursor:
                            query = sql.SQL("""
                                DELETE FROM {} 
                                WHERE _source_filename = %s 
                                AND _content_hash = ANY(%s)
                            """).format(sql.Identifier(file_context.target_table))
                            cursor.execute(query, (file_context.filename, list(to_delete)))
                            logger.info(f"Deleted {cursor.rowcount} obsolete rows")
                    
                    # Insert new/changed rows
                    if to_insert:
                        new_rows = df[df['_content_hash'].isin(to_insert)]
                        return self._load_data_to_db(new_rows, file_context.target_table, conn)
                    
                    logger.info(f"No changes detected: {file_context.filename}")
                    return True
                else:
                    # Delete existing records
                    deleted_count = self.db_manager.delete_by_source_filename(
                        conn, 
                        file_context.target_table, 
                        file_context.filename
                    )
                    logger.info(f"Deleted {deleted_count} old records")
                    return self._load_data_to_db(df, file_context.target_table, conn)
        except Exception as e:
            logger.error(f"Cancel/replace failed: {e}")
            return False
    
    def _load_data_to_db(self, df: pd.DataFrame, target_table: str, conn=None) -> bool:
        """Load DataFrame to database."""
        try:
            use_external_conn = bool(conn)
            if not conn:
                conn = self.db_manager.get_connection().__enter__()
            
            with conn.cursor() as cursor:
                columns = [c for c in df.columns if not c.startswith("_")]
                placeholders = ", ".join(["%s"] * len(columns))
                
                # Prepare insert query
                query = sql.SQL("""
                    INSERT INTO {} ({})
                    VALUES ({})
                """).format(
                    sql.Identifier(target_table),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(placeholders)
                )
                
                # Insert in batches
                values = [tuple(row) for row in df[columns].itertuples(index=False)]
                for i in range(0, len(values), self.config.batch_size):
                    batch = values[i:i+self.config.batch_size]
                    execute_values(cursor, query, batch)
                
                conn.commit()
                logger.info(f"Loaded {len(df)} rows to {target_table}")
                return True
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            return False
        finally:
            if not use_external_conn and conn:
                conn.close()

def generate_empty_mapping_file(
    source_filepath: Path, 
    mapping_filepath: Path,
    start_row: int = 0, 
    start_col: int = 0,
    sample_rows: int = 100
) -> None:
    try:
        if source_filepath.exists():
            ext = source_filepath.suffix.lower()
            if ext == '.csv':
                df_sample = pd.read_csv(source_filepath, skiprows=range(1, start_row + 1), 
                                       nrows=sample_rows, header=0)
            elif ext in ['.xlsx', '.xls']:
                df_sample = pd.read_excel(source_filepath, skiprows=start_row, 
                                         nrows=sample_rows, header=0)
            else:
                df_sample = pd.DataFrame()
            
            if not df_sample.empty:
                df_sample = df_sample.iloc[:, start_col:]
            columns = df_sample.columns.tolist()
            dtypes = df_sample.dtypes.apply(str).to_dict()
        else:
            columns = []
            dtypes = {}
        
        mapping_data = []
        for col in columns:
            pd_type = dtypes.get(col, 'object')
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
                'RawColumn': col,
                'TargetColumn': col.lower().replace(' ', '_'),
                'DataType': sql_type,
                'LoadFlag': 1,
                'IndexColumn': 'N'
            })
        
        # Add system columns
        for col in RESERVED_COLUMNS:
            mapping_data.append({
                'RawColumn': col,
                'TargetColumn': col,
                'DataType': 'TEXT' if col != '_loaded_timestamp' else 'TIMESTAMP',
                'LoadFlag': 1,
                'IndexColumn': 'N'
            })

        mapping_filepath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mapping_data).to_csv(mapping_filepath, index=False)
    except Exception as e:
        logger.error(f"Mapping generation failed: {e}")
        pd.DataFrame(columns=['RawColumn','TargetColumn','DataType','LoadFlag','IndexColumn']) \
            .to_csv(mapping_filepath, index=False)

def create_sample_configs():
    """Create sample configuration files."""
    os.makedirs("rules", exist_ok=True)
    
    # Global Config
    global_config = {
        "chunk_size": 10000,
        "batch_size": 1000,
        "max_connections": 5,
        "retry_attempts": 3,
        "enable_data_validation": True,
        "timestamp_tolerance_seconds": 1.0,
        "change_detection_strategy": "content_hash",
        "hash_batch_size": 5000
    }
    with open(GLOBAL_CONFIG_FILE, 'w') as f:
        yaml.dump(global_config, f)
    
    # Sales Rule
    sales_rule = {
        "directory": "./sales_data",
        "file_pattern": "^sales_report_(\\d{8})\\.csv$",
        "date_format": "%Y%m%d",
        "mode": "cancel_and_replace",
        "change_detection": "content_hash"
    }
    with open("rules/sales_rule.yaml", 'w') as f:
        yaml.dump(sales_rule, f)
    
    # Inventory Rule
    inventory_rule = {
        "directory": "./inventory_data",
        "file_pattern": "^inventory_(\\d{4}-\\d{2}-\\d{2})\\.xlsx$",
        "date_format": "%Y-%m-%d",
        "mode": "audit",
        "change_detection": "content_hash"
    }
    with open("rules/inventory_rule.yaml", 'w') as f:
        yaml.dump(inventory_rule, f)
    
    print("Sample configuration files created")

if __name__ == "__main__":
    create_sample_configs()
    
    db_config = {
        "dbname": "your_db",
        "user": "your_user",
        "password": "your_password",
        "host": "localhost",
        "port": 5432
    }
    
    try:
        loader = PostgresLoader(
            db_config=db_config,
            global_config_file=GLOBAL_CONFIG_FILE,
            rules_folder_path='rules'
        )
        
        result = loader.process_files()
        print(f"Processed: {result['processed']}, Failed: {result['failed']}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        print("Check processing.log for details")