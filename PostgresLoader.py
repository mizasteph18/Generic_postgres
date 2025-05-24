import os
import csv
import time
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch
from tenacity import retry, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)

class AtomicTransactionError(Exception):
    """Atomic operation failure"""

class ColumnValidationError(Exception):
    """Column mapping issue"""

class PostgresLoader:
    def __init__(self, db_config: Dict, input_file: Path, 
                 mapping_file: Path, table_name: str, config: Dict):
        self.db_config = db_config
        self.input_file = Path(input_file)
        self.mapping_file = Path(mapping_file)
        self.table_name = table_name
        self.config = config
        self.progress_file = Path("progress/processing_progress.json")
        
        self._validate_inputs()
        self._load_mapping()

    def _validate_inputs(self):
        """Validate all input paths"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file missing: {self.input_file}")
        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Mapping file missing: {self.mapping_file}")

    def _load_mapping(self):
        """Load and validate mapping file"""
        self.mapping_df = pd.read_csv(self.mapping_file)
        required_columns = ['RawColumn', 'TargetColumn', 'IndexColumn', 'LoadFlag']
        
        if not all(col in self.mapping_df.columns for col in required_columns):
            raise ColumnValidationError("Mapping file missing required columns")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def get_connection(self):
        """Get database connection with retries"""
        return psycopg2.connect(**self.db_config)

    def process_file(self) -> Dict[str, Any]:
        """Process single file with atomic guarantee"""
        start_time = time.time()
        
        if self._is_processed():
            return {
                'status': 'skipped',
                'duration': 0,
                'message': 'File already processed'
            }

        with self.get_connection() as conn:
            conn.autocommit = False
            cursor = conn.cursor()

            try:
                self._process_with_validation(cursor)
                self._update_progress()
                conn.commit()
                
                return {
                    'status': 'success',
                    'duration': time.time() - start_time
                }

            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction rolled back: {e}")
                raise AtomicTransactionError from e
            finally:
                cursor.close()

    def _process_with_validation(self, cursor):
        """Full processing pipeline"""
        # 1. Column validation
        with open(self.input_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            self._validate_columns(header)

        # 2. Create staging table
        cursor.execute(
            sql.SQL("CREATE TEMP TABLE {} (LIKE {} INCLUDING ALL)")
            .format(sql.Identifier('staging'), sql.Identifier(self.table_name))
        )

        # 3. Load data
        with open(self.input_file, 'r') as f:
            reader = csv.DictReader(f)
            self._batch_load(cursor, reader)

        # 4. Merge data
        self._merge_data(cursor)

        # 5. Cleanup
        if self.config.get('delete_files', False):
            self.input_file.unlink()

    def _validate_columns(self, csv_columns: List[str]):
        """Validate CSV columns against mapping"""
        mapped_columns = set(self.mapping_df['RawColumn'])
        new_columns = set(csv_columns) - mapped_columns
        
        if new_columns:
            self._handle_new_columns(new_columns)

    # Remaining methods (_handle_new_columns, _batch_load, _merge_data, 
    # _update_progress, _is_processed) follow previous implementation
    # with progress tracking by input file name + table + mapping
