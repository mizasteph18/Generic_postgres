#!/usr/bin/env python3
"""
Flexible PostgreSQL Data Loader
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

sys.path.append(str(Path(__file__).parent))

from postgres_loader import PostgresLoader, AtomicTransactionError, ColumnValidationError

def load_config(config_file: str) -> Dict[str, Any]:
    """Load processing configuration"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str = None):
    """Configure timestamped logging"""
    log_format = '%(asctime)s.%(msecs)03dZ - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%dT%H:%M:%S'
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    else:
        handlers = [logging.StreamHandler()]
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()

def main():
    parser = argparse.ArgumentParser(description='Flexible Data Loader')
    parser.add_argument('--input-file', required=True, help='Input file path')
    parser.add_argument('--mapping-file', required=True, help='Mapping CSV path')
    parser.add_argument('--table-name', required=True, help='Target table name')
    parser.add_argument('--log-file', help='Custom log file path')
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Disable column prompts')
    args = parser.parse_args()

    try:
        # Initialize logging
        setup_logging(args.log_file)
        logger = logging.getLogger(__name__)
        
        # Load configurations
        config = load_config(Path('config/loader_config.yaml'))
        config['processing']['interactive_mode'] = not args.non_interactive

        # Initialize loader
        loader = PostgresLoader(
            db_config={
                'dbname': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'),
                'sslmode': os.getenv('DB_SSLMODE', 'require')
            },
            input_file=args.input_file,
            mapping_file=args.mapping_file,
            table_name=args.table_name,
            config=config['processing']
        )

        # Process file
        result = loader.process_file()
        logger.info(f"Processing complete: {result['status']}")
        logger.info(f"Duration: {result['duration']:.2f}s")

    except ColumnValidationError as e:
        logger.error(f"Column validation failed: {e}")
        sys.exit(1)
    except AtomicTransactionError as e:
        logger.error(f"Transaction failed: {e}")
        sys.exit(2)
    except Exception as e:
        logger.exception("Critical error occurred")
        sys.exit(3)

if __name__ == "__main__":
    main()
