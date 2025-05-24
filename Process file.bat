@echo off
REM Flexible data loader batch file
SET PY_SCRIPT="main_loader.py"
SET LOG_DIR="logs"

REM Process different file types with custom mappings
python %PY_SCRIPT% ^
  --input-file "data\sales\january_2024.csv" ^
  --mapping-file "mappings\sales_map.csv" ^
  --table-name monthly_sales ^
  --log-file %LOG_DIR%\sales_jan.log

python %PY_SCRIPT% ^
  --input-file "data\inventory\warehouse_stock_q1.csv" ^
  --mapping-file "mappings\inventory_map.csv" ^
  --table-name inventory ^
  --log-file %LOG_DIR%\inventory_q1.log

python %PY_SCRIPT% ^
  --input-file "data\customers\contacts.csv" ^
  --mapping-file "mappings\customer_map.csv" ^
  --table-name customer_base ^
  --log-file %LOG_DIR%\customer_import.log

pause
