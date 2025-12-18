"""
Sandboxed Python execution environment for chatbot data queries.

This module provides secure, read-only access to the medallion architecture
data (Bronze/Silver/Gold layers) through sandboxed Python code execution.

Adapted for Dell Pro 16 RyzenAI benchmark data.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from io import StringIO
import signal
import pandas as pd
import numpy as np

from RestrictedPython import compile_restricted, safe_globals, limited_builtins
from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr


class TimeoutException(Exception):
    """Raised when code execution exceeds timeout limit."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException("Code execution timed out")


class DataSandbox:
    """
    Secure sandbox environment for executing Python code to query benchmark data.

    Features:
    - Read-only access to Parquet files
    - Restricted imports (only pandas, numpy)
    - No file writes, network access, or system calls
    - Execution timeout limits
    - Result size limits
    """

    def __init__(self, data_root: str = "./data", timeout: int = 10):
        """
        Initialize the sandbox environment.

        Args:
            data_root: Root directory containing data folders
            timeout: Maximum execution time in seconds
        """
        self.data_root = Path(data_root).resolve()
        self.timeout = timeout
        self.max_result_rows = 1000

        # Verify data directories exist
        self.bronze_path = self.data_root / "bronze"
        self.silver_path = self.data_root / "silver"
        self.gold_path = self.data_root / "gold"

    def _safe_import(self, name, *args, **kwargs):
        """Restricted import function - only allows specific modules."""
        allowed_modules = {
            'pandas': pd,
            'numpy': np,
            'pd': pd,
            'np': np,
        }

        if name in allowed_modules:
            return allowed_modules[name]

        raise ImportError(f"Import of '{name}' is not allowed in sandbox")

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for code execution."""

        restricted_globals = {
            '__builtins__': {
                **limited_builtins,
                '__import__': self._safe_import,
                '__name__': 'sandbox',
                '__metaclass__': type,
                # Add safe built-in functions
                'round': round,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
            },
            '_getitem_': default_guarded_getitem,
            '_getiter_': default_guarded_getiter,
            '_iter_unpack_sequence_': guarded_iter_unpack_sequence,
            '_getattr_': safer_getattr,
            '_print_': lambda x: None,
            '_write_': lambda x: x,
            'write': lambda x: x,
        }

        # Add safe data loading functions
        restricted_globals.update({
            'load_gold': self._load_gold,
            'load_silver': self._load_silver,
            'load_bronze': self._load_bronze,
            'pd': pd,
            'pandas': pd,
            'np': np,
            'numpy': np,
        })

        # Pre-load common DataFrames
        preloaded_data = self._preload_common_dataframes()
        restricted_globals.update(preloaded_data)

        return restricted_globals

    def _preload_common_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Pre-load commonly used DataFrames into the execution environment.
        """
        preloaded = {}

        # Define common variable names and their source files
        dataframe_mappings = {
            # Gold layer - analytics-ready data
            'df_model_summary': ('gold', 'model_summary'),
            'df_model_summary_by_run': ('gold', 'model_summary_by_run'),
            'df_provider_comparison': ('gold', 'provider_comparison'),
            'df_reliability': ('gold', 'reliability_analysis'),
            'df_run_comparison': ('gold', 'run_comparison'),
            'df_error_root_causes': ('gold', 'error_root_causes'),
            'df_run_status': ('gold', 'run_status_summary'),
            'df_power_efficiency': ('gold', 'power_efficiency'),
            'df_thermal_profile': ('gold', 'provider_thermal_profile'),
            # Silver layer - transformed data
            'df_run_metrics': ('silver', 'run_metrics'),
            'df_error_tracking': ('silver', 'error_tracking'),
            'df_perf_comparison': ('silver', 'performance_comparison'),
            'df_sensor_summary': ('silver', 'sensor_summary'),
            # Bronze layer - raw data
            'df_master_raw': ('bronze', 'master_scores'),
            'df_model_scores_raw': ('bronze', 'model_scores'),
        }

        for var_name, (layer, filename) in dataframe_mappings.items():
            try:
                if layer == 'gold':
                    preloaded[var_name] = self._load_gold(filename)
                elif layer == 'silver':
                    preloaded[var_name] = self._load_silver(filename)
                elif layer == 'bronze':
                    preloaded[var_name] = self._load_bronze(filename)
            except (ValueError, FileNotFoundError):
                pass
            except Exception:
                pass

        return preloaded

    def _load_gold(self, filename: str) -> pd.DataFrame:
        """Load a Parquet file from the Gold layer."""
        return self._load_parquet(self.gold_path / filename)

    def _load_silver(self, filename: str) -> pd.DataFrame:
        """Load a Parquet file from the Silver layer."""
        return self._load_parquet(self.silver_path / filename)

    def _load_bronze(self, filename: str) -> pd.DataFrame:
        """Load a Parquet file from the Bronze layer."""
        return self._load_parquet(self.bronze_path / filename)

    def _load_parquet(self, filepath: Path) -> pd.DataFrame:
        """
        Safely load a Parquet file.
        """
        filepath = filepath.resolve()

        # Security check: ensure file is within data root
        if not str(filepath).startswith(str(self.data_root)):
            raise ValueError(f"Access denied: File outside data root")

        # Add .parquet extension if not present
        if not str(filepath).endswith('.parquet'):
            filepath = Path(str(filepath) + '.parquet')

        if not filepath.exists():
            raise ValueError(f"File not found: {filepath.name}")

        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {str(e)}")

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in the sandbox environment.

        Args:
            code: Python code string to execute

        Returns:
            Dictionary with execution results
        """
        # Set up timeout handler (Unix-like systems only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Compile code with RestrictedPython
            compile_result = compile_restricted(
                code,
                filename='<sandbox>',
                mode='exec'
            )

            # Check for compilation errors
            if isinstance(compile_result, tuple):
                byte_code, errors, warnings, used_names = compile_result
                if errors:
                    return {
                        'success': False,
                        'result': None,
                        'output': '',
                        'error': f"Compilation errors: {', '.join(errors)}"
                    }
            else:
                byte_code = compile_result

            # Create restricted execution environment
            restricted_globals = self._create_safe_globals()
            restricted_locals = {}

            # Execute the code
            exec(byte_code, restricted_globals, restricted_locals)

            # Get result
            result = restricted_locals.get('result', None)

            # If result is a DataFrame, limit rows
            if isinstance(result, pd.DataFrame) and len(result) > self.max_result_rows:
                result = result.head(self.max_result_rows)
                output_msg = f"\nNote: Result truncated to {self.max_result_rows} rows"
            else:
                output_msg = ""

            output = captured_output.getvalue() + output_msg

            return {
                'success': True,
                'result': result,
                'output': output,
                'error': None
            }

        except TimeoutException as e:
            return {
                'success': False,
                'result': None,
                'output': captured_output.getvalue(),
                'error': f"Execution timed out after {self.timeout} seconds"
            }

        except Exception as e:
            return {
                'success': False,
                'result': None,
                'output': captured_output.getvalue(),
                'error': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }

        finally:
            sys.stdout = old_stdout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

    def get_available_tables(self) -> Dict[str, list]:
        """Get list of available Parquet files in each layer."""
        tables = {}

        for layer_name, layer_path in [
            ('gold', self.gold_path),
            ('silver', self.silver_path),
            ('bronze', self.bronze_path)
        ]:
            if layer_path.exists():
                tables[layer_name] = [
                    f.stem for f in layer_path.glob('*.parquet')
                ]
            else:
                tables[layer_name] = []

        return tables

    def get_table_info(self, layer: str, table_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific table."""
        layer_paths = {
            'gold': self.gold_path,
            'silver': self.silver_path,
            'bronze': self.bronze_path
        }

        if layer not in layer_paths:
            return None

        filepath = layer_paths[layer] / f"{table_name}.parquet"

        if not filepath.exists():
            return None

        try:
            df = pd.read_parquet(filepath)
            return {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'row_count': len(df),
                'memory_usage': df.memory_usage(deep=True).sum(),
            }
        except Exception as e:
            return {'error': str(e)}


def test_sandbox():
    """Test the sandbox environment."""
    sandbox = DataSandbox()

    print("Test 1: Load gold table")
    code1 = """
result = df_model_summary.head(5)[['model_clean', 'provider', 'throughput_mean_ips']]
"""
    result1 = sandbox.execute(code1)
    print(f"Success: {result1['success']}")
    if result1['success']:
        print(result1['result'])
    else:
        print(f"Error: {result1['error']}")

    print("\n\nTest 2: Try forbidden import")
    code2 = """
import os
result = os.listdir('.')
"""
    result2 = sandbox.execute(code2)
    print(f"Success: {result2['success']}")
    print(f"Error: {result2['error']}")

    print("\n\nTest 3: Available tables")
    print(sandbox.get_available_tables())


if __name__ == '__main__':
    test_sandbox()
