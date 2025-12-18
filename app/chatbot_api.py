"""
FastAPI Chatbot API for Dell Pro 16 RyzenAI Benchmark Analysis

This module provides a REST API for AI-powered data exploration using Claude.
It generates Python code to query benchmark data and executes it in a secure sandbox.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import anthropic
from dotenv import load_dotenv

from sandbox import DataSandbox

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Dell Pro 16 Benchmark Chatbot API",
    description="AI-powered data exploration for RyzenAI benchmark analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sandbox
# Default to sibling data directory relative to app/
_default_data_dir = str(Path(__file__).parent.parent / "data")
DATA_DIR = os.getenv("DATA_DIR", _default_data_dir)
sandbox = DataSandbox(data_root=DATA_DIR)

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
else:
    client = None
    logger.warning("ANTHROPIC_API_KEY not set - chatbot will be disabled")


# System prompt for Claude
SYSTEM_PROMPT = """You are an AI assistant for analyzing Dell Pro 16 RyzenAI benchmark data.

## Available Data Tables with EXACT Column Names

### Gold Layer (Analytics-Ready)

**df_model_summary** - Per-model performance metrics aggregated across all runs
Columns: model_clean, provider, run_count, latency_mean_ms, latency_mean_std, latency_p50_ms, latency_p95_ms, latency_p99_ms, latency_min_ms, latency_max_ms, throughput_mean_ips, throughput_std_ips, throughput_max_ips, performance_score, efficiency_score, stability_score, total_iters, speedup_vs_cpu

**df_model_summary_by_run** - Per-model metrics PER RUN (for run-to-run comparisons)
Columns: model_clean, provider, run_name, run_count, latency_mean_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms, latency_min_ms, latency_max_ms, throughput_mean_ips, throughput_max_ips, performance_score, efficiency_score, stability_score, total_iters, speedup_vs_cpu

**df_provider_comparison** - Provider-level statistics
Columns: provider, models_tested, total_wins, win_rate_pct, avg_throughput_ips, max_throughput_ips, avg_latency_ms, min_latency_ms, avg_speedup_vs_cpu, max_speedup_vs_cpu, avg_performance_score, avg_efficiency_score, avg_stability_score

**df_reliability** - Success rates by model+provider combination
Columns: model_clean, provider, total_runs, successful_runs, failed_runs, success_rate, is_reliable, error_type, config_key

**df_run_comparison** - Run-to-run performance deltas (run01 vs run02)
Columns: model_clean, provider, config_key, run01_throughput, run01_mean_ms, run01_p99_ms, run01_has_error, run02_throughput, run02_mean_ms, run02_p99_ms, run02_has_error, throughput_delta_pct, improved, latency_delta_pct, performance_change, abs_delta_pct

**df_run_status** - Run-level health summary
Columns: run_name, total_configs, successful_configs, error_count, success_rate, avg_throughput_ips, avg_latency_ms, test_date, models_tested, providers_tested

**df_power_efficiency** - Power and thermal metrics per model+provider
Columns: model_clean, provider, run_count, latency_mean_ms, throughput_mean_ips, speedup_vs_cpu, s_cpu_package_power_mean_w, s_core_powers_avg_mean_w, s_core_0_power_mean_w, inferences_per_watt, energy_per_inference_mj

**df_thermal_profile** - Aggregated thermal behavior by provider
Columns: provider, test_count, avg_s_cpu_package_power_mean_w, max_s_cpu_package_power_mean_w, avg_s_cpu_package_power_max_w, max_s_cpu_package_power_max_w, avg_s_cpu_package_power_p95_w, max_s_cpu_package_power_p95_w, avg_s_page_file_usage_mean_pct, avg_s_page_file_usage_max_pct, avg_s_core_usage_avg_mean_pct

## Benchmark Context

The Dell Pro 16 benchmark tests 16 ONNX AI models on 3 execution providers across 2 benchmark runs:
- **cpu**: CPUExecutionProvider (baseline x86 execution)
- **dml**: DmlExecutionProvider (GPU via DirectML)
- **vitisai**: VitisAIExecutionProvider (AMD NPU acceleration)
- **run01**: First benchmark run
- **run02**: Second benchmark run

Key metrics:
- `throughput_mean_ips`: Inferences per second (higher is better)
- `latency_mean_ms`: Average inference latency in milliseconds (lower is better)
- `latency_p99_ms`: 99th percentile latency (tail latency)
- `speedup_vs_cpu`: Performance multiplier vs CPU baseline
- `success_rate`: Proportion of successful benchmark iterations (0.0 to 1.0)
- `total_runs`, `successful_runs`, `failed_runs`: For reliability analysis

## Real-time FPS Capability

To determine if a model can achieve a target frame rate, check if its P99 latency meets the timing budget:
- **30 FPS**: latency_p99_ms <= 33.33 ms (1000 / 30)
- **60 FPS**: latency_p99_ms <= 16.67 ms (1000 / 60)
- **120 FPS**: latency_p99_ms <= 8.33 ms (1000 / 120)

IMPORTANT: FPS capability is based on LATENCY, not throughput. Use latency_p99_ms for worst-case real-time guarantees.

## Response Format - IMPORTANT

ALWAYS structure your response in this order:
1. **Direct Answer First**: Start with a clear, specific answer. When listing items, name them explicitly:
   - BAD: "9 models achieve 120 FPS"
   - GOOD: "**9 models achieve 120 FPS**: sesr (VitisAI), movenet (DML), resnet50 (VitisAI), efficientnet-es (VitisAI), mnasnet_b1 (VitisAI), sesr (CPU), sesr (DML), SemanticFPN (VitisAI), and yolox-s (VitisAI)."
2. **Key Insight**: One sentence with the most important takeaway (e.g., "VitisAI dominates with 7 of the 9 fastest configurations.")
3. **Code Block**: The Python code that produces the supporting data (will be shown in collapsed expander)

## Code Generation Guidelines - CRITICAL

1. Generate Python code using pandas to answer questions
2. Store the final result in a variable named `result`
3. Use pre-loaded DataFrames (e.g., `df_model_summary`, `df_reliability`)
4. For filtering: `df[df['column'] == value]`
5. For aggregation: `df.groupby('column').agg({})`
6. For sorting: `df.sort_values('column', ascending=False)`
7. Limit results to top 10-20 rows for readability
8. For run-to-run comparisons, use `df_model_summary_by_run` or `df_run_comparison`

**SANDBOX RESTRICTIONS - MUST FOLLOW:**
- **NEVER use print() statements** - the sandbox doesn't support them
- **NEVER use lambda functions** - they cause scope errors in the sandbox
- **NEVER create intermediate variables** - assign directly to `result` in a single chain
- **NEVER use apply() with custom functions** - use built-in pandas methods only
- **ALWAYS write single-line or simple chained operations**

BAD (will fail):
```python
filtered_data = df_model_summary[df_model_summary['provider'] == 'vitisai']
result = filtered_data.groupby('model_clean').apply(lambda x: x.mean())
```

GOOD (will work):
```python
result = df_model_summary[df_model_summary['provider'] == 'vitisai'][['model_clean', 'throughput_mean_ips', 'latency_mean_ms']].sort_values('throughput_mean_ips', ascending=False)
```

## Example Responses

Q: "Which provider has the best throughput?"
A: **VitisAI has the best throughput** with an average of 285 ips, followed by DML (180 ips) and CPU (95 ips). VitisAI wins on 16 of 19 models tested.
```python
result = df_provider_comparison[['provider', 'avg_throughput_ips', 'total_wins']].sort_values('avg_throughput_ips', ascending=False)
```

Q: "What models achieve 120 FPS?"
A: **9 model-provider combinations achieve 120 FPS**: sesr (VitisAI, CPU, DML), movenet (DML), resnet50 (VitisAI), efficientnet-es (VitisAI), mnasnet_b1 (VitisAI), SemanticFPN (VitisAI), and yolox-s (VitisAI). VitisAI dominates with 7 of the 9 fastest configurations.
```python
result = df_model_summary[df_model_summary['latency_p99_ms'] <= 8.33][['model_clean', 'provider', 'latency_p99_ms', 'throughput_mean_ips']].sort_values('latency_p99_ms')
```

Q: "What's the fastest model?"
A: **sesr on VitisAI is the fastest** with 1187 ips throughput and just 1.2ms P99 latency. This super-resolution model is highly optimized for the NPU.
```python
result = df_model_summary[['model_clean', 'provider', 'throughput_mean_ips', 'latency_p99_ms']].sort_values('throughput_mean_ips', ascending=False).head(10)
```

Remember: Always name specific models/providers in your answer. Never use print() - only assign to `result`."""


# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = None
    dashboard_context: Optional[Dict[str, Any]] = None


class CodeResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    result_type: Optional[str] = None
    preview: Optional[str] = None
    error: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    executed_code: Optional[str] = None
    code_result: Optional[CodeResult] = None


# Helper functions
def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from markdown code blocks."""
    import re

    # Look for python code blocks
    pattern = r'```python\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Try generic code blocks
    pattern = r'```\n(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        return matches[0].strip()

    return None


def format_result_preview(result: Any, max_rows: int = 10) -> str:
    """Format result for preview display."""
    if result is None:
        return "No result"

    if isinstance(result, pd.DataFrame):
        if len(result) > max_rows:
            return result.head(max_rows).to_string() + f"\n... ({len(result)} total rows)"
        return result.to_string()

    if isinstance(result, pd.Series):
        if len(result) > max_rows:
            return result.head(max_rows).to_string() + f"\n... ({len(result)} total items)"
        return result.to_string()

    return str(result)[:1000]


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "chatbot_enabled": client is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat message and return response with optional code execution."""

    if not client:
        raise HTTPException(
            status_code=503,
            detail="Chatbot unavailable - ANTHROPIC_API_KEY not configured"
        )

    try:
        # Build messages for Claude
        messages = []

        # Add conversation history if provided
        if request.conversation_history:
            for msg in request.conversation_history[-10:]:  # Limit history
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Add current message
        messages.append({
            "role": "user",
            "content": request.message
        })

        # Call Claude API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        assistant_response = response.content[0].text

        # Try to extract and execute code
        code = extract_code_from_response(assistant_response)
        code_result = None

        if code:
            logger.info(f"Executing code: {code[:100]}...")
            exec_result = sandbox.execute(code)

            if exec_result['success']:
                result = exec_result['result']
                code_result = CodeResult(
                    success=True,
                    result_type=type(result).__name__ if result is not None else None,
                    preview=format_result_preview(result)
                )
            else:
                code_result = CodeResult(
                    success=False,
                    error=exec_result['error']
                )

        return ChatResponse(
            response=assistant_response,
            executed_code=code,
            code_result=code_result
        )

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        raise HTTPException(status_code=502, detail=f"AI service error: {str(e)}")

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables")
async def list_tables():
    """List all available data tables."""
    return sandbox.get_available_tables()


@app.get("/table/{layer}/{table_name}")
async def get_table_info(layer: str, table_name: str):
    """Get metadata about a specific table."""
    info = sandbox.get_table_info(layer, table_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Table not found: {layer}/{table_name}")
    return info


@app.post("/execute")
async def execute_code(code: str):
    """Execute Python code in sandbox (for testing)."""
    result = sandbox.execute(code)

    if result['success'] and isinstance(result['result'], pd.DataFrame):
        result['result'] = result['result'].to_dict('records')

    return result


# Main entry point
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("CHATBOT_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
