"""
Local proxy server replacing ccx.
Listens on localhost, translates Responses API <-> Chat Completions,
and forwards to multiple LLM providers with HTTPS encryption.
"""

import json
import logging
import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Tuple
from logging.handlers import RotatingFileHandler

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ---- Crash Protection ----

def safe_create_task(coro):
    async def _wrapper():
        try:
            await coro
        except Exception:
            log.exception("Background task failed")
    try:
        asyncio.create_task(_wrapper())
    except RuntimeError:
        log.warning("No running event loop")

def _global_exc_handler(loop, context):
    msg = context.get("message", "Unknown")
    exc = context.get("exception")
    if exc:
        log.error(f"Asyncio error: {msg} - {exc}")
    else:
        log.error(f"Asyncio error: {msg}")

from translator import (
    _has_image_content,
    responses_to_chat,
    chat_to_responses,
    chat_error_to_responses_error,
    StreamTranslator,
    _gen_id,
)

# ---- Config ----

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def load_config():
    with open(CONFIG_PATH, encoding="utf-8-sig") as f:
        return json.load(f)

def save_config(cfg):
    tmp_path = CONFIG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, CONFIG_PATH)

def reload_globals():
    global PORT, DEFAULT_MODEL, MODEL_MAP, PROVIDERS, MAX_RETRIES, LOG_LEVEL, DEFAULT_PROVIDER, VISION_MODEL
    cfg = load_config()
    PORT = cfg.get("port", 2000)
    DEFAULT_MODEL = cfg.get("default_model", "deepseek-chat")
    VISION_MODEL = cfg.get("vision_model", "")
    MODEL_MAP = cfg.get("model_map", {})
    PROVIDERS = cfg.get("providers", {})
    MAX_RETRIES = cfg.get("max_retries", 3)
    LOG_LEVEL = cfg.get("log_level", "INFO")
    DEFAULT_PROVIDER = cfg.get("default_provider", "deepseek")
    # Update logger level without restart
    logging.getLogger("local-proxy").setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

reload_globals()


def resolve_model(model_name: str) -> Tuple[str, str, str, str, str]:
    """
    Resolve a model name to its provider details.
    Returns: (provider_name, actual_model, base_url, api_key, api_path)
    """
    mapping = MODEL_MAP.get(model_name)
    if mapping is None:
        provider = PROVIDERS.get(DEFAULT_PROVIDER, {})
        if not provider:
            raise ValueError(f"Default provider '{DEFAULT_PROVIDER}' not found")
        base_url = provider.get("base_url", "").rstrip("/")
        # Remove trailing /v1 if present to avoid double /v1
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        api_key = provider.get("api_key") or provider.get("api_key_masked", "")
        api_path = provider.get("api_path", "/v1/chat/completions")
        if not base_url:
            raise ValueError(f"Provider '{DEFAULT_PROVIDER}' has no base_url configured")
        log.debug(f"Model '{model_name}' routed to default provider '{DEFAULT_PROVIDER}'")
        return DEFAULT_PROVIDER, model_name, base_url, api_key, api_path

    # Support both new format (dict) and old format (string)
    if isinstance(mapping, str):
        if DEFAULT_PROVIDER and DEFAULT_PROVIDER in PROVIDERS:
            provider_name = DEFAULT_PROVIDER
        elif PROVIDERS:
            provider_name = list(PROVIDERS.keys())[0]
        else:
            raise ValueError(f"No providers configured for model '{model_name}'")
        actual_model = mapping
    else:
        provider_name = mapping.get("provider", "")
        actual_model = mapping.get("model", model_name)

    provider = PROVIDERS.get(provider_name, {})
    base_url = provider.get("base_url", "").rstrip("/")
    # Remove trailing /v1 if present to avoid double /v1
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    api_key = provider.get("api_key") or provider.get("api_key_masked", "")
    api_path = provider.get("api_path", "/v1/chat/completions")

    if not base_url:
        raise ValueError(f"Provider '{provider_name}' has no base_url configured")
    if not api_key:
        log.warning(f"Provider '{provider_name}' has no API key configured")

    return provider_name, actual_model, base_url, api_key, api_path



# ---- Logging ----

log = logging.getLogger("local-proxy")
log.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# Avoid duplicate handlers on reload
if log.handlers:
    log.handlers.clear()

# Stream handler
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
log.addHandler(_stream_handler)

# Rotating file handler (10MB per file, keep 5 backups)
_file_handler = RotatingFileHandler(
    os.path.join(BASE_DIR, "proxy.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
log.addHandler(_file_handler)
# ---- Token Stats ----

_token_stats = {
    "total_input": 0,
    "total_output": 0,
    "requests": [],
    "by_model": {}  # model_name -> {input: N, output: N, requests: N}
}
_token_lock = asyncio.Lock()

# ---- Response Cache (for previous_response_id support) ----
_response_cache = {}  # response_id -> response output items
_response_cache_lock = asyncio.Lock()
MAX_CACHED_RESPONSES = 100


async def cache_response(response_id: str, output_items: list):
    """Cache response output for previous_response_id chaining."""
    async with _response_cache_lock:
        if len(_response_cache) >= MAX_CACHED_RESPONSES:
            # Remove oldest (simple FIFO)
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[response_id] = output_items


async def get_cached_response(response_id: str) -> list:
    """Get cached response output by ID."""
    async with _response_cache_lock:
        return _response_cache.get(response_id, None)

def _write_usage_log(entry: dict):
    """Write usage entry to log file (runs in thread)."""
    log_path = os.path.join(BASE_DIR, "usage.log")

    # Rotate if file exceeds 10MB
    if os.path.exists(log_path) and os.path.getsize(log_path) > 10 * 1024 * 1024:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.rename(log_path, f"usage.log.{timestamp}")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def record_usage(model: str, input_tokens: int, output_tokens: int):
    """Record token usage and persist to usage.log."""
    async with _token_lock:
        _token_stats["total_input"] += input_tokens
        _token_stats["total_output"] += output_tokens

        # Per-model stats
        if model not in _token_stats["by_model"]:
            _token_stats["by_model"][model] = {"input": 0, "output": 0, "requests": 0}
        _token_stats["by_model"][model]["input"] += input_tokens
        _token_stats["by_model"][model]["output"] += output_tokens
        _token_stats["by_model"][model]["requests"] += 1

        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "input": input_tokens,
            "output": output_tokens
        }
        _token_stats["requests"].append(entry)
        # Keep last 200 requests in memory
        if len(_token_stats["requests"]) > 200:
            _token_stats["requests"] = _token_stats["requests"][-200:]
        # Append to usage.log asynchronously to avoid blocking event loop
        await asyncio.to_thread(_write_usage_log, entry)


# ---- HTTP Client ----

http_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=300.0,
                write=60.0,
                pool=30.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
            ),
        )
    return http_client


# ---- FastAPI App ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"Local proxy starting on port {PORT}")
    log.info(f"   Providers: {list(PROVIDERS.keys())}")
    log.info(f"   Default model: {DEFAULT_MODEL}")
    log.info(f"   Web UI: http://localhost:{PORT}/")
    yield
    log.info("Shutting down proxy...")
    client = get_client()
    await client.aclose()


app = FastAPI(title="Local Proxy", version="2.0.0", lifespan=lifespan)

# Static Files (Web UI)
STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return Response("<h1>Local Proxy Running</h1><p>Web UI not found.</p>", media_type="text/html")

# Config API
@app.get("/api/config")
async def get_config_api():
    cfg = load_config()
    # Mask all api_keys
    providers = cfg.get("providers", {})
    masked_providers = {}
    for name, p in providers.items():
        mp = dict(p)
        key = mp.get("api_key", "")
        if key:
            if len(key) > 11:
                mp["api_key_masked"] = key[:7] + "..." + key[-4:]
            else:
                mp["api_key_masked"] = "***"
        else:
            mp["api_key_masked"] = "unset"
        mp.pop("api_key", None)
        masked_providers[name] = mp
    cfg["providers"] = masked_providers
    return cfg

@app.post("/api/config")
async def update_config_api(request: Request):
    try:
        body = await request.json()
        cfg = load_config()
        allowed = ["port", "default_model", "model_map", "providers",
                    "max_retries", "log_level"]
        for key in allowed:
            if key not in body:
                continue
            if key in ("model_map",):
                if not isinstance(body[key], dict):
                    continue
                # Replace entirely (not merge) so deletions work
                cfg[key] = body[key]
            elif key == "providers":
                if not isinstance(body[key], dict):
                    continue
                # Direct replacement: use request body as new providers list
                # But preserve api_keys that are masked in the request
                new_providers = {}
                for name, new_p in body[key].items():
                    if not isinstance(new_p, dict):
                        continue
                    old_p = cfg.get("providers", {}).get(name, {})
                    merged = {}

                    # Start with old provider's api_key (if exists)
                    if "api_key" in old_p:
                        merged["api_key"] = old_p["api_key"]

                    # Override with new values
                    for k, v in new_p.items():
                        if k == "api_key":
                            if "..." in str(v) or not v or v == "unset":
                                # Masked or empty, keep old
                                continue
                            else:
                                merged["api_key"] = v
                        elif k == "api_key_masked":
                            if "..." not in str(v) and v and v != "unset":
                                merged["api_key"] = v
                            # Don't copy api_key_masked to config
                        else:
                            merged[k] = v

                    # Only add provider if it has at least base_url or api_key
                    if merged:
                        new_providers[name] = merged

                cfg["providers"] = new_providers
            else:
                cfg[key] = body[key]
        save_config(cfg)
        reload_globals()
        logging.getLogger("local-proxy").info(f"Config updated: {list(body.keys())}")
        return await get_config_api()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# Stats API
@app.get("/api/stats")
async def get_stats():
    return _token_stats


# ---- Test Provider Connection ----
@app.post("/api/providers/test")
async def test_provider(request: Request):
    try:
        body = await request.json()
        provider_name = body.get("provider_name", "")
        base_url = body.get("base_url", "").rstrip("/")
        api_key = body.get("api_key", "")
        api_path = body.get("api_path", "/v1/chat/completions")

        # If provider_name is provided, load config from file (to get real api_key)
        if provider_name and not api_key:
            cfg = load_config()
            prov = cfg.get("providers", {}).get(provider_name, {})
            if prov:
                base_url = prov.get("base_url", base_url)
                api_key = prov.get("api_key", "")
                api_path = prov.get("api_path", api_path)

        if not base_url:
            return JSONResponse(status_code=400, content={"error": "base_url is required"})

        # Remove trailing /v1 if present to avoid double /v1
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Test connection using chat endpoint (more reliable than models endpoint)
        chat_url = f"{base_url}{api_path}"
        client = get_client()
        try:
            # Send a minimal chat request to test connection
            chat_resp = await client.post(
                chat_url,
                headers=headers,
                json={"model": "test", "messages": [{"role": "user", "content": "test"}], "max_tokens": 1, "stream": False},
                timeout=15.0
            )

            # Check response
            # 401/403 = invalid api key
            # 400/422 = connection ok but invalid request (expected since we used "test" as model)
            # Other 2xx/4xx = connection successful
            if chat_resp.status_code in (401, 403):
                return JSONResponse(
                    status_code=chat_resp.status_code,
                    content={"error": "Invalid API key", "detail": chat_resp.text[:500]}
                )

            # Connection successful! Try to get models if available
            models = []
            try:
                models_url = f"{base_url}/v1/models"
                models_resp = await client.get(models_url, headers=headers, timeout=5.0)
                if models_resp.status_code < 400:
                    models_data = models_resp.json()
                    if "data" in models_data and isinstance(models_data["data"], list):
                        for m in models_data["data"]:
                            if isinstance(m, dict) and "id" in m:
                                models.append({
                                    "id": m["id"],
                                    "object": m.get("object", "model"),
                                    "owned_by": m.get("owned_by", "unknown")
                                })
            except Exception:
                pass  # Models endpoint not available, that's ok

            return {
                "status": "ok",
                "models": models,
                "warning": None if models else "Models endpoint not available, but API connection successful"
            }

        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content={"error": "Connection timeout"})
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": f"Connection failed: {str(e)}"})

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ---- Health Check ----

@app.get("/health")
async def health():
    return {"status": "ok", "providers": list(PROVIDERS.keys())}


# ---- Model List ----

@app.get("/v1/models")
async def list_models():
    models = []
    for display_name, mapping in MODEL_MAP.items():
        if isinstance(mapping, dict):
            provider = mapping.get("provider", "unknown")
            ctx = mapping.get("context_window")
        else:
            provider = list(PROVIDERS.keys())[0] if PROVIDERS else "unknown"
            ctx = None
        entry = {
            "id": display_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": provider
        }
        if ctx:
            entry["context_window"] = ctx
            entry["max_context_length"] = ctx
        models.append(entry)
    return {"object": "list", "data": models}


# ---- Switch Model ----

@app.post("/switch")
async def switch_model(request: Request):
    try:
        body = await request.json()
        new_model = body.get("model", "")
        if new_model in MODEL_MAP:
            cfg = load_config()
            cfg["default_model"] = new_model
            save_config(cfg)
            reload_globals()
            log.info(f"Switched model to: {new_model}")
            return {"status": "ok", "model": new_model}
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown model: {new_model}", "available": list(MODEL_MAP.keys())}
            )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/switch")
async def get_current_model():
    return {"model": DEFAULT_MODEL, "available": list(MODEL_MAP.keys())}


# ---- Main Proxy: /v1/{path} ----

@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_handler(request: Request, path: str):
    request_id = _gen_id("req")
    method = request.method
    body = None
    
    try:
        body = await request.body()
    except Exception:
        body = b""
    
    body_json = None
    if body:
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError:
            body_json = None
    
    # Log incoming request for debugging
    if path == "responses" and method == "POST" and body_json:
        log.debug(f"[{request_id}] Incoming request: model={body_json.get('model')}, stream={body_json.get('stream')}")
        # Log the full request for debugging (truncate if too long)
        req_str = json.dumps(body_json, ensure_ascii=False)
        if len(req_str) < 1000:
            log.debug(f"[{request_id}] Full request: {req_str}")
        return await handle_responses(request_id, body_json, request.headers)
    
    # Generic pass-through
    return await handle_pass_through(request_id, method, path, body, request.headers, body_json)
# ---- Short path: /responses (without /v1/ prefix) ----
@app.post("/responses")
async def responses_short(request: Request):
    """Handle /responses directly, some Codex configs omit /v1/ prefix."""
    request_id = _gen_id("req")
    body = await request.body()
    body_json = json.loads(body) if body else None
    if not body_json:
        return JSONResponse(status_code=400, content={"error": {"message": "Empty body"}})
    return await handle_responses(request_id, body_json, request.headers)


# ---- Short path catch-all: /{path} (without /v1/ prefix) ----
@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def direct_handler(request: Request, path: str):
    """Handle requests without /v1/ prefix (e.g. /chat/completions)."""
    # Skip paths already handled by specific routes
    if path in ("", "health", "switch", "api/config", "responses", "favicon.ico"):
        return JSONResponse(status_code=404, content={"error": {"message": "Not found"}})
    if path.startswith("static/") or path.startswith("api/"):
        return JSONResponse(status_code=404, content={"error": {"message": "Not found"}})
    request_id = _gen_id("req")
    method = request.method
    body = await request.body()
    body_json = None
    if body:
        try:
            body_json = json.loads(body)
        except json.JSONDecodeError:
            log.warning(f"[{request_id}] Invalid JSON in request body")
    return await handle_pass_through(request_id, method, path, body, request.headers, body_json)


async def handle_responses(
    request_id: str,
    body_json: dict,
    request_headers: dict,
):
    is_stream = body_json.get("stream", False)
    request_model = body_json.get("model", DEFAULT_MODEL)
    
    # Handle previous_response_id for conversation chaining
    previous_response_id = body_json.get("previous_response_id")
    if previous_response_id:
        cached_output = await get_cached_response(previous_response_id)
        if cached_output:
            # Convert cached output to input items and prepend to current input
            previous_input = []
            for item in cached_output:
                if item.get("type") == "message":
                    # Convert message output to input format
                    content = item.get("content", [])
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text_parts.append(part.get("text", ""))
                    if text_parts:
                        previous_input.append({
                            "role": "assistant",
                            "content": "\n".join(text_parts)
                        })
                elif item.get("type") == "function_call":
                    # Keep function_call items as-is
                    previous_input.append(item)
            
            # Prepend previous input to current input
            current_input = body_json.get("input", [])
            body_json["input"] = previous_input + current_input
            log.info(f"[{request_id}] Using previous_response_id={previous_response_id}, added {len(previous_input)} items")
        else:
            log.warning(f"[{request_id}] previous_response_id={previous_response_id} not found in cache")
    
    # Resolve provider
    try:
        provider_name, actual_model, base_url, api_key, api_path = resolve_model(request_model)
    except ValueError as e:
        log.error(f"[{request_id}] Model resolution failed: {e}")
        return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "invalid_request_error"}})

    # Build model_map with only the resolved actual_model
    single_map = {request_model: actual_model}

    # Auto-switch to vision model if request contains images
    if _has_image_content(body_json):
        vision_cfg = VISION_MODEL or ""
        if not vision_cfg:
            mm = MODEL_MAP.get(request_model, {})
            if isinstance(mm, dict):
                vision_cfg = mm.get("vision_model", "")
        if vision_cfg and vision_cfg != request_model:
            log.info(f"[{request_id}] Images detected, switching {request_model} -> {vision_cfg}")
            try:
                provider_name, actual_model, base_url, api_key, api_path = resolve_model(vision_cfg)
                single_map = {vision_cfg: actual_model}
                request_model = vision_cfg
            except ValueError as e:
                log.warning(f"[{request_id}] Vision model {vision_cfg} failed: {e}")

    try:
        chat_request = responses_to_chat(body_json, single_map)
        log.debug(f"[{request_id}] Translated request: {json.dumps(chat_request, ensure_ascii=False)[:500]}")
    except Exception as e:
        log.error(f"[{request_id}] Failed to translate request: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Request translation failed: {e}", "type": "invalid_request_error"}}
        )

    # Remove reasoning_effort if provider doesn't support it
    provider = PROVIDERS.get(provider_name, {})
    supports_reasoning = provider.get("supports_reasoning_effort", provider_name == "deepseek")
    if not supports_reasoning and "reasoning_effort" in chat_request:
        chat_request.pop("reasoning_effort")
        log.debug(f"[{request_id}] Removed reasoning_effort (provider '{provider_name}' doesn't support it)")

    upstream_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if is_stream else "application/json",
    }

    upstream_url = f"{base_url}{api_path}"
    log.info(f"[{request_id}] Forwarding to: {upstream_url}")
    if not is_stream:
        for attempt in range(MAX_RETRIES):
            try:
                client = get_client()
                resp = await client.post(
                    upstream_url,
                    headers=upstream_headers,
                    json=chat_request,
                )
                
                if resp.status_code >= 500:
                    if attempt < MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        log.warning(f"[{request_id}] Upstream 5xx, retry in {wait}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait)
                        continue
                
                resp_json = resp.json()
                
                if resp.status_code >= 400:
                    log.error(f"[{request_id}] Upstream error: {resp.status_code} - {resp.text[:500]}")
                    error_resp = chat_error_to_responses_error(resp_json)
                    return JSONResponse(status_code=resp.status_code, content=error_resp)
                
                responses_resp = chat_to_responses(resp_json, request_model, response_id=request_id)
                
                # Cache the response output for previous_response_id support
                if "output" in responses_resp:
                    safe_create_task(cache_response(request_id, responses_resp["output"]))
                
                usage = responses_resp.get("usage", {})
                log.info(
                    f"[{request_id}] Done "
                    f"input={usage.get('input_tokens', '?')} "
                    f"output={usage.get('output_tokens', '?')}"
                )
                inp = usage.get('input_tokens', 0)
                outp = usage.get('output_tokens', 0)
                if inp == 0:
                    inp = _estimate_input_tokens(chat_request)
                safe_create_task(record_usage(request_model, inp, outp))
                
                return JSONResponse(content=responses_resp)
                
            except httpx.TimeoutException:
                log.error(f"[{request_id}] Timeout (attempt {attempt + 1})")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return JSONResponse(
                        status_code=504,
                        content={"error": {"message": "Upstream timeout", "type": "timeout", "code": "timeout"}}
                    )
            except Exception as e:
                log.error(f"[{request_id}] Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return JSONResponse(
                        status_code=502,
                        content={"error": {"message": str(e), "type": "proxy_error", "code": "proxy_error"}}
                    )
    
    # Streaming
    else:
        return StreamingResponse(
            stream_responses(request_id, upstream_url, upstream_headers, chat_request, request_model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def _estimate_input_tokens(chat_request: dict) -> int:
    """Estimate input token count from chat messages when provider returns 0."""
    total_chars = 0
    for msg in chat_request.get("messages", []):
        c = msg.get("content", "")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
    return max(1, total_chars // 3) if total_chars > 0 else 0


async def stream_responses(
    request_id: str,
    upstream_url: str,
    upstream_headers: dict,
    chat_request: dict,
    request_model: str,
):
    for attempt in range(MAX_RETRIES):
        translator = StreamTranslator(response_id=request_id, model=request_model)
        data_started = False
        try:
            client = get_client()
            async with client.stream(
                "POST",
                upstream_url,
                headers=upstream_headers,
                json=chat_request,
            ) as resp:
                if resp.status_code >= 400:
                    error_body = await resp.aread()
                    log.error(f"[{request_id}] Upstream error: {resp.status_code} - {error_body[:500]}")
                    try:
                        error_json = json.loads(error_body)
                    except json.JSONDecodeError:
                        error_json = {"error": {"message": error_body.decode()}}
                    error_resp = chat_error_to_responses_error(error_json)
                    yield f"data: {json.dumps(error_resp)}\n\n"
                    return

                buffer = ""
                async for chunk in resp.aiter_bytes():
                    text = chunk.decode("utf-8", errors="replace")
                    buffer += text

                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        event_block = event_block.strip()
                        if not event_block:
                            continue
                        data_str = None
                        for line in event_block.split("\n"):
                            line = line.strip()
                            if line.startswith("data: "):
                                data_str = line[6:]
                            elif line.startswith("data:"):
                                data_str = line[5:]
                        if not data_str:
                            continue
                        if data_str == "[DONE]":
                            final_events = translator.finalize()
                            for event in final_events:
                                yield f"{event}\n\n"
                            output_items = translator.get_output_items()
                            if output_items:
                                safe_create_task(cache_response(request_id, output_items))
                            usage_input = translator._usage.get("prompt_tokens", 0)
                            usage_output = translator._usage.get("completion_tokens", 0)
                            if usage_input == 0:
                                usage_input = _estimate_input_tokens(chat_request)
                            if usage_output == 0:
                                usage_output = translator._output_char_count // 3 if translator._output_char_count > 0 else 0
                            safe_create_task(record_usage(request_model, usage_input, usage_output))
                            return
                        try:
                            chunk_data = json.loads(data_str)
                            translated = translator.process_chunk(chunk_data)
                            data_started = True
                            for event in translated:
                                yield f"{event}\n\n"
                        except json.JSONDecodeError:
                            log.warning(f"[{request_id}] Invalid JSON in stream: {data_str[:100]}")
                            continue

                if buffer.strip():
                    event_block = buffer.strip()
                    data_str = None
                    for line in event_block.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                        elif line.startswith("data:"):
                            data_str = line[5:]
                    if data_str and data_str != "[DONE]":
                        try:
                            chunk_data = json.loads(data_str)
                            translated = translator.process_chunk(chunk_data)
                            for event in translated:
                                yield f"{event}\n\n"
                        except json.JSONDecodeError:
                            log.warning(f"[{request_id}] Invalid JSON in remaining buffer: {data_str[:100]}")

                final_events = translator.finalize()
                for event in final_events:
                    yield f"{event}\n\n"
                output_items = translator.get_output_items()
                if output_items:
                    safe_create_task(cache_response(request_id, output_items))
                usage_input = translator._usage.get("prompt_tokens", 0)
                usage_output = translator._usage.get("completion_tokens", 0)
                if usage_input == 0:
                    usage_input = _estimate_input_tokens(chat_request)
                if usage_output == 0:
                    usage_output = translator._output_char_count // 3 if translator._output_char_count > 0 else 0
                log.info(f"[{request_id}] Stream END. usage={translator._usage}")
                safe_create_task(record_usage(request_model, usage_input, usage_output))
                return

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            if data_started:
                log.error(f"[{request_id}] Stream interrupted after data: {e}")
                yield f"event: error\ndata: {json.dumps({'error': {'message': str(e), 'type': 'proxy_error'}})}\n\n"
                return
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                log.warning(f"[{request_id}] Stream connect failed (attempt {attempt+1}), retry in {wait}s: {e}")
                await asyncio.sleep(wait)
            else:
                log.error(f"[{request_id}] Stream failed after {MAX_RETRIES} attempts: {e}")
                yield f"event: error\ndata: {json.dumps({'error': {'message': str(e), 'type': 'proxy_error'}})}\n\n"
        except BaseException as e:
            if "Cancelled" in type(e).__name__:
                log.info(f"[{request_id}] Stream cancelled (client disconnected)")
            else:
                log.error(f"[{request_id}] Stream error: {type(e).__name__}: {e}")
            yield f"event: error\ndata: {json.dumps({'error': {'message': str(e), 'type': 'proxy_error'}})}\n\n"
            return


async def handle_pass_through(
    request_id: str,
    method: str,
    path: str,
    body: bytes,
    request_headers: dict,
    body_json: dict = None,
):
    # Try to resolve model from body (e.g., chat/completions), fall back to default
    model_name = DEFAULT_MODEL
    if body_json and "model" in body_json:
        model_name = body_json["model"]
    
    try:
        provider_name, actual_model, base_url, api_key, api_path = resolve_model(model_name)
    except ValueError:
        # If resolution fails, use first available provider
        if PROVIDERS:
            first = list(PROVIDERS.keys())[0]
            base_url = PROVIDERS[first].get("base_url", "")
            api_key = PROVIDERS[first].get("api_key", "")
            provider_name = first
            actual_model = model_name
        else:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "No providers configured", "type": "proxy_error"}}
            )
    if path == "chat/completions":
        upstream_url = f"{base_url}{api_path}"
    else:
        upstream_url = f"{base_url}/v1/{path}"
    
    upstream_headers = {
        "Authorization": f"Bearer {api_key}",
    }
    
    ct = request_headers.get("content-type", "")
    if ct:
        upstream_headers["Content-Type"] = ct
    
    accept = request_headers.get("accept", "")
    if accept:
        upstream_headers["Accept"] = accept
    
    log.debug(f"[{request_id}] {method} /v1/{path} -> {provider_name} (pass-through)")
    
    client = get_client()
    
    try:
        resp = await client.request(
            method=method,
            url=upstream_url,
            headers=upstream_headers,
            content=body or None,
        )
        
        ct_lower = resp.headers.get("content-type", "").lower()
        if "text/event-stream" in ct_lower:
            return StreamingResponse(
                resp.aiter_bytes(),
                media_type="text/event-stream",
                headers=dict(resp.headers),
            )
        
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
        
    except Exception as e:
        log.error(f"[{request_id}] Pass-through error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(e), "type": "proxy_error"}}
        )


# ---- Entry Point ----

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(_global_exc_handler)
    asyncio.set_event_loop(loop)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level=LOG_LEVEL.lower())