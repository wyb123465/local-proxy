"""
Protocol translation between OpenAI Responses API and Chat Completions API.

Codex uses the Responses API (wire_api = "responses") but DeepSeek only
supports Chat Completions. This module handles bidirectional translation
for both streaming and non-streaming modes.
"""

import json
import uuid
import time
import logging
from typing import Optional

def _has_image_content(request_body):
    items = request_body.get("input", [])
    # Only check the last user message for images
    last_user_content = None
    for item in reversed(items):
        if isinstance(item, dict) and item.get("role") == "user":
            last_user_content = item.get("content", "")
            break
    if isinstance(last_user_content, list):
        for part in last_user_content:
            if isinstance(part, dict) and part.get("type") in ("input_image", "image_url", "image"):
                return True
    return False


log = logging.getLogger("local-proxy")


def _gen_id(prefix: str) -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _extract_text_content(content) -> str:
    """
    Extract plain text from content which may be:
    - a plain string
    - a list of content parts like [{"type": "input_text", "text": "..."}, ...]
    - a list with image parts (skip those)
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("input_text", "output_text", "text"):
                    parts.append(part.get("text", ""))
                # Skip image parts
        return "\n".join(parts) if parts else ""
    return str(content)


def _extract_multimodal_content(content):
    """
    Extract content for Chat Completions, preserving multimodal format.
    Returns string for simple text, or list for multimodal.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype in ("input_text", "output_text", "text"):
                parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "input_image":
                image_url = part.get("image_url", "")
                parts.append({"type": "text", "text": f"<image>{image_url}</image>"})
            elif ptype == "image_url":
                img = part.get("image_url", {})
                url = img.get("url", "") if isinstance(img, dict) else part.get("image_url", "")
                parts.append({"type": "text", "text": f"<image>{url}</image>"})
        if not parts:
            return ""
        return parts
    return str(content)
def responses_to_chat(request_body: dict, model_map: dict) -> dict:
    """
    Convert Responses API request to Chat Completions API request.
    
    Key transformations:
    - "input" array -> "messages" array
    - Content types: input_text/output_text -> text
    - function_call -> assistant message with tool_calls
    - function_call_output -> tool role message
    - instructions -> prepend system message if not present
    - max_output_tokens -> max_tokens
    """
    model_name = request_body.get("model", "")
    actual_model = model_map.get(model_name, model_name)
    
    messages = []
    input_items = request_body.get("input", [])
    
    # Handle top-level "instructions" as system message
    instructions = request_body.get("instructions", "")
    has_system = False
    
    # Track function calls by call_id to ensure proper pairing
    # Store as: call_id -> {"item": function_call_item, "output": function_call_output_item or None}
    function_calls = {}
    
    # First pass: collect all function_calls and function_call_outputs
    for item in input_items:
        if not isinstance(item, dict):
            continue
        
        item_type = item.get("type", "")
        
        if item_type == "function_call":
            call_id = item.get("call_id", item.get("id", ""))
            if call_id:
                if call_id not in function_calls:
                    function_calls[call_id] = {"item": None, "output": None}
                function_calls[call_id]["item"] = item
        
        elif item_type == "function_call_output":
            call_id = item.get("call_id", "")
            if call_id:
                if call_id not in function_calls:
                    function_calls[call_id] = {"item": None, "output": None}
                function_calls[call_id]["output"] = item
    
    # Second pass: process all items in order
    processed_call_ids = set()
    
    for item in input_items:
        if not isinstance(item, dict):
            continue
        
        item_type = item.get("type", "")
        role = item.get("role", "")
        
        # Skip function_call and function_call_output here, handle them separately
        if item_type in ("function_call", "function_call_output"):
            call_id = item.get("call_id", item.get("id", ""))
            if call_id and call_id not in processed_call_ids and call_id in function_calls:
                fc_data = function_calls[call_id]
                # Add assistant message with tool_calls
                if fc_data["item"]:
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc_data["item"].get("name", ""),
                                "arguments": fc_data["item"].get("arguments", "")
                            }
                        }]
                    }
                    messages.append(assistant_msg)
                
                # Add tool message if output exists
                if fc_data["output"]:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": fc_data["output"].get("output", "")
                    })
                
                processed_call_ids.add(call_id)
            continue
        
        # Regular message with role
        if role:
            if role == "system":
                has_system = True
            elif role == "developer":
                # Map developer role to system for DeepSeek compat
                role = "system"
                has_system = True
            
            content = item.get("content", "")
            converted_content = _extract_multimodal_content(content)

            msg = {"role": role, "content": converted_content}

            # Handle tool_calls in assistant messages
            if role == "assistant" and "tool_calls" in item:
                tc_list = []
                for tc in item["tool_calls"]:
                    if isinstance(tc, dict):
                        tc_list.append({
                            "id": tc.get("id", tc.get("call_id", "")),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": tc.get("arguments", "")
                            }
                        })
                if tc_list:
                    msg["tool_calls"] = tc_list

            messages.append(msg)
    
    # Handle any remaining pending function calls (no corresponding output)
    for call_id, fc_data in function_calls.items():
        if call_id in processed_call_ids:
            continue
        if fc_data["item"]:
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fc_data["item"].get("name", ""),
                        "arguments": fc_data["item"].get("arguments", "")
                    }
                }]
            }
            messages.append(assistant_msg)
    
    # Handle instructions - always apply if present
    if instructions:
        if has_system:
            # Merge into first system message
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    # Prepend instructions to existing system content
                    existing = msg.get("content", "")
                    if isinstance(existing, str):
                        msg["content"] = instructions + "\n\n" + existing
                    elif isinstance(existing, list):
                        # Multimodal content, prepend as text
                        msg["content"] = [{"type": "text", "text": instructions}] + existing
                    break
        else:
            # Insert new system message at beginning
            messages.insert(0, {"role": "system", "content": instructions})
    
    # Build chat completions request
    chat_request = {
        "model": actual_model,
        "messages": messages,
    }
    
    # Copy tools and tool_choice
    if "tools" in request_body:
        tools = request_body["tools"]
        # Filter out invalid tools (must have 'function' field or be in simple format)
        valid_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                if "function" in tool and isinstance(tool["function"], dict):
                    # Standard format - validate required fields
                    func = tool["function"]
                    if "name" in func:
                        valid_tools.append(tool)
                elif "name" in tool and "parameters" in tool:
                    # Convert simple format to proper format
                    valid_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {})
                        }
                    })
        if valid_tools:
            chat_request["tools"] = valid_tools
        elif tools:  # Log if we filtered out tools
            log.warning(f"Filtered out {len(tools) - len(valid_tools)} invalid tools")
    
    if "tool_choice" in request_body:
        chat_request["tool_choice"] = request_body["tool_choice"]
    if "parallel_tool_calls" in request_body:
        chat_request["parallel_tool_calls"] = request_body["parallel_tool_calls"]
    
    # Map parameters
    if "temperature" in request_body:
        chat_request["temperature"] = request_body["temperature"]
    if "top_p" in request_body:
        chat_request["top_p"] = request_body["top_p"]
    if "max_output_tokens" in request_body:
        chat_request["max_tokens"] = request_body["max_output_tokens"]
    
    # Handle text format (structured output)
    text_config = request_body.get("text", {})
    if isinstance(text_config, dict) and "format" in text_config:
        format_data = text_config["format"]
        # Convert to Chat Completions response_format
        if isinstance(format_data, dict):
            if "type" in format_data:
                if format_data["type"] == "json_object":
                    chat_request["response_format"] = {"type": "json_object"}
                elif format_data["type"] == "json_schema":
                    log.debug("Dropping json_schema response_format (unsupported by provider)")
                else:
                    # Unsupported format, drop it
                    log.debug("Dropping unsupported response_format: " + str(format_data.get("type", "?")))
            else:
                # Might be direct json_schema format (no 'type' field)
                if "name" in format_data and "schema" in format_data:
                    log.debug("Dropping json_schema response_format (unsupported by provider)")
                else:
                    # Unknown format, skip to avoid API errors
                    log.warning(f"Skipping unsupported response_format: {format_data}")
    
    # Stream mode
    if "stream" in request_body:
        chat_request["stream"] = request_body["stream"]
    # Request usage in streaming if supported
    if request_body.get("stream") and "stream_options" not in chat_request:
        chat_request["stream_options"] = {"include_usage": True}
    
    # Handle reasoning effort (for R1 models)
    if "reasoning" in request_body:
        reasoning = request_body["reasoning"]
        if isinstance(reasoning, dict) and "effort" in reasoning:
            chat_request["reasoning_effort"] = reasoning["effort"]
    
    return chat_request


def chat_to_responses(
    chat_response: dict,
    request_model: str,
    response_id: Optional[str] = None
) -> dict:
    """
    Convert Chat Completions non-streaming response to Responses API format.
    """
    if response_id is None:
        response_id = _gen_id("resp")
    
    created_at = chat_response.get("created", int(time.time()))
    model = chat_response.get("model", "")
    choices = chat_response.get("choices", [])
    usage = chat_response.get("usage", {})
    
    output_items = []
    
    for choice in choices:
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")
        
        # Map finish_reason
        if finish_reason == "length":
            finish_reason = "max_output_tokens"
        
        has_tool_calls = bool(message.get("tool_calls"))
        has_content = bool(message.get("content"))
        
        if has_tool_calls:
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                call_id = tc.get("id", _gen_id("call"))
                output_items.append({
                    "id": _gen_id("fc"),
                    "type": "function_call",
                    "call_id": call_id,
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                    "status": "completed"
                })
        
        if has_content:
            content_text = message["content"]
            if isinstance(content_text, list):
                # Handle multimodal content in response (rare but possible)
                text_parts = []
                for part in content_text:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content_text = "\n".join(text_parts) if text_parts else ""
            
            msg_id = _gen_id("msg")
            output_items.append({
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content_text,
                        "annotations": []
                    }
                ]
            })
    
    # Build Responses API response
    input_tok = usage.get("prompt_tokens", 0)
    output_tok = usage.get("completion_tokens", 0)
    total_tok = usage.get("total_tokens", 0)
    # Estimate when provider returns 0 usage (e.g. Mimo)
    if output_tok == 0 and output_items:
        output_chars = sum(
            len(p.get("text", ""))
            for item in output_items
            for p in item.get("content", [])
            if isinstance(p, dict)
        )
        output_chars += sum(
            len(item.get("arguments", ""))
            for item in output_items
            if item.get("type") == "function_call"
        )
        if output_chars > 0:
            output_tok = max(1, output_chars // 3)
            log.debug(f"Estimated output tokens: {output_tok} from {output_chars} chars")
    if total_tok == 0:
        total_tok = input_tok + output_tok
    responses_resp = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": request_model,
        "output": output_items,
        "usage": {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "total_tokens": total_tok,
        }
    }
    
    # If deepseek returned reasoning tokens, include them
    if "completion_tokens_details" in usage:
        details = usage["completion_tokens_details"]
        if "reasoning_tokens" in details:
            responses_resp["usage"]["output_tokens_details"] = {
                "reasoning_tokens": details["reasoning_tokens"]
            }
    
    return responses_resp


def chat_error_to_responses_error(chat_error: dict) -> dict:
    """Convert Chat Completions error to Responses API error format."""
    error_info = chat_error.get("error", {})
    return {
        "error": {
            "message": error_info.get("message", "Unknown error"),
            "type": error_info.get("type", "api_error"),
            "code": error_info.get("code", "unknown"),
            "param": error_info.get("param")
        }
    }


class StreamTranslator:
    """
    Translate Chat Completions SSE stream to Responses API SSE stream.
    
    State machine for handling text and tool_call chunks.
    """
    
    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.created_at = int(time.time())
        
        # State tracking
        self._started = False
        self._finished = False
        self._current_item_type = None  # "message", "function_call", or "reasoning"
        self._current_item_id = None
        self._current_content_index = 0
        self._next_index = 0  # Next index to allocate for new output items
        self._current_output_index = None  # output_index of the current item
        self._text_started = False
        self._text_content = ""
        # Track tool calls being accumulated
        self._tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments, done}
        # Reasoning content tracking (for DeepSeek R1 etc.)
        self._reasoning_started = False
        self._reasoning_content = ""
        self._reasoning_item_id = None
        
        # Usage tracking
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Output character count for token estimation
        self._output_char_count = 0

        # Final output items (for caching)
        self._final_output_items = []
    
    def process_chunk(self, chunk: dict) -> list[str]:
        """
        Process a single Chat Completions chunk and return
        a list of SSE lines (without the "data: " prefix).
        Returns empty list if nothing to emit.
        """
        events = []
        
        choices = chunk.get("choices", [])
        if not choices:
            return events
        
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")
        
        # Start events on first chunk
        if not self._started:
            self._started = True
            events.extend(self._emit_start_events())
        
        # Handle tool calls in delta
        tool_calls_delta = delta.get("tool_calls", [])
        if tool_calls_delta:
            events.extend(self._handle_tool_call_delta(tool_calls_delta))
        
        # Handle text content in delta
        content = delta.get("content")
        if content:
            events.extend(self._handle_text_delta(content))

        # Handle reasoning_content (for DeepSeek R1 etc.)
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content:
            events.extend(self._handle_reasoning_delta(reasoning_content))

        # Handle finish_reason
        if finish_reason:
            events.extend(self._handle_finish())
        
        # Capture usage from final chunk
        usage = chunk.get("usage")
        if usage:
            self._usage = usage
        
        return events
    
    def finalize(self) -> list[str]:
        """Get final events after stream ends."""
        if self._finished:
            return []
        events = self._handle_finish()
        return events
    
    def _emit_start_events(self) -> list[str]:
        """Emit initial Responses API stream events."""
        resp = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "output": []
        }
        return [
            f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': resp})}",
            f"event: response.in_progress\ndata: {json.dumps({'type': 'response.in_progress', 'response': resp})}",
        ]
    
    def _handle_text_delta(self, content: str) -> list[str]:
        """Handle text content delta."""
        events = []

        # First text: emit output_item.added and content_part.added
        if self._current_item_type != "message":
            # Close previous item if any (e.g., tool call finished)
            if self._current_item_type == "function_call":
                events.extend(self._close_function_call_item())
            elif self._current_item_type == "reasoning":
                events.extend(self._close_reasoning_item())

            self._current_item_type = "message"
            self._current_item_id = _gen_id("msg")
            self._text_started = False

            # Allocate output_index for this item
            self._current_output_index = self._next_index
            self._next_index += 1

            # Emit output_item.added for message
            item = {
                "id": self._current_item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }
            events.append(
                f"event: response.output_item.added\n"
                f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': self._current_output_index, 'item': item})}"
            )

        if not self._text_started:
            self._text_started = True
            self._current_content_index = 0
            part = {"type": "output_text", "text": "", "annotations": []}
            events.append(
                f"event: response.content_part.added\n"
                f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': self._current_item_id, 'output_index': self._current_output_index, 'content_index': self._current_content_index, 'part': part})}"
            )


        self._text_content += content
        self._output_char_count += len(content)

        # Emit text delta
        events.append(
            f"event: response.output_text.delta\n"
            f"data: {json.dumps({'type': 'response.output_text.delta', 'item_id': self._current_item_id, 'output_index': self._current_output_index, 'content_index': self._current_content_index, 'delta': content})}"
        )

        return events

    def _handle_reasoning_delta(self, content: str) -> list[str]:
        """Handle reasoning content delta (for DeepSeek R1 etc.)."""
        events = []

        # First reasoning: emit output_item.added
        if not self._reasoning_started:
            self._reasoning_started = True
            self._reasoning_item_id = _gen_id("reasoning")
            self._current_item_type = "reasoning"
            self._current_item_id = self._reasoning_item_id
            self._current_output_index = self._next_index
            self._next_index += 1

            item = {
                "id": self._reasoning_item_id,
                "type": "reasoning",
                "status": "in_progress",
                "content": []
            }
            events.append(
                f"event: response.output_item.added\n"
                f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': self._current_output_index, 'item': item})}"
            )

        self._reasoning_content += content
        self._output_char_count += len(content)

        # Emit reasoning delta
        events.append(
            f"event: response.reasoning_text.delta\n"
            f"data: {json.dumps({'type': 'response.reasoning_text.delta', 'item_id': self._reasoning_item_id, 'output_index': self._current_output_index, 'delta': content})}"
        )

        return events

    def _close_reasoning_item(self) -> list[str]:
        """Close reasoning item."""
        events = []
        if self._reasoning_started and self._reasoning_item_id:
            # Emit reasoning_text.done
            events.append(
                f"event: response.reasoning_text.done\n"
                f"data: {json.dumps({'type': 'response.reasoning_text.done', 'item_id': self._reasoning_item_id, 'output_index': self._current_output_index, 'text': self._reasoning_content})}"
            )

            # Emit output_item.done
            item = {
                "id": self._reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "content": [{"type": "text", "text": self._reasoning_content}]
            }
            events.append(
                f"event: response.output_item.done\n"
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': self._current_output_index, 'item': item})}"
            )

            self._reasoning_started = False
            self._reasoning_content = ""
            self._reasoning_item_id = None
            self._current_item_type = None
            self._current_item_id = None

        return events

    def _handle_tool_call_delta(self, tool_calls_delta: list) -> list[str]:
        """Handle tool call deltas."""
        events = []
        
        for tc in tool_calls_delta:
            index = tc.get("index", 0)
            
            # Initialize tracking for new tool call
            if index not in self._tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                self._tool_calls[index] = {
                    "id": tc_id,
                    "name": func.get("name", ""),
                    "arguments": "",
                    "item_id": _gen_id("fc"),
                    "output_index": self._next_index,
                    "started": False,
                }
                self._next_index += 1
            
            tc_state = self._tool_calls[index]
            func_delta = tc.get("function", {})
            
            # Name might come in a separate chunk
            if func_delta.get("name") and not tc_state["name"]:
                tc_state["name"] = func_delta["name"]
            
            # Accumulate arguments
            args_delta = func_delta.get("arguments", "")
            if args_delta and not tc_state["started"]:
                # First argument delta: emit output_item.added
                tc_state["started"] = True
                
                # Switch from text to function_call
                if self._current_item_type == "message":
                    events.extend(self._close_message_item())
                
                self._current_item_type = "function_call"
                self._current_item_id = tc_state["item_id"]
                self._current_output_index = tc_state["output_index"]
                
                item = {
                    "id": tc_state["item_id"],
                    "type": "function_call",
                    "call_id": tc_state["id"],
                    "name": tc_state["name"],
                    "arguments": "",
                    "status": "in_progress"
                }
                events.append(
                    f"event: response.output_item.added\n"
                    f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': tc_state['output_index'], 'item': item})}"
                )
            
            if args_delta:
                tc_state["arguments"] += args_delta
                self._output_char_count += len(args_delta)
                if tc_state["started"]:
                    events.append(
                        f"event: response.function_call_arguments.delta\n"
                        f"data: {json.dumps({'type': 'response.function_call_arguments.delta', 'item_id': tc_state['item_id'], 'output_index': tc_state['output_index'], 'delta': args_delta})}"
                    )
        
        return events
    
    def _close_message_item(self) -> list[str]:
        """Close current message item."""
        events = []
        if self._current_item_type == "message" and self._current_item_id:
            if self._text_started:
                # Emit content_part.done
                events.append(
                    f"event: response.content_part.done\n"
                    f"data: {json.dumps({'type': 'response.content_part.done', 'item_id': self._current_item_id, 'output_index': self._current_output_index, 'content_index': self._current_content_index, 'part': {'type': 'output_text', 'text': self._text_content, 'annotations': []}})}"
                )
            
            # Emit output_item.done for message
            item = {
                "id": self._current_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": self._text_content, "annotations": []}]
            }
            events.append(
                f"event: response.output_item.done\n"
                f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': self._current_output_index, 'item': item})}"
            )
            self._current_item_type = None
            self._current_item_id = None
        return events
    
    def _close_function_call_item(self) -> list[str]:
        """Close function call items."""
        events = []
        for index, tc_state in self._tool_calls.items():
            if tc_state["started"]:
                # Emit function_call_arguments.done
                events.append(
                    f"event: response.function_call_arguments.done\n"
                    f"data: {json.dumps({'type': 'response.function_call_arguments.done', 'item_id': tc_state['item_id'], 'output_index': tc_state['output_index'], 'arguments': tc_state['arguments']})}"
                )
                
                # Emit output_item.done
                item = {
                    "id": tc_state["item_id"],
                    "type": "function_call",
                    "call_id": tc_state["id"],
                    "name": tc_state["name"],
                    "arguments": tc_state["arguments"],
                    "status": "completed"
                }
                events.append(
                    f"event: response.output_item.done\n"
                    f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': tc_state['output_index'], 'item': item})}"
                )
        self._tool_calls.clear()
        self._current_item_type = None
        self._current_item_id = None
        return events
    
    def _handle_finish(self) -> list[str]:
        """Handle stream completion."""
        if self._finished:
            return []
        self._finished = True

        events = []

        # Close any open items
        saved_item_type = self._current_item_type
        saved_item_id = self._current_item_id
        saved_text = self._text_content
        saved_reasoning = self._reasoning_content

        if self._current_item_type == "message":
            events.extend(self._close_message_item())
        # Save tool_calls before close clears them
        saved_tool_calls = {k: dict(v) for k, v in self._tool_calls.items()}

        if self._current_item_type == "function_call":
            events.extend(self._close_function_call_item())
        if self._current_item_type == "reasoning":
            events.extend(self._close_reasoning_item())

        # Build final response with output items and usage
        output_items = []
        for index, tc_state in sorted(saved_tool_calls.items()):
            if tc_state["started"]:
                output_items.append({
                    "id": tc_state["item_id"],
                    "type": "function_call",
                    "call_id": tc_state["id"],
                    "name": tc_state["name"],
                    "arguments": tc_state["arguments"],
                    "status": "completed"
                })

        if saved_item_id and saved_item_type == "message":
            output_items.append({
                "id": saved_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": saved_text, "annotations": []}]
            })

        if saved_item_id and saved_item_type == "reasoning":
            output_items.append({
                "id": saved_item_id,
                "type": "reasoning",
                "status": "completed",
                "content": [{"type": "text", "text": saved_reasoning}]
            })

        input_tok = self._usage.get("prompt_tokens", 0)
        output_tok = self._usage.get("completion_tokens", 0)
        total_tok = self._usage.get("total_tokens", 0)
        # Estimate when provider returns 0 usage (e.g. Mimo)
        if output_tok == 0 and self._output_char_count > 0:
            output_tok = max(1, self._output_char_count // 3)
            log.debug(f"Estimated output tokens: {output_tok} from {self._output_char_count} chars")
        if total_tok == 0:
            total_tok = input_tok + output_tok
        usage = {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "total_tokens": total_tok,
        }

        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output_items,
            "usage": usage
        }

        # Save output items for caching
        self._final_output_items = output_items

        events.append(
            f"event: response.completed\n"
            f"data: {json.dumps({'type': 'response.completed', 'response': response})}"
        )

        return events

    def get_output_items(self) -> list:
        """Get the final output items (for response caching)."""
        return getattr(self, '_final_output_items', [])