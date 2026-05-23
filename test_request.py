#!/usr/bin/env python3
"""测试脚本：模拟客户端发送请求到本地代理"""
import json
import httpx
import time

# 模拟你的客户端可能发送的请求
test_requests = [
    {
        "name": "简单聊天请求",
        "data": {
            "model": "deepseek-chat",
            "input": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "stream": True
        }
    },
    {
        "name": "带 tools 的请求",
        "data": {
            "model": "deepseek-chat",
            "input": [
                {"role": "user", "content": "What's the weather?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                }
            ],
            "stream": True
        }
    },
    {
        "name": "带 response_format 的请求",
        "data": {
            "model": "deepseek-chat",
            "input": [
                {"role": "user", "content": "Return JSON"}
            ],
            "text": {
                "format": {
                    "type": "json_object"
                }
            },
            "stream": True
        }
    }
]

base_url = "http://localhost:2000/v1/responses"

for test in test_requests:
    print(f"\n{'='*60}")
    print(f"测试: {test['name']}")
    print(f"{'='*60}")
    
    try:
        with httpx.stream("POST", base_url, json=test["data"], timeout=10.0) as response:
            print(f"状态码: {response.status_code}")
            if response.status_code == 200:
                print("✓ 请求成功!")
                # 读取前几个事件
                count = 0
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        count += 1
                        if count <= 3:
                            print(f"  收到事件: {line[:100]}...")
                    if count >= 5:
                        break
                print("  ...")
            else:
                print(f"✗ 请求失败!")
                print(f"响应: {response.text[:500]}")
    except Exception as e:
        print(f"✗ 异常: {e}")
    
    time.sleep(1)

print("\n测试完成!")
