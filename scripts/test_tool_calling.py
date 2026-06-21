"""
驗證 vLLM endpoint 是否支援 tool calling。
用法：
  conda run -n hospital-rag python3 scripts/test_tool_calling.py
"""
from openai import OpenAI

ENDPOINTS = {
    "gemma3": "http://172.31.6.3:8080/gemma3/v1",
    "gemma4": "http://172.31.6.3:8080/gemma4/v1",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_page",
            "description": "取得 PDF 指定頁碼的截圖",
            "parameters": {
                "type": "object",
                "properties": {
                    "page_no": {"type": "integer", "description": "頁碼（從 1 開始）"},
                },
                "required": ["page_no"],
            },
        },
    }
]

TEST_MESSAGE = "請使用 get_page 工具取得第 5 頁的內容。"


def test_endpoint(name: str, base_url: str) -> None:
    print(f"\n{'='*50}")
    print(f"測試 {name}: {base_url}")
    print("=" * 50)

    client = OpenAI(api_key="not-needed", base_url=base_url)

    # 測試 1：tool_choice=auto
    print("\n[測試 1] tool_choice=auto")
    try:
        resp = client.chat.completions.create(
            model="/model",
            messages=[{"role": "user", "content": TEST_MESSAGE}],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            print(f"  ✅ 成功！呼叫了 {tc.function.name}({tc.function.arguments})")
        else:
            print(f"  ⚠️  沒有 tool_calls，模型回文字：{repr(msg.content[:100])}")
    except Exception as e:
        print(f"  ❌ 失敗：{e}")

    # 測試 2：tool_choice=required（強制呼叫工具）
    print("\n[測試 2] tool_choice=required")
    try:
        resp = client.chat.completions.create(
            model="/model",
            messages=[{"role": "user", "content": TEST_MESSAGE}],
            tools=TOOLS,
            tool_choice="required",
            temperature=0.0,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            print(f"  ✅ 成功！呼叫了 {tc.function.name}({tc.function.arguments})")
        else:
            print(f"  ⚠️  沒有 tool_calls，模型回文字：{repr(msg.content[:100])}")
    except Exception as e:
        print(f"  ❌ 失敗：{e}")


if __name__ == "__main__":
    for name, url in ENDPOINTS.items():
        test_endpoint(name, url)
    print("\n完成。")
