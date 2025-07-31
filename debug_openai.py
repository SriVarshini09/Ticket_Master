import openai
import os
import json

# Test OpenAI directly to see what's wrong
def debug_openai():
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OpenAI API key found!")
        print("Set it with: $env:OPENAI_API_KEY = 'your-key-here'")
        return
    
    print(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Simple test prompt
    try:
        print("\n🧪 Testing OpenAI API...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Respond with valid JSON only."
                },
                {
                    "role": "user", 
                    "content": "Classify this ticket: 'URGENT: Server down'. Respond with JSON: {\"priority\": \"high\", \"category\": \"technical\"}"
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        print("✅ OpenAI API call successful!")
        print(f"📝 Raw response content: '{response.choices[0].message.content}'")
        print(f"🔢 Response length: {len(response.choices[0].message.content)}")
        print(f"📊 Usage: {response.usage}")
        
        # Try to parse JSON
        content = response.choices[0].message.content.strip()
        if content:
            try:
                parsed = json.loads(content)
                print("✅ JSON parsing successful!")
                print(f"📋 Parsed result: {parsed}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print(f"🔍 Content that failed: '{content}'")
        else:
            print("❌ Empty response content!")
    
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        print(f"🔍 Error type: {type(e)}")

if __name__ == "__main__":
    debug_openai()