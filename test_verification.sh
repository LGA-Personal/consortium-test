#!/bin/bash
# Test script to verify consortium verification is working

echo "=============================================="
echo "Testing Consortium + EXO Verification"
echo "=============================================="
echo ""
echo "Make sure EXO is running in another terminal:"
echo "  source .venv/bin/activate && python -m exo --api-port 52415"
echo ""
echo "Press Enter to place the model instance..."
read

echo "Step 1: Placing model instance (llama-3.2-1b)..."
PLACE_RESULT=$(curl -s -X POST http://localhost:52415/place_instance \
  -H "Content-Type: application/json" \
  -d '{"model_id": "llama-3.2-1b"}')

if echo "$PLACE_RESULT" | grep -q "Command received"; then
    echo "   Model placement command received!"
    echo "   Waiting 5 seconds for model to load..."
    sleep 5
else
    echo "   Error placing model: $PLACE_RESULT"
    exit 1
fi

echo ""
echo "Step 2: Sending chat completion request..."
echo ""

RESPONSE=$(curl -s -X POST http://localhost:52415/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama-3.2-1b",
    "messages": [{"role": "user", "content": "explain why the sky is blue"}],
    "max_tokens": 50,
    "stream": false
  }')

CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; r=json.load(sys.stdin); print(r.get('choices', [{}])[0].get('message', {}).get('content', 'No response'))" 2>/dev/null)

if [ -n "$CONTENT" ] && [ "$CONTENT" != "No response" ]; then
    echo "Response: $CONTENT"
else
    echo "Error: Could not get response from EXO API"
    echo "Raw response: $RESPONSE"
fi

echo ""
echo "=============================================="
echo "Check the EXO terminal for verification logs!"
echo "You should see:"
echo "  🔐 CONSORTIUM VERIFICATION ENABLED"
echo "  🔐 CONSORTIUM: Layer X computing commitment..."
echo "  ✅ Commitment: abc123..."
echo "=============================================="
