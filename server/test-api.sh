#!/bin/bash

# Test API script for the RAG server

echo "=========================================="
echo "Testing RAG Server API"
echo "=========================================="
echo ""

# Health check
echo "1. Health Check:"
curl -X GET https://stipulatory-equatorially-azalea.ngrok-free.dev/health
echo -e "\n"

# Test query 1
echo "2. Test Query 1 - 'Ate pasta':"
curl -X POST https://stipulatory-equatorially-azalea.ngrok-free.dev/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Ate pasta"}'
echo -e "\n"

# Test query 2
echo "3. Test Query 2 - 'Had a burger for lunch':"
curl -X POST https://stipulatory-equatorially-azalea.ngrok-free.dev/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Had a burger for lunch"}'
echo -e "\n"

# Test query 3
echo "4. Test Query 3 - 'Ate eggs':"
curl -X POST https://stipulatory-equatorially-azalea.ngrok-free.dev/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Ate eggs"}'
echo -e "\n"

echo "=========================================="
echo "Tests completed!"
echo "=========================================="

