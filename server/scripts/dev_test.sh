#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
YT_URL="${TEST_YT_URL:-}"
UPLOAD_FILE="${TEST_AUDIO_FILE:-}"

echo "== Test YouTube =="
if [[ -z "$YT_URL" ]]; then
  echo "Définissez TEST_YT_URL pour lancer le test YouTube."
else
  JOB_ID=$(curl -s -X POST "$BASE_URL/jobs?outputType=tab&tuning=EADGBE&capo=0&quality=fast" \
    -H "Content-Type: application/json" \
    -d "{\"youtubeUrl\":\"$YT_URL\"}" | python -c "import sys, json; print(json.load(sys.stdin)['jobId'])")
  echo "JobId: $JOB_ID"
  while true; do
    STATUS=$(curl -s "$BASE_URL/jobs/$JOB_ID" | python -c "import sys, json; data=json.load(sys.stdin); print(data['status'], data.get('progress',0), data.get('stage',''))")
    echo "$STATUS"
    if [[ "$STATUS" == DONE* ]] || [[ "$STATUS" == FAILED* ]]; then
      break
    fi
    sleep 2
  done
  curl -s "$BASE_URL/jobs/$JOB_ID/result" | python -m json.tool
fi

echo "== Test Upload =="
if [[ -z "$UPLOAD_FILE" ]]; then
  echo "Définissez TEST_AUDIO_FILE pour lancer le test upload."
  exit 0
fi

JOB_ID=$(curl -s -X POST "$BASE_URL/jobs?outputType=tab&tuning=EADGBE&capo=0&quality=fast" \
  -F "audio=@${UPLOAD_FILE}" | python -c "import sys, json; print(json.load(sys.stdin)['jobId'])")
echo "JobId: $JOB_ID"
while true; do
  STATUS=$(curl -s "$BASE_URL/jobs/$JOB_ID" | python -c "import sys, json; data=json.load(sys.stdin); print(data['status'], data.get('progress',0), data.get('stage',''))")
  echo "$STATUS"
  if [[ "$STATUS" == DONE* ]] || [[ "$STATUS" == FAILED* ]]; then
    break
  fi
  sleep 2
done
TAB_URL=$(curl -s "$BASE_URL/jobs/$JOB_ID/result" | python -c "import sys, json; print(json.load(sys.stdin).get('tabTxtUrl',''))")
if [[ -n "$TAB_URL" ]]; then
  curl -s "$TAB_URL" | head -n 12
fi
