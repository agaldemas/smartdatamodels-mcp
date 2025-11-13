# test using Mistral AI agent

```shell
curl --location 'https://api.mistral.ai/v1/conversations' \
--header "Content-Type: application/json" \
--header "X-API-KEY: $MISTRAL_API_KEY" \
--data '{
    "agent_id": "ag_019a39b2dccb70708cfc8adbc980aea9",
    "inputs": "Hello there!"
}'
```
