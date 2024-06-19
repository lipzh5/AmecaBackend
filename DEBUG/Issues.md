## Issues

### Autonomous behavior
1. should look around on purpose (Torso Look At.py and Neck Look At.py), **avoid unnecessary moves**, which will introduce more noises in the captured image frames
2. initiate/start conversations only when there are human faces in view

### VQA tasks
1. "but I cannot see what is in your hand as I am an AI and do not have visual capabilities."
    - Solution: modify chat personalities 
    - Test: 

2024-06-19T00:23:18.475885+00:00 [WARNING] OpenAI Error: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4-0613 in organization org-ytF0gqTk7vPck2HRMphLsqy6 on tokens per min (TPM): Limit 10000, Used 8710, Requested 1385. Please try again in 570ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}

2024-06-19T00:23:18.479931+00:00 [WARNING] Retrying the OpenAI call...


## Suggestions
1. Eye look around range (resticted a little bit)
2. Focus on persona in current conversation (do not look around while talking to specific person)
3. Prepare some tools/gadgets with high success rate for the VQA/action recognition task

