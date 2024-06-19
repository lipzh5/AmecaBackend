## Issues

### User Experience
1. round-trip latency
    - Solution: Router/Switch
        - Test: 
    - Backup solution: add physical animations
        - Test
    


### Autonomous Behavior
1. should look around on purpose (Torso Look At.py and Neck Look At.py), **avoid unnecessary moves**, which will introduce more noises in the captured image frames
    - Solution: 
    ```
    Eye Look At.py (consumer): self.talking  # self.consumer.active = None if self.talking
    Torso Look At.py (consumerref),    # tick fps: 20-->5
    Neck Look At.py (consumerref),    # tick fps: 20-->5
    ```
    - Test: 
2. initiate/start conversations only when there are human faces in view


### VQA Tasks
1. "but I cannot see what is in your hand as I am an AI and do not have visual capabilities."
    - Solution: modify chat personalities 
    - Test: 

### OpenAI API
1. OpenAI Error: Error code: 429 
```
2024-06-19T00:23:18.475885+00:00 [WARNING] OpenAI Error: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4-0613 in organization org-ytF0gqTk7vPck2HRMphLsqy6 on tokens per min (TPM): Limit 10000, Used 8710, Requested 1385. Please try again in 570ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}

2024-06-19T00:23:18.479931+00:00 [WARNING] Retrying the OpenAI call...
```
    - Solution: max_tokens//2
    - Test: 




## Suggestions
1. Eye look around range (resticted a little bit)
2. Focus on persona in current conversation (do not look around while talking to specific person)
3. Prepare some tools/gadgets with high success rate for the VQA/action recognition task

