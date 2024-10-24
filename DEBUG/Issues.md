## Notes
Penny:
1. Restart ChatController.py when there is a switch between tester. 
2. It is necessary to restart the Python3 node if the configuration of the "Speech Recognition" is modified

Guests:
1. Sentence can be separated by pauses, so try to reduce pauses between words in one question. 
2. Try not to talk to Ameca when she is still talking. 

## Scripts:
1. Service charter specific question answering
2. Visual tasks
3. ~~Minic emotions~~


## Issues

Note test state: preliminary (1), stable (2)

### User Experience
1. round-trip latency
    - ~~Solution: Router/Switch~~
        - Test: 1
    - Backup solution: add physical animations (eyes closed, suggested by Xiang and Jianbo)
        - Test: 2
    - ~~Backup solution: Use local LLM (LLma) to polish answer and the return answer directly~~
        - Test: 1

2. eyes and necks should look at the same diretion
    - Solution: Adjust frequency
        - Test: 2


3. smaller lookaround range (**)
    - Solution: Eye Look At.py (yaw_ratio=0.6)
    - Test: 2


4. focus target and stay for a while (1-2 s), a bit wierd when looking around according to sound
    - Solution: turn off Add_Sound_Lookaround
    - Test: 2


5. ~~eyes closed while thinking (x, y, z), z is up~~ (eye loop down instead)


6. weather query (done)

7. **message queue** while speaking
    - Solution: Modify config (DISABLE_ASR_WHILE_SPEAKING)
    - Test: 2

8. service charter (done)
    - Test: 2


    


### Autonomous Behavior
1. should look around on purpose (Torso Look At.py and Neck Look At.py), **avoid unnecessary moves**, which will introduce more noises in the captured image frames
    - Solution: 
    ```
    Eye Look At.py (consumer): self.talking  # self.consumer.active = None if self.talking
    Torso Look At.py (consumerref),    # tick fps: 20, will cause inconsistant look around if fps=5
    Neck Look At.py (consumerref),     # tick fps: 20, will cause inconsistant look around if fps=5
    
    # contributor.LookAtItem
    ```
    - Test: 1


2. initiate/start conversations only when there are human faces in view


### VQA Tasks
1. "but I cannot see what is in your hand as I am an AI and do not have visual capabilities."
    - Solution: modify chat personalities 
    - Test: 2

### OpenAI API
1. OpenAI Error: Error code: 429 
```
2024-06-19T00:23:18.475885+00:00 [WARNING] OpenAI Error: Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-4-0613 in organization org-ytF0gqTk7vPck2HRMphLsqy6 on tokens per min (TPM): Limit 10000, Used 8710, Requested 1385. Please try again in 570ms. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}

2024-06-19T00:23:18.479931+00:00 [WARNING] Retrying the OpenAI call...
```
    - Solution: max_tokens//2; GPT-4o
    - Test: 2



## Suggestions
1. Eye look around range (resticted a little bit)
2. Focus on persona in current conversation (do not look around while talking to specific person)
3. Prepare some tools/gadgets with high success rate for the VQA/action recognition task

