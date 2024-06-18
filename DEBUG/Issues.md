## Issues

### Autonomous behavior
1. should look around on purpose (Torso Look At.py and Neck Look At.py), **avoid unnecessary moves**, which will introduce more noises in the captured image frames
2. initiate/start conversations only when there are human faces in view

### VQA tasks
1. "but I cannot see what is in your hand as I am an AI and do not have visual capabilities."
    - Solution: modify chat personalities 
    - Test: 


## Suggestions
1. Eye look around range (resticted a little bit)
2. Focus on persona in current conversation (do not look around while talking to specific person)
3. Prepare some tools/gadgets with high success rate for the VQA/action recognition task