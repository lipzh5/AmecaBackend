# Issues observed during demonstration

#### 1. fail to answer questions like "how do you know my name?" or "how did you recognize me?"
   - reason: will trigger face recognition because of the keywords in the questions.
   - solution: modify function doc to redirect the answer (**solved**)

#### 2. fail to deliver facial expressions sometimes when performing emotion imitation task
   - reason: facial expressions and lipsyncing may happen at the same time (see [Mouth Driver](https://docs.engineeredarts.co.uk/en/user/mouth_driver). A mixture between them will weaken the facial expression.)
   - solution: 1) trigger tts say after playing facial expression; 2) reset interaction history when executing emotion imitation task

