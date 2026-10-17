# task 
## avwan task 
### adjust the offlineRL reward experiment 
1. first, you should adjust the offlineRL reward during training. In past few experiments , we can clearly draw the conclusion that the bad model performance caused by the long distance navigation task ,such as >24m task in some episodes. howevery, when we adjust the distribution of dataset , increase the ratio of long distance episodes, we find the result decrease yet. This study tell us the curriculum study is not suitable in this task. So ,now we pay attention to the offlineRL itself, If the agent reaches the sound source from a distant position, we can give it some reward. If it fails to reach the sound source, or moves away from it, we should give it some penalty.
2. All the result should test in the whole test dataset , 1000episode.

### attempt other solution
1. If the final result of the avwan task 1 "adjust the offline reward experiment" is not good(failed to surpass the 0.4spl) , you should attempt other solution.
2. None of the solutions may incorporate any rule-based methods, and the val/test datasets must not be added to training.
3. The best model should be chosen based on its performance on the validation dataset.
4. The final result must be tested on the test dataset split (1000 episodes).
