# A Blog (record): Understanding Decision Transformer paper

---

Ref. paper: “Decision Transformer: Reinforcement Learning via Sequence Modeling”

### **Outline I fed to gpt**

The goal of writing this blog: serve as a personal recap and proof of effort.

### **1. Introduction**

- **Motivation Behind the Paper**
    - What challenges in traditional RL motivated the introduction of Decision Transformers?
    - The limitations of existing RL techniques, such as value-function-based methods and the need for more scalable, efficient learning models.

### **2. Preliminaries**

- Offline RL - MDP process with fixed limited dataset (trajectory rollouts of arbitrary interactions)
- **Transformer Architecture in RL**
    - Summarize the architecture of **Transformers** and why they are suitable for handling sequential decision-making tasks.
    - Highlight key components such as **self-attention** and **multi-head attention**, and how they are applied to RL.
- **Sequence Modeling in RL**
    - Explain how Decision Transformers treat **RL tasks as sequence modeling problems**.
    - Discuss the core innovation: **state-action-reward** sequences are modeled similarly to NLP tasks, where decisions are based on previous context, just like a sentence prediction task.

### **3. Method**

- **What is Return-to-Go?**
    - Define **return-to-go**, its role in the Decision Transformer model, and how it differs from traditional reward signals in RL.
- **Using Return-to-Go for Trajectory representation**
    - Describe how past actions, states, and return-to-go values are input to the model to predict future actions.

### **4. Evaluation on Offline RL Benchmarks**

- two benchmarks to compare with:
    - TD-learning i.e. CQL
    - Imitation learning i.e. Behavior cloning
- Comparing in Atari benchmark (discrete environment)
- Comparing in OenAI Gym control tasks (continuous environment)

### 5. Discussion

- Does DT perform behavior cloning on a sebset of the data?
    - When data is plentiful: Percentile Behavior Cloning (%BC) match or beat other offline RL methods, and DT is competitive with the performance of the best %BC.
    - When in low data regimes: %BC is weak, and DT can be more effective and outperforms %BC in most games.
- DT’s distribution of returns?
    - almost perfectly match the desired returns
    - even higher returns than expected
- Benefit of using a longer context length?
    - context length K means past information
    - bigger K allows DT to identify which policy generated the actions, enabling better learning/or improving the training dynamics.
- DT perform effective long-term credit assignment, and be accurate critics in sparse reward settings?
    - Key-to-Door environment: This is a grid-based environment with a sequence of three phases: (1) in the first phase, the agent is placed in a room with a key; (2) then, the agent is placed in an empty room; (3) and finally, the agent is placed in a room with a door. The agent receives a binary reward when reaching the door in the third phase, but only if it picked up the key in the first phase.
        - DT can learn successful policies, while TD learning struggles to perform.
        - The transformer continuously updates reward probability based on events during the episode, and attends to critical events in the episode (enabling accurate value prediction)
    - a delayed return version of the D4RL benchmarks where the agent does not receive any rewards along the trajectory, and instead receives the cumulative reward of the trajectory in the final timestep
        - Delayed returns minimally affect Decision Transformer; and due to the nature of the training process, while imitation learning methods are reward agnostic. While TD learning collapses, Decision Transformer and %BC still perform well, indicating that Decision Transformer can be more robust to delayed rewards.
- Since Decision Transformer does not require explicit optimization using learned functions as objectives, it avoids the need for regularization or conservatism.
- Decision Transformer can serve as a powerful “memorization engine” and in conjunction with powerful exploration algorithms like Go-Explore [28], has the potential to simultaneously model and generative a diverse set of behaviors.

### **6. Conclusion**

- Decision Transformer can match or outperform strong algorithms designed explicitly for offline RL with minimal modifications from standard language modeling architectures.
- Transformer models can also be used to model the state evolution of trajectory, potentially serving as an alternative to model-based RL.

### **7. Applications and Future Directions**

- **7.1 Applications in Different Domains**
    - Potential applications of Decision Transformers beyond the experimental tasks in the paper, such as:
        - **Finance**: Predicting future market actions based on historical trading sequences.
        - **Healthcare**: Using patient history data for treatment recommendation sequences.
        - **Autonomous Systems**: Applying DT to **self-driving cars** or **drones**, where long-term decision-making is crucial.
- **7.2 Limitations and Challenges**
    - The challenges and limitations of DT, such as high computational costs and reliance on large offline datasets.
    - Highlight possible future research directions to improve DT, such as reducing the need for large amounts of data or combining DT with online RL.

---

[**Blog: Understanding Decision Transformer Paper**](https://www.notion.so/Blog-Understanding-Decision-Transformer-Paper-127ab1128b4080ddb5f9d595f3f48b22?pvs=21)

---