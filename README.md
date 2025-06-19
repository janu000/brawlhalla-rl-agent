## Observation Space

- The agent observes the **last 4 frames** from the game screen.
- Each frame is:
  - Converted to **grayscale** (to simplify visual information).
  - **Downsampled** to a smaller resolution (to reduce input size and speed up processing).
- These 4 processed frames are **stacked together** to form a single input.
- Stacking frames helps the agent understand **motion and changes over time**, not just a single snapshot.
