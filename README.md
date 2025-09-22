# Lunar Lander — DQN from Scratch (Hackathon Edition)

> A from‑scratch **Deep Q‑Network (DQN)** that learns to solve Gymnasium’s **LunarLander‑v3**. Implemented in PyTorch with **experience replay**, a **target network**, and **ε‑greedy** exploration. Includes experiments, hyperparameter sweeps, and a brief Double DQN attempt.

---

## 🎬 Demo (side‑by‑side)

Paste two GitHub *user‑attachments* video links **on their own lines** (GitHub will auto‑embed). Use the same clip, one with the untrained policy, one with the trained DQN.

**Untrained (random policy)**

![readMe_crash](https://github.com/user-attachments/assets/1baf5c10-b72b-4c4d-8c48-1a1471466018)

**Trained DQN (exploitation)**

`TRAINED_MP4_URL`
![lunar_lander](https://github.com/user-attachments/assets/84abf31a-cb73-44db-be06-1e130fea053c)

> Tip: to record, save 60–90s MP4 from your evaluation script, then drag‑and‑drop into a GitHub comment to obtain a permanent `user-attachments` URL.

---

## 💡 Intuition first (no jargon)

Imagine teaching a small rocket to land. At each moment it sees a snapshot of its situation (where it is, how fast it’s moving, how tilted it is, and whether its legs touch). It can fire one of four thrusters. Early on, it tries lots of moves to learn what works; over time it prefers moves that usually lead to soft, centered landings. To keep learning stable, it studies from a shuffled memory of past moments and compares against a slower, frozen copy of itself so its targets don’t shift while it’s learning.

---

## 🚗 What this project does

1. Observes the **8‑dim state** $s=(x,\,y,\,v_x,\,v_y,\,\theta,\,\dot\theta,\,c_L,\,c_R)$ and picks one of **4 actions** (do nothing / left / main / right thruster).
2. Learns an **action‑value function** $Q_\theta(s,a)$ that estimates the expected return if action $a$ is taken in state $s$.
3. Improves the policy by acting **greedily** w\.r.t. $Q_\theta$ while still exploring via **ε‑greedy**.
4. Stabilizes training with **replay** (decorrelates samples) and a slowly‑updated **target network**.

<div align="center">
  <img src="FIG5_FROM_REPORT_URL" width="500" alt="Report Fig. 5 — DQN vs Double DQN: average reward over 2000 episodes"/>
  <br/>
  <sub><b>Fig 5 (from report).</b> DQN vs Double DQN average reward over 2000 episodes. The baseline DQN outperformed our first Double DQN attempt in this setting.</sub>
</div>

---

## 🧠 RL refresher — intuition → math (fully explained)

**Goal.** Learn a rule for choosing thruster actions that leads to gentle, centered landings. After each action the agent gets a **reward** (good if progress, bad if crashes). The aim is to maximize the **discounted return**

G\_t = r\_{t+1} + γ·r\_{t+2} + γ²·r\_{t+3} + … ,

where **γ** (gamma) in \[0,1) makes near‑term rewards count a bit more than far‑future ones (stabilizes learning and encodes preference for sooner success).

**What is Q(s,a)?** The **action‑value** of taking action *a* in state *s* and then following the current strategy. Think of **Q** as a **scorecard**: higher means that action tends to lead to better landings from that situation.

**How DQN learns Q.** We approximate Q with a neural net **Q\_theta(s,a)**. For a sampled step (s,a,r,s′), we build a one‑step look‑ahead **target** using a **frozen copy** of the network **Q\_theta\_minus**:

y = r + γ · max over a′ of Q\_theta\_minus(s′, a′)

and nudge Q\_theta(s,a) toward *y* (Huber loss).
• **Why a frozen copy?** If the target also changed every step, the network would chase a moving goalpost. Freezing it briefly makes the target **stable**.
• **Why the max over actions?** It assumes we’ll take the **best** next action in s′; that’s the Bellman “look ahead then be greedy” idea.

**Exploration vs exploitation.** With **ε‑greedy**, choose a random action with probability ε (to explore); otherwise take the action with the **highest Q‑score** (to exploit what is known). Start high (≈1.0) and **decay** toward a small floor (≈0.01) so exploration fades as the agent becomes competent.

---

## 🏗️ Implementation highlights

* **Network (PyTorch):** 2 hidden layers (typically 64→64 ReLU) mapping the 8‑D state to 4 action scores (Q‑values).
* **Experience replay (memory):** a FIFO buffer of past steps; sampling random minibatches breaks short‑term correlations.
* **Target network (slow copy):** a periodically updated clone of the policy network that provides stable targets during training.
* **Loss & optimizer:** Huber (Smooth‑L1) with Adam; gradient clipping prevents occasional exploding updates.
* **Evaluation:** a test loop with ε set to 0 to report average return over 100 episodes.

> Files: `agent_template.py` (DQN + training loop), `state_discretizer.py` (aux utils), `submit_agent.py` (hackathon entrypoint), `best_model_dqn.pth` (weights). `agent_template.py` (DQN + training loop), `state_discretizer.py` (aux. utils, if used), `submit_agent.py` (entrypoint for hackathon runtime), `best_model_dqn.pth` (weights).

---

## 🧪 Experiments & hyperparameter sweeps (from the report)

Here, “experiments” means **change one thing at a time** (batch size, network width) and see how the learning curve shifts. The two most influential settings from our runs:

<div align="center">
  <img src="FIG3_FROM_REPORT_URL" width="500" alt="Report Fig. 3 — 64×64 network; batch size 64"/>
  <br/>
  <sub><b>Fig 1 (report Fig. 3).</b> 64×64 network, batch 64 — strong mean performance (~249), good stability.</sub>
</div>

<div align="center">
  <img src="FIG4_FROM_REPORT_URL" width="500" alt="Report Fig. 4 — 128×64 network; batch size 128"/>
  <br/>
  <sub><b>Fig 2 (report Fig. 4).</b> 128×64 network, batch 128 — best of our single‑DQN runs (~268 average over 100 tests).</sub>
</div>

*Why these two?* Batch size governs the **noise** of updates (bigger = smoother but less reactive), while network width governs **capacity** (wider = more expressive but easier to overfit). These plots show the clearest trade‑off we observed.

---

## ⚙️ Final hyperparameters (winning DQN)

* **Network:** 2×(64) ReLU → 4 actions
* **Optimizer / loss:** Adam (alpha = 5e-4), Huber
* **Discount:** gamma = 0.99
* **Replay:** buffer 1e5–1e6 (tested), batch 128
* **ε‑greedy:** epsilon\_start = 1.0, decay = 0.995 → epsilon\_min = 0.01
* **Target net update:** every 5 episodes
* **Gradient clip:** L2 norm ≤ 1.0

**Why these settings worked (in plain terms):**

* **64×64 network:** enough capacity to model the lander’s dynamics without memorizing noise.
* **Adam @ 5e‑4 + Huber:** Adam adapts step sizes per weight; Huber behaves like L2 near the target and L1 on outliers, so occasional bad targets don’t derail training.
* **gamma = 0.99:** keeps the agent focused on a *whole landing sequence* (not just the next second) while still discounting the far future.
* **Replay 1e5–1e6, batch 128:** a large, diverse memory prevents seeing the same short pattern repeatedly; batch 128 gives **stable** gradient estimates without making updates too sluggish.
* **Epsilon schedule 1.0 → 0.01 (decay 0.995):** plenty of early exploration to discover strategies; settles to exploitation as performance improves.
* **Target update every 5 episodes:** keeps the target network “fresh enough” while staying frozen long enough to stabilize bootstrapping.
* **Grad clip at 1.0:** caps rare gradient spikes that can occur when rewards or Q‑targets momentarily jump.

---

## 📜 Pseudocode (training loop)

```text
initialize Q_theta and target Q_theta_minus with same weights
replay D ← empty FIFO buffer
for episode = 1..N:
  s ← env.reset(); done ← false
  while not done:
    a ← epsilon_greedy(Q_theta, s)
    s', r, done ← env.step(a)
    push(D, (s,a,r,s',done))
    if |D| ≥ batch:  # update step
      B ← sample_minibatch(D)
      y ← r + gamma * max_{a'} Q_theta_minus(s', a') * (1 - done)
      loss ← Huber( Q_theta(s,a), y )
      theta ← AdamStep(grad_theta loss); clip ||grad_theta||_2 ≤ 1
    s ← s'
  if episode % K == 0:  theta_minus ← theta  # hard target update
```

**What each block is doing:**

* **Initialize networks:** create the main Q‑network and its frozen copy (target network).
* **Replay buffer:** store past transitions so updates can sample *shuffled* experience (breaks short‑term correlations).
* **Action selection:** `epsilon_greedy` mixes random exploration with greedy choices from the current Q.
* **Store transition:** append `(s,a,r,s',done)` to replay so it can be reused many times.
* **Update step:** when enough samples exist, draw a minibatch, compute the Bellman targets `y`, measure loss, step Adam, and clip gradients to avoid spikes.
* **Target refresh:** every `K` episodes, copy weights into the target network to re‑stabilize the bootstrapping target.

---

## ✅ Expected behavior & limitations

* **Learning signal:** rewards rise from strongly negative (crashing) to $>200$ as landings become soft and centered; legs‑contact indicators help trigger the final burn.
* **Robustness:** still sensitive to seed and ε‑schedule; larger networks can overfit replay content.
* **Double DQN:** our first pass underperformed baseline DQN (see Fig. 5); likely needs longer training and/or tuned target‑update cadence.

---


---

## 🙏 Acknowledgements

Guidance and course material by **Dr. Sorina Dumitrescu** (COMPENG 4SL4 — Fundamentals of Machine Learning). This project was completed with teammates **David Sagalovitch** and **Arji Thaiyib**.

---

## License

MIT — see `LICENSE`.
