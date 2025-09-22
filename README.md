# Lunar Lander — DQN from Scratch (Hackathon Edition)

> A from‑scratch **Deep Q‑Network (DQN)** that learns to solve Gymnasium’s **LunarLander‑v3**. Implemented in PyTorch with **experience replay**, a **target network**, and **ε‑greedy** exploration. Includes experiments, hyperparameter sweeps, and a brief Double DQN attempt.

---

## 🎬 Demo (side‑by‑side)

<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/1baf5c10-b72b-4c4d-8c48-1a1471466018" width="370" alt="Failed Lunar Lander">
<br>
<p align="center"><strong>Untrained (random policy)</strong></p>
</td>
</tr>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/8ee73791-eab1-4d2f-874b-edd66a73245d" width="370" alt="Passed Lunar Lander">
<br>
<p align="center"><strong>Trained DQN (exploitation)</strong></p>
</td>
</tr>
</table>

---

## 💡 Intuition first 

Imagine teaching a small rocket to land. At each moment it sees a snapshot of its situation (where it is, how fast it’s moving, how tilted it is, and whether its legs touch). It can fire one of four thrusters. Early on, it tries lots of moves to learn what works; over time it prefers moves that usually lead to soft, centered landings. To keep learning stable, it studies from a shuffled memory of past moments and compares against a slower, frozen copy of itself so its targets don’t shift while it’s learning.

---

## 🚗 What this project does

**Senses → decides → learns.** On each frame, the lander receives a compact snapshot of its situation (the **state**), chooses a thruster action, gets a reward, and updates its strategy so good choices become more likely.

**State (8 numbers, fully spelled out).**

* \$x, y\$: horizontal and vertical position relative to the landing pad center (meters).
* \$v\_x, v\_y\$: horizontal and vertical velocities (m/s).
* \$\theta\$: body angle (radians), where \$0\$ means upright.
* \$\dot{\theta}\$: angular velocity (radians/s).
* \$c\_L, c\_R\$: left and right leg contact indicators (\$0\$ = no contact, \$1\$ = touching ground).

**Actions (4 choices).** Do nothing, fire **left** thruster, fire **main** thruster, or fire **right** thruster.

**What the network learns.** A function \$Q\_\theta(s,a)\$ that scores “how good” action \$a\$ looks in state \$s\$. The agent mostly takes the action with the **highest** score, but sometimes explores.

**How it stays stable.** Two practical tricks are used:

* **Experience replay** (shuffle and reuse past steps so updates aren’t myopic).
* A slowly refreshed **target network** (a frozen copy that provides stable training targets).

---

## 🧠 RL refresher — intuition → math (fully explained)

**Goal.** Maximize the total future reward (the **discounted return**):

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k} \, r_{t+k+1}
$$

where \$\gamma \in \[0,1)\$ (the **discount factor**) slightly prefers near‑term reward and stabilizes learning.

**Action‑value (Q) function.** The expected return if we take action \$a\$ in state \$s\$ and then keep following our current strategy (policy \$\pi\$):

$$
Q^{\pi}(s,a) = E[ G_t \mid s_t = s,\ a_t = a,\ \pi ]
$$

Think of \$Q\$ as a **scorecard**: higher \$Q\$ means that action tends to lead to softer, centered landings from that situation.

**How DQN learns \$Q\$.** A neural network \$Q\_\theta(s,a)\$ is trained to match a **one‑step look‑ahead target** built from a **frozen copy** \$Q\_{\theta^-}\$:

$$
 y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')
$$

The loss (Huber) nudges \$Q\_\theta(s,a)\$ toward \$y\$.
• *Why a frozen copy?* If the target changed every step, we’d chase a moving goalpost and risk divergence. Freezing \$Q\_{\theta^-}\$ briefly makes targets stable.
• *Why the max?* It encodes the **Bellman optimality** idea: after moving to \$s'\$, we’ll take the **best** next action.

**Exploration vs. exploitation.** With **\$\varepsilon\$‑greedy**, pick a random action with probability \$\varepsilon\$ (to explore); otherwise pick \$\arg\max\_a Q\_\theta(s,a)\$ (to exploit). Start with large \$\varepsilon\$ (≈1.0) and **decay** to a small floor (≈0.01) so exploration fades as the agent becomes competent.

---

## 🏗️ Implementation highlights

* **Network (PyTorch):** 2 hidden layers (typically 64→64 ReLU) mapping the 8‑D state to 4 action scores (Q‑values).
* **Experience replay (memory):** a FIFO buffer of past steps; sampling random minibatches breaks short‑term correlations.
* **Target network (slow copy):** a periodically updated clone of the policy network that provides stable targets during training.
* **Loss & optimizer:** Huber (Smooth‑L1) with Adam; gradient clipping prevents occasional exploding updates.
* **Evaluation:** a test loop with \$\varepsilon=0\$ to report average return over 100 episodes.

> Files: `agent_template.py` (DQN + training loop), `state_discretizer.py` (aux utils), `submit_agent.py` (hackathon entrypoint), `best_model_dqn.pth` (weights).

---

## 🧪 Experiments & hyperparameter sweeps (from the report)

Here, “experiments” means **change one thing at a time** (batch size, network width) and see how the learning curve shifts. The two most influential settings from our runs:

<div align="center">
  <img src="https://github.com/user-attachments/assets/f46e7ec0-11ba-40ac-b0d6-3bce6debaa17" width="500" alt="Report Fig. 3 — 64×64 network; batch size 64"/>
  <br/>
  <sub><b>Fig 1.</b> 64×64 network, batch 64 — strong mean performance (~249), good stability.</sub>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/50ef8abe-0b8a-47a6-abcf-ac3cc56aaa49" width="500" alt="Report Fig. 4 — 128×64 network; batch size 128"/>
  <br/>
  <sub><b>Fig 2.</b> 128×64 network, batch 128 — best of our single‑DQN runs (~268 average over 100 tests).</sub>
</div>

*Why these two?* Batch size governs the **noise** of updates (bigger = smoother but less reactive), while network width governs **capacity** (wider = more expressive but easier to overfit). These plots show the clearest trade‑off we observed.

---

## ⚙️ Final hyperparameters (winning DQN)

* **Network:** 2×(64) ReLU \$\to\$ 4 actions
* **Optimizer / loss:** Adam (\$\alpha=5\times10^{-4}\$), Huber
* **Discount:** \$\gamma=0.99\$
* **Replay:** buffer \$10^{5}\-$\$10^{6}\$ (tested), batch \$128\$
* **\$\varepsilon\$‑greedy:** \$\varepsilon\_0=1.0\$, decay \$0.995 \to \varepsilon\_{\min}=0.01\$
* **Target net update:** every \$5\$ episodes
* **Gradient clip:** \$\lVert \nabla\_\theta \rVert\_2 \le 1.0\$

**Why these settings worked (in plain terms):**

* **64×64 network:** enough capacity to model the lander’s dynamics without memorizing noise.
* **Adam @ \$5\times10^{-4}\$ + Huber:** Adam adapts step sizes per weight; Huber behaves like L2 near the target and L1 on outliers, so occasional bad targets don’t derail training.
* **\$\gamma=0.99\$:** keeps the agent focused on a *whole landing sequence* (not just the next second) while still discounting the far future.
* **Replay \$10^{5}\$–\$10^{6}\$, batch \$128\$:** a large, diverse memory prevents reusing the same short pattern; batch 128 yields **stable** gradients without making updates too sluggish.
* **\$\varepsilon\$ schedule \$1.0 \to 0.01\$ (decay \$0.995\$):** plenty of early exploration to discover strategies; settles to exploitation as performance improves.
* **Target update every \$5\$ episodes:** keeps the target network fresh enough while staying frozen long enough to stabilize bootstrapping.
* **Grad clip at \$1.0\$:** caps rare gradient spikes when rewards or Q‑targets momentarily jump.

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

## 📹 Side‑by‑side results (recap)

Re‑embed the two MP4s here if you want the final results close to the conclusion:

**Untrained**

`UNTRAINED_MP4_URL`

**Trained DQN**

`TRAINED_MP4_URL`

---

## 🙏 Acknowledgements

Guidance and course material by **Dr. Sorina Dumitrescu** (COMPENG 4SL4 — Fundamentals of Machine Learning). 

---

## License

MIT — see `LICENSE`.
