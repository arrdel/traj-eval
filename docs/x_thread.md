# X thread — what traj-eval already shows

Draft for posting. It describes the synthetic demo only. It does not claim a result about LLaDA, Dream, or MMaDA.

---

1/ Final-answer accuracy can be perfect and still tell you nothing about how a diffusion language model got there.

I built traj-eval to score the denoising trajectory, not just the last token string.

2/ Three hand-built regimes, 5 trajectories each. Every one of them ends on the gold answer.

Accuracy: 1.000, 1.000, 1.000.

3/ Genuine revision. The verdict moves, then holds at step 4 of 8. Flip rate 0.29. The reasoning text settles with the verdict. Rationalization score 0.33.

4/ Rationalization, under an explicit heuristic. The verdict is fixed at step 0. The reasoning text keeps growing afterwards. Flip rate 0.00. Rationalization score 1.00.

5/ Oscillation. The verdict flips every step and is "stable" only because the trajectory ended. Flip rate 1.00. Rationalization score 0.04, because there is no stable verdict to decorate.

6/ Same accuracy. Three different stories. That is the whole demo. The rationalization number is a timing-and-text heuristic, not proof of intent.

7/ v0.1 does not load weights. You bring JSONL, it returns convergence, flip rate, and that heuristic.

Real checkpoint numbers are the next measurement, not this thread.

Repo: https://github.com/arrdel/traj-eval

---

Post as-is or cut 6 if the thread should stay shorter. Do not add Yeom et al. numbers to this thread. Those are their measurements.
