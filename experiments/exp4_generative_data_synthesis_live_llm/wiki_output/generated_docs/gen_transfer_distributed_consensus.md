# Section 5.4 Re-Enactment: Normalized Cluster State and Bounded-Round Consensus

**Type:** `Distributed Systems & Consensus Protocols (Section 5.4 Re-Enactment)` | **Structural Isomorphism Energy $E_\theta$:** `0.55`

$$
Z_C = Σ_{s∈S_C} exp[-E_C(s)]; q(x) = α q_gossip(x) + (1−α) q_replay(x); s_{t+1} ~ p_C(·|s_t), t=0,…,K−1, with rollout objective D_KL(Q_K || P_C); g_θ(u,v) = A_θ(u,v) · V_qc(u,v)
$$

This architecture treats a replicated cluster’s admissible configurations as a global state space S_C. Each configuration s has an energy E_C(s) encoding protocol constraints, such as log consistency, membership rules, and safety conditions. The cluster state normalizer Z_C sums the weights of all admissible configurations, giving a normalized distribution over global states rather than independent local-node beliefs.

To propose candidate states efficiently, the protocol mixes two sources: q_gossip, which assembles proposals from recent peer exchanges, and q_replay, which resamples logged transitions or previously observed recovery traces. The mixture q supports importance-weighted evaluation of rare but consequential configurations while retaining coverage of states suggested by current gossip traffic.

From a current state, the coordinator performs a bounded K-round gossip rollout. At each round, nodes exchange state summaries and update their local candidate view; the rollout distribution Q_K is shaped toward the protocol-consistent target P_C by a KL divergence objective. The bound K limits communication and decision latency, while the divergence term discourages rollout proposals that depart too far from the target cluster-state distribution.

Finally, a pairwise attention gate assigns weight A_θ(u,v) to the relationship between a proposer u and verifier v, using context such as membership, recency, and observed message consistency. The pairwise quorum-certificate verifier V_qc(u,v) checks signatures, voting eligibility, and certificate compatibility. Their combination g_θ(u,v) makes pairwise quorum-certificate verification the acceptance critic: a candidate transition is admitted only when relevant pairwise evidence is sufficiently weighted and the required quorum conditions hold. Thus normalization, proposal generation, bounded rollout, and pairwise verification form one relational pipeline for consensus-state selection.
