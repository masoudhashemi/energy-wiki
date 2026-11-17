# Allosteric Folding and Ribosomal Proofreading in a Shared Free-Energy Landscape

**Type:** `Molecular Biology / Biochemistry (Adversarial Hard Negative)`  
**Unary Keyword Energy:** `0.2` (Low — fools Unary RAG!) | **Pairwise Relational Energy:** `4.8` (High — rejected by Relation Network!)

$$
P(s) = exp[-βG(s)] / Z, where Z = Σ_s exp[-βG(s)] and G(s) includes conformational and ligand-binding contributions
$$

Allosteric proteins distribute among conformational states according to their free energies: the Boltzmann weights of those states are normalized by the partition function Z. Ligand binding shifts that partition, changing the occupancy of active and inactive structures without requiring a change in the protein’s covalent sequence. In a folding intermediate, a solvent-exposed inner-loop can couple a distant binding pocket to the catalytic site; its backbone flexibility contributes to the conformational ensemble and to the rate of transition between states.

A Langevin description of this motion treats solvent collisions as stochastic forces acting on the protein’s coordinates while preserving the appropriate free-energy landscape. The resulting trajectories can reveal how a ligand stabilizes one basin and alters the probability of productive folding. Along a polypeptide, the textual gradient of residue identities—read here as the ordered biochemical sequence of amino acids and their changing side-chain properties—can be compared with the physical gradient of hydrophobic exposure as folding proceeds.

A related selectivity problem occurs during ribosomal proofreading. The decoding center gives molecular attention to the geometry of codon–anticodon pairing: near-cognate tRNAs are more likely to dissociate before accommodation, while cognate pairing favors forward progression. This kinetic critique is biochemical selection, not an editorial judgment: it reflects differential binding, conformational rearrangement, and GTP-coupled steps. Together, allosteric folding and translational proofreading show how molecular ensembles partition among competing states and how selective transitions enrich functional outcomes.

## Relational Critic Rejection Verdict
The draft uses relevant surface terms such as Boltzmann weights, partition functions, Langevin motion, and textual gradients, and its biological examples are coherent on their own. It does not connect energy-based scoring and stochastic candidate sampling to discrete natural-language critique and editing, and it contains no required wiki-links or discussion of short-run sampling and edit-stagnation risks.
