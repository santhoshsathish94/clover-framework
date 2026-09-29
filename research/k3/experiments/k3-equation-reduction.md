# K3 Equation Reduction

## Purpose

This document records the **Clover equation-reduction exploration** that follows the verified K3 model equation.

The existing `k3-model-equation.md` is an exact reference implementation of the model equation. This document has a different purpose:

> **Find mathematical invariants, cycles, sufficient representations, and result-preserving transformations that can make unnecessary computation disappear.**

The target is stronger than ordinary optimization.

We are looking for reductions analogous to:

[
1+2+3+cdots+n
;ightarrow;
rac{n(n+1)}2
]

The intermediate partial sums are not approximated or merely computed faster. They disappear because they are unnecessary for obtaining the final result.

The guiding Clover cycle is:

[
oxed{	ext{Context}ightarrow	ext{Direction}ightarrow	ext{Action}ightarrow	ext{Success}ightarrow	ext{Context}}
]

Human direction remains explicit. The mathematical observations below are hypotheses and reductions to be tested against the actual K3 execution.

---

# Step 1 — L2 normalization exposes direction

K3's per-head L2 normalization is:

[
hat q=rac{q}{sqrt{|q|^2+epsilon}},
qquad
hat k=rac{k}{sqrt{|k|^2+epsilon}}.
]

Therefore the attention relation is:

[
hat q^That k
=
rac{q^Tk}
{sqrt{(|q|^2+epsilon)(|k|^2+epsilon)}}.
]

For (epsilon=0), this is cosine similarity.

The normalization therefore removes radial magnitude and preserves directional information.

This suggests the first general principle:

> **When the downstream result is invariant to a degree of freedom, that degree of freedom may not need to be represented.**

For K3 the non-zero epsilon means this must not be treated as exact unit-sphere projection without accounting for epsilon.

---

# Step 2 — softmax represents a partition of a whole

For scores (x_i):

[
p_i=rac{e^{x_i}}{sum_j e^{x_j}}
]

and therefore:

[
p_ige0,qquad sum_i p_i=1.
]

Softmax is invariant to a common additive offset:

[
operatorname{softmax}(x)
=
operatorname{softmax}(x+c).
]

Thus absolute score origin is irrelevant; only relative score differences matter.

This is another exact invariance.

---

# Step 3 — remove representations that are only intermediate

Attention can be written as:

[
o=sum_s p_sv_s
=
rac{sum_s e^{r_s}v_s}{sum_s e^{r_s}}.
]

Define:

[
A=sum_s e^{r_s}v_s,
qquad
Z=sum_s e^{r_s}.
]

Then:

[
o=rac AZ.
]

The individual softmax probabilities are therefore not mathematically fundamental to the final weighted sum.

This is an **exact representation reduction**, although it does not by itself prove lower computation.

---

# Step 4 — attention can be viewed as a weighted whole

For a source group (G) with identical relation score (r_G):

[
w_G=e^{r_G},
qquad
V_G=sum_{sin G}v_s.
]

Then the group contribution is:

[
w_GV_G.
]

So exact equal-relations can be grouped.

However, arbitrary K3 keys will not generally have exactly equal scores. Approximate clustering must not be substituted for exact equality when bit-exact reproduction is the success criterion.

---

# Step 5 — MLA's shared rope structure

For MLA:

[
r_{h,s}
=
q_h^{lat}cdot k_{h,s}^{lat}
+
q_h^{rope}cdot k_s^{rope}.
]

The 64-dimensional rope component is shared across all heads.

Therefore:

[
R=Q^{rope}(K^{rope})^T
]

is a shared structural component.

The exponential factor also separates algebraically:

[
e^{L_{h,s}+R_{h,s}}
=
e^{L_{h,s}}e^{R_{h,s}}.
]

This exposes reusable structure, although reuse is not automatically a computation reduction.

---

# Step 6 — attention as a generating function

Under a simplified structural form:

[
k_s=Kz_s,qquad v_s=Vz_s,
]

we get:

[
q^Tk_s=(K^Tq)^Tz_s.
]

Define:

[
a=K^Tq.
]

Then:

[
o(q)
=
rac{sum_s e^{a^Tz_s}Vz_s}
{sum_s e^{a^Tz_s}}.
]

Define the generating function:

[
Z(a)=sum_s e^{a^Tz_s}.
]

Then:

[

abla_a Z(a)
=
sum_s e^{a^Tz_s}z_s
]

and therefore:

[
oxed{
o(q)=V
abla_alog Z(a)
}
]

for this simplified structural model.

This is a useful mathematical unification, but the exact K3 normalization, arithmetic order, and generated states must still be restored before using it as an implementation.

---

# Step 7 — normalized latent states as a distribution

If the relevant latent vectors are normalized, conceptually:

[
|z_s|approx1.
]

Represent the sequence as:

[
mu=sum_sdelta_{z_s}.
]

Then:

[
Z(a)
=
int e^{a^Tz},dmu(z).
]

This says attention can be viewed as operating on a distribution of latent states rather than merely a list of independent vectors.

This is a structural representation, not yet a computational shortcut.

---

# Step 8 — search for a cycle in generated states

If the latent sequence had an exact recurrence:

[
z_{s+1}=F(z_s)
]

and were periodic:

[
F^P(z)=z,
]

then a long sequence could potentially be represented by a finite period.

We must not assume such periodicity.

The correct Clover action is to observe the actual K3 trajectory and test for:

- recurrence,
- periodicity,
- low-dimensional structure,
- invariants.

---

# Step 9 — normalization itself has an invariant

Consider:

[
N_epsilon(x)
=
rac{x}{sqrt{|x|^2+epsilon}}.
]

Write:

[
x=sqrt r,u,qquad |u|=1.
]

Then:

[
N_epsilon(x)
=
sqrt{rac r{r+epsilon}},u.
]

The **direction (u) is invariant**.

Only the scalar radius changes.

For repeated application:

[
r_{n+1}
=
rac{r_n}{r_n+epsilon}.
]

Therefore a high-dimensional repeated normalization reduces to:

[
oxed{	ext{fixed direction}+	ext{one-dimensional scalar recurrence}}
]

This is a genuine algebraic reduction of the repeated operator.

---

# Step 10 — repeated normalization has a scalar closed recurrence

Because:

[
r_{n+1}=rac{r_n}{r_n+epsilon},
]

the vector does not need to be repeatedly represented to understand its direction.

The repeated high-dimensional operation has collapsed to one scalar recurrence plus an invariant direction.

This is the type of structure we are searching for in K3: repeated computation whose apparent dimensionality is larger than the information actually changing.

---

# Step 11 — RMSNorm and linear projections

K3 RMSNorm has the form:

[
N(x;w)=ho(x)(wodot x)
]

with scalar:

[
ho(x)
=
left(
rac1n|x|^2+epsilon
ight)^{-1/2}.
]

For a linear projection (W):

[
WN(x;w)
=
ho(x)W(wodot x).
]

The scalar normalization factor can therefore be factored through a purely linear region.

But it cannot generally be moved through nonlinear operations such as sigmoid or softmax.

This gives a useful rule:

> **Scalar factors can travel through linear regions; nonlinearities are information boundaries.**

---

# Step 12 — SiTU has two representations of one source state

K3 SiTU is:

[
SiTU(g,u)
=
[b_1	anh(g/b_1)sigma(g)]
[b_2	anh(u/b_2)]
]

with:

[
g=Gx,qquad u=Ux.
]

Define:

[
f(g)=b_1	anh(g/b_1)sigma(g),
]

[
h(u)=b_2	anh(u/b_2).
]

Then:

[
SiTU(Gx,Ux)=f(Gx)odot h(Ux).
]

The two projected representations originate from the same (x), then interact element-wise.

This is another instance of:

[
	ext{one state}ightarrow	ext{multiple representations}ightarrow	ext{relation}ightarrow	ext{result}.
]

Approximate Taylor expansion may be useful for analysis, but it is not an exact reduction.

---

# Step 13 — KDA is a structured state transformation

KDA begins with:

[
q=W_qx,qquad k=W_kx,qquad v=W_vx.
]

Again, three representations derive from one source state.

The recurrent state is:

[
Sleftarrow D_alpha S,
]

[
u=S^Tk,
]

[
Sleftarrow S+eta k(v-u)^T,
]

[
o=S^Tq.
]

Algebraically:

[
S'
=
(D_alpha-eta kk^T)S+eta kv^T.
]

Define:

[
A=D_alpha-eta kk^T,
qquad
b=eta kv^T.
]

Then:

[
oxed{S'=AS+b}
]

where (A) is diagonal plus rank-1.

This is the first major state-reduction target.

---

# Step 14 — exact mathematical elimination of the KDA state

For:

[
S_t=A_tS_{t-1}+b_t,
]

we have:

[
S_t^T=S_{t-1}^TA_t^T+b_t^T.
]

For a requested current query (q_t), define:

[
y_t=q_t,
qquad
y_{i-1}=A_i^Ty_i.
]

Then:

[
S_t^Tq_t
=
S_0^Ty_0
+
sum_{i=1}^{t}b_i^Ty_i.
]

Since:

[
A_i^T=D_{alpha_i}-eta_i k_ik_i^T,
]

we obtain:

[
oxed{
y_{i-1}
=
D_{alpha_i}y_i
-
eta_i k_i(k_i^Ty_i)
}
]

and:

[
oxed{
b_i^Ty_i
=
eta_i v_i(k_i^Ty_i).
}
]

Therefore:

[
oxed{
o_t
=
S_0^Ty_0
+
sum_i
eta_i v_i(k_i^Ty_i)
}
]

in exact real arithmetic.

The full (128	imes128) state is not required to express the requested projection mathematically.

However, this is **not yet a bit-exact K3 replacement**, because K3 performs float32 rounding at each original state update.

---

# Step 15 — the float32 rounding barrier

The real-arithmetic identity does not automatically preserve K3's execution.

In general:

[
operatorname{round}_{32}(AS+b)^Tq

eq
operatorname{round}_{32}ig((AS+b)^Tqig).
]

K3 performs:

1. diagonal scaling of (S),
2. sequential float32 (S^Tk),
3. float32 rank-1 update,
4. sequential float32 (S^Tq).

The backward formulation changes this operation order.

Therefore:

> **Mathematical equality and bit-exact K3 equality are separate claims.**

The next test must preserve or explicitly characterize the rounding path.

---

# Step 16 — KDA update has rank-1 correction structure

For each column:

[
s'=
D_alpha s+
eta k(v-k^Ts).
]

Equivalently:

[
s'
=
(D_alpha-eta kk^T)s+eta kv.
]

The update is:

[
oxed{	ext{diagonal scaling}+	ext{rank-1 correction}.}
]

This is highly structured compared with an arbitrary (128	imes128) transformation.

---

# Step 17 — rank of the key sequence

Let:

[
K=[k_1,k_2,ldots,k_t].
]

One possible reduction path is whether:

[
operatorname{rank}(K)ll128.
]

But this is not sufficient.

The diagonal operation:

[
D_alpha S
]

can alter the effective subspace coordinate-wise.

Therefore:

[
operatorname{rank}(K)ll128
]

does not prove that (S) itself is low-rank or that a compressed state is sufficient.

The actual observation must include both key-space structure and state-space behavior.

---

# Step 18 — move from state space to dual/query space

The requested result is:

[
o=S^Tq.
]

Rather than maintaining the entire state, propagate the query backward:

[
y_{i-1}
=
D_{alpha_i}y_i
-
eta_i k_i(k_i^Ty_i).
]

The diagonal state transformation becomes a diagonal query transformation.

The rank-1 state correction becomes a rank-1 query correction.

This is a **dual representation** of the same requested projection.

The dimensionality has moved from:

[
128	imes128=16,384
]

state elements to:

[
128
]

query elements for one requested projection.

That is a substantial mathematical reduction in representation, but not yet an end-to-end speedup.

---

# Step 19 — only one scalar couples each KDA step

Define:

[
c_i=k_i^Ty_i.
]

Then:

[
y_{i-1}
=
alpha_iodot y_i-eta_i k_i c_i.
]

And the output contribution is:

[
Delta o_i
=
eta_i v_i c_i.
]

Thus each historical KDA step communicates with the backward query through one scalar:

[
oxed{c_i=k_i^Ty_i}.
]

This exposes a scalar interaction chain hidden inside the matrix recurrence.

---

# Step 20 — expanding the scalar chain

Substitute:

[
y_i=
alpha_{i+1}odot y_{i+1}
-eta_{i+1}k_{i+1}c_{i+1}.
]

Then:

[
c_i
=
(k_iodotalpha_{i+1})^Ty_{i+1}
-
eta_{i+1}
(k_i^Tk_{i+1})c_{i+1}.
]

Further expansion produces weighted key-to-key interactions.

The matrix state has disappeared, but arbitrary histories can generate many independent interactions.

Therefore we must not mistake elimination of the explicit state for elimination of the underlying information.

---

# Step 21 — cumulative diagonal structure

Ignoring the rank-1 corrections temporarily:

[
y_i
=
q_todot
prod_{j=i+1}^{t}alpha_j.
]

Define the cumulative element-wise product:

[
P_i=prod_{j=1}^{i}alpha_j.
]

Then:

[
prod_{j=i+1}^{m-1}alpha_j
=
P_{m-1}oslash P_i
]

when the required components are non-zero.

Thus long sequences of diagonal transformations share a cumulative structure.

This is another exact algebraic simplification.

---

# Step 22 — transformed-key representation

Define schematically:

[
	ilde k_i=k_ioslash P_{i-1}.
]

Then the expanded scalar interactions can be expressed through cumulative-coordinate transformations of the keys.

This gives a concrete empirical question:

> Do the transformed K3 keys contain repeated directions, a low-dimensional basis, a recurrence, or another exact invariant?

If they do, the history may admit a compact sufficient representation.

If they do not, the sequence is carrying genuinely growing information.

---

# Step 23 — bounded sufficient state versus growing information

For a history (H), seek a summary (C_t) such that:

[
C_{t+1}
=
F(C_t,k_t,v_t,alpha_t,eta_t)
]

and:

[
o_t=G(C_t,q_t).
]

If (C_t) has fixed size independent of sequence length, then the history has a **bounded sufficient state**.

If the required summary grows with (t), the history carries increasing independent information.

This is the key distinction between:

- compression of existing information, and
- a true elimination of repeated computation.

---

# Step 24 — composition algebra

A KDA history can be represented mathematically by:

[
F_H(S)=A_HS+b_H.
]

Two histories compose as:

[
F_B(F_A(S))
=
(A_BA_A)S+(A_Bb_A+b_B).
]

Therefore:

[
oxed{
(A_B,b_B)circ(A_A,b_A)
=
(A_BA_A,;A_Bb_A+b_B).
}
]

This is a genuine composition algebra.

If a compact representation remains closed under this composition, long histories could potentially be composed hierarchically instead of replaying every step.

However, the product of diagonal-plus-rank-1 matrices is generally not diagonal-plus-rank-1.

So the obvious representation is not closed.

---

# Step 25 — compose the action instead of the matrix

For a history (H), define its backward action:

[
T_H(q)=A_H^Tq.
]

For two histories:

[
T_{Acirc B}(q)
=
T_A(T_B(q)).
]

A single step is:

[
T_i(q)
=
D_{alpha_i}q
-
eta_i k_i(k_i^Tq).
]

It consists of:

1. element-wise scaling,
2. one scalar dot product,
3. one rank-1 correction.

But composition can accumulate independent directions.

Therefore rank-1 per update does not imply rank-1 for an arbitrarily long history.

---

# Step 26 — arbitrary future query versus known current query

For arbitrary future queries:

[
T_H(q)=M_Hq
]

for:

[
M_H=A_1^TA_2^Tcdots A_t^T.
]

A representation supporting **every possible future query** may require enough information to represent (M_H).

But for one known current query (q_t), we only need:

[
M_Hq_t.
]

This distinction is crucial:

[
oxed{
	ext{arbitrary future query}

eq
	ext{known current query}.
}
]

The full KDA state exists partly because it preserves information usable by future queries.

A result-directed calculation can potentially retain only the information required for the current query.

---

# Step 27 — reduce the final target

The K3 tail produces:

[
ellinmathbb R^{163840}
]

and chooses:

[
j^*=argmax_jell_j.
]

Therefore the ultimate result is not necessarily the entire logit vector.

It is the identity of the maximum.

This suggests exact result-directed elimination through bounds.

If:

[
L_jleell_jle U_j
]

and candidate (a) satisfies:

[
L_a>max_{j
e a}U_j,
]

then:

[
oxed{a=argmax_jell_j}
]

without evaluating the remaining logits.

This is an exact elimination principle.

---

# Step 28 — final RMSNorm scalar is irrelevant to argmax

The final normalized state is:

[
x'_i=(w_ix_i)ho
]

where:

[
ho>0.
]

The lm_head is linear:

[
ell_j
=
ho
sum_iW_{ji}(w_ix_i).
]

Therefore:

[
oxed{
argmax_jell_j
=
argmax_j
sum_iW_{ji}(w_ix_i)
}
]

The final scalar normalization factor cannot affect the selected token.

So for next-token selection, that scalar computation can disappear exactly.

This is a genuine result-preserving reduction, not an approximation.

---

# Step 29 — vocabulary scores only require differences

For two vocabulary candidates:

[
z_a-z_b
=
(W_a-W_b)^T(wodot x).
]

A common additive value also disappears:

[
argmax_j(z_j+c)
=
argmax_jz_j.
]

Thus the final decision depends on vocabulary-row differences rather than absolute score origin.

The decision space is therefore a quotient of logit space by the common-all-ones direction.

---

# Step 30 — centered lm_head

Let:

[
ar W=rac1Vsum_{j=1}^{V}W_j
]

and:

[
Delta W_j=W_j-ar W.
]

Then:

[
W_j^Tx
=
ar W^Tx+Delta W_j^Tx.
]

The common term is identical for every token, so:

[
oxed{
argmax_jW_j^Tx
=
argmax_jDelta W_j^Tx.
}
]

This is another exact invariant.

However, removing a common component does not automatically reduce the number of vocabulary dot products.

---

# Step 31 — vocabulary decision as maximum inner-product search

For a fixed hidden query (x):

[
s_j=W_j^Tx.
]

For candidates (a,b):

[
s_a>s_b
iff
(W_a-W_b)^Tx>0.
]

Each pair defines a hyperplane:

[
(W_a-W_b)^Tx=0.
]

The hidden-state space is therefore partitioned into regions where the same token is the argmax.

The final computation can be viewed geometrically as:

[
oxed{
	ext{identify the vocabulary decision region containing }x.
}
]

This suggests hierarchical exact search if the vocabulary geometry admits safe bounds.

---

# Step 32 — positive scaling is irrelevant at the final decision

For (c>0):

[
argmax_jW_j^T(cx)
=
argmax_jW_j^Tx.
]

Therefore the final decision depends on the **ray/direction** of the weighted hidden state.

However, this invariance belongs to the final linear-plus-argmax boundary.

It cannot automatically be propagated backward through nonlinear K3 operations.

This establishes another important rule:

> **An invariance is only valid across the operations that preserve it.**

---

# Step 33 — trace the decision boundary, not the winner

For two candidate tokens:

[
d=W_a-W_b.
]

The final ordering depends only on:

[
d^Tx.
]

The boundary is:

[
d^Tx=0.
]

Therefore, instead of calculating the entire final state, a result-directed calculation can seek the sign of:

[
oxed{d^TF(x_L)}
]

where (F) is the remaining transformer computation.

For a residual layer:

[
x_{	ext{out}}
=
x_{	ext{in}}+F(x_{	ext{in}})
]

we have:

[
d^Tx_{	ext{out}}
=
d^Tx_{	ext{in}}
+
d^TF(x_{	ext{in}}).
]

Linear transformations can move the decision direction backward:

[
d^T(Wx)
=
(W^Td)^Tx.
]

Nonlinear operations remain information boundaries.

---

# Step 34 — exact bounds through nonlinear operations

For a monotonic nonlinearity (sigma), if:

[
l_ile z_ile u_i,
]

then:

[
sigma(l_i)lesigma(z_i)lesigma(u_i).
]

Therefore a decision-direction contribution can sometimes be bounded without evaluating the exact nonlinear output.

If the accumulated decision margin (M) exceeds the maximum possible contribution of all remaining unevaluated terms:

[
|M|>sum_{	ext{remaining}}B_i,
]

the decision is already determined exactly.

This gives a general exact branch-and-bound principle.

But bounds can become too loose and merely relocate computation.

---

# Step 35 — SiTU exposes an information boundary

K3 SiTU is:

[
SiTU(g,u)=f(g)odot h(u)
]

where:

[
g=Gx,qquad u=Ux.
]

For decision direction (d):

[
d^TSiTU
=
sum_i d_if(g_i)h(u_i).
]

Each output coordinate depends on the pair:

[
(g_i,u_i).
]

Unlike a linear transformation, the decision direction cannot simply be pushed through the nonlinear multiplication.

Therefore SiTU is an **information boundary** unless its generated inputs contain additional exact structure.

---

# Step 36 — universal SiTU bounds

Because:

[
|	anh(z)|<1,
qquad
0<sigma(g)<1,
]

we have:

[
|f(g)|<b_1,
qquad
|h(u)|<b_2.
]

Therefore:

[
|f(g)h(u)|<b_1b_2
]

and:

[
|d_if(g_i)h(u_i)|
<
|d_i|b_1b_2.
]

These are exact contribution bounds.

But they may be too loose to eliminate meaningful work.

---

# Step 37 — trajectory-specific SiTU bounds

If we know:

[
g_iin[l_g,u_g],
qquad
u_iin[l_u,u_u],
]

then:

[
B_i
=
max_{gin[l_g,u_g]}|f(g)|
;
max_{uin[l_u,u_u]}|h(u)|.
]

Hence:

[
|d_if(g_i)h(u_i)|
le
|d_i|B_i.
]

For an accumulated margin (M), define:

[
R=sum_{	ext{remaining}}|d_i|B_i.
]

If:

[
|M|>R,
]

the sign of the final decision is fixed exactly.

This is potentially useful branch-and-bound elimination.

---

# Step 38 — bounds can merely relocate computation

The intervals for (g_i,u_i) originate from:

[
g=Gx,qquad u=Ux.
]

If obtaining tight intervals requires computing the entire projections, then the apparent reduction has only moved the original cost earlier.

Loose norm bounds may avoid some projections but can become too weak to prove a decision.

Therefore:

> **A valid bound is not automatically a computational reduction.**

The reduction must eliminate actual computation, not merely rename it.

---

# Step 39 — SiTU channel structure is the next empirical target

Because:

[
(g_i,u_i)
]

are both generated from the same (x), search the real K3 trajectory for exact structure among the pairs:

[
(g_0,u_0),ldots,(g_{7167},u_{7167}).
]

Candidate exact structures include:

- repeated pairs,
- deterministic channel transformations,
- low-dimensional subspaces,
- recurrence,
- exact symmetry,
- other invariants.

If multiple channels can be generated from a smaller exact representation, the nonlinear work may collapse.

If the pairs are independent, the nonlinear channels are carrying independent information and should not be artificially reduced.

---

# Step 40 — reduction versus compression

At this point we have several classes of transformations:

### Exact invariance

Example:

[
argmax_j(ho z_j)
=
argmax_jz_j,qquad ho>0.
]

A computation can disappear.

### Exact representation reduction

Example:

[
S^Tq
]

can be expressed through a backward 128-dimensional query instead of explicitly forming (S), in real arithmetic.

The representation is smaller, but K3 arithmetic and total work still need validation.

### Compression

A smaller representation may reproduce the same information but still require essentially the same information-generation work.

Compression is not automatically the desired closed-form reduction.

### True closed-form elimination

The strongest target is:

[
oxed{
	ext{repeated computation}
ightarrow
	ext{a smaller mathematical relation}
}
]

where intermediate computation genuinely disappears, analogous to:

[
1+2+cdots+n
ightarrow
rac{n(n+1)}2.
]

This is the standard required for claiming a true equation reduction.

---

# Step 41 — current boundary and next Clover action

The latest observation is:

[
SiTU(Gx,Ux)
=
f(Gx)odot h(Ux).
]

The nonlinear multiplication is an information boundary.

The next empirical question is therefore:

[
oxed{
	ext{What exact structure exists in the actual }(g_i,u_i)	ext{ pairs?}
}
]

The test should use the real K3 execution rather than assuming a cycle.

For the actual K3 trajectory:

1. capture (g_i) and (u_i);
2. test exact equality/repetition;
3. measure numerical and exact rank structure;
4. test recurrence candidates;
5. test whether groups of channels share an exact generating relation;
6. only then derive a candidate reduced equation;
7. validate the candidate against K3's required arithmetic;
8. measure whether actual computation disappears.

---

# Clover reduction principles established so far

## 1. Start from the result

Do not automatically preserve every intermediate state.

Ask:

[
oxed{	ext{What information does the requested result actually depend on?}}
]

## 2. Find invariants

If:

[
R(x)=R(y)
]

for all downstream-relevant cases, then the difference between (x) and (y) is unnecessary to the result.

## 3. Respect nonlinear boundaries

Linear transformations allow dual movement:

[
d^TWx=(W^Td)^Tx.
]

Nonlinear transformations generally do not.

## 4. Distinguish mathematics from implementation arithmetic

A real-number identity is not automatically a bit-exact K3 identity.

The verified K3 equation makes arithmetic order part of the model.

## 5. Distinguish compression from elimination

A smaller representation is useful.

But the stronger target is:

[
oxed{	ext{computation that no longer needs to happen}.}
]

## 6. Let observation decide whether the cycle exists

The circle/cycle idea is a search direction, not evidence.

The actual K3 trajectory must reveal the invariant.

---

# Current status

The verified K3 equation remains the reference:

- 93 layers;
- 24 MLA;
- 69 KDA;
- 1 dense MLP;
- 92 MoE;
- exact prefill reproduction;
- exact one-step decode reproduction;
- final token reproduction;
- bit-exact verification where documented in `k3-model-equation.md`.

This reduction document does **not** claim that K3 has already been reduced.

It records the sequence of mathematical reductions and the exact questions required to discover whether a true reduced equation exists.

The next action is empirical:

[
oxed{
	ext{Observe the actual generated states}
ightarrow
	ext{find an exact invariant}
ightarrow
	ext{derive the reduced equation}
ightarrow
	ext{validate}
ightarrow
	ext{measure eliminated computation}.
}
]

That is the Clover loop.
