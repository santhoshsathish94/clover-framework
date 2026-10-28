"""
Executive routine (`Ex`) and Proof Methods (`MSb`, `MDt`, `MCh`, `CX`) for Newell & Simon (1956) P-868,
plus experimental drivers for E2, E3, and E4.
"""
from typing import Dict, List, Optional, Tuple
from lt1956_engine import (
    Expr,
    parse_pm_expr,
    view_variables_vv,
    view_contracted_vct,
    describe_d,
    test_similarity_csm,
    match_lmc,
)


def compare_expressions_cx(x: Expr, y: Expr) -> bool:
    """CX (p. 57): Exact structural and variable-name equality check (used to reject duplicate subproblems on Q)."""
    if x.neg != y.neg or x.is_var != y.is_var:
        return False
    if x.is_var:
        return x.name == y.name
    if x.conn != y.conn:
        return False
    return compare_expressions_cx(x.left, y.left) and compare_expressions_cx(x.right, y.right)


def clear_free_flags(e: Expr) -> None:
    """Marks all variables in a newly created subproblem as bound (non-free), matching PFB in logic-routines.txt."""
    e.free = False
    if not e.is_var:
        clear_free_flags(e.left)
        clear_free_flags(e.right)


class LogicTheorist1956:
    def __init__(
        self,
        theorems: List[Tuple[str, Expr]],
        use_csm: bool = True,
        use_vct: bool = True,
        priority_mode: str = "min_k",  # "min_k" (Ex(CK) p. 46), "fifo", or "max_k"
        work_limit: int = 30,
    ):
        self.theorems = [(tid, t.clone()) for tid, t in theorems]
        self.use_csm = use_csm
        self.use_vct = use_vct
        self.priority_mode = priority_mode
        self.work_limit = work_limit
        self.work_w = 0
        self.stats: Dict[str, int] = {"csm_calls": 0, "csm_passed": 0, "lmc_top_calls": 0}
        self.P: List[Tuple[Expr, str]] = []
        self.Q: List[Expr] = []

    def method_msb(self, target: Expr) -> Optional[str]:
        """MSb (p. 52): Prove `target` by substitution & replacement from theorem list T."""
        self.work_w += 1
        view_variables_vv(target)
        for tid, thm in self.theorems:
            view_variables_vv(thm)
            if self.use_csm and not test_similarity_csm(target, thm, self.stats):
                continue
            t_copy = thm.clone()
            p_copy = target.clone()
            if match_lmc(t_copy, t_copy, p_copy, p_copy, self.stats):
                return f"MSb({tid})"
        return None

    def method_mdt(self, target: Expr) -> Optional[str]:
        """MDt (p. 52): Prove `target` by detachment from T -> (B -> A'), matching A' to target and MSb on B."""
        for tid, thm in self.theorems:
            if thm.is_var or thm.neg != 0 or thm.conn != "->":
                continue
            view_variables_vv(thm)
            right_t = thm.right
            matched_csm = False
            if not self.use_csm:
                matched_csm = True
            else:
                view_variables_vv(target)
                if test_similarity_csm(target, right_t, self.stats):
                    matched_csm = True
                elif self.use_vct:
                    view_contracted_vct(target)
                    if test_similarity_csm(target, right_t, self.stats):
                        matched_csm = True
            if not matched_csm:
                continue

            t_copy = thm.clone()
            p_copy = target.clone()
            if not match_lmc(t_copy.right, t_copy, p_copy, p_copy, self.stats):
                continue
            subprob = t_copy.left.clone()
            clear_free_flags(subprob)
            self.P.append((subprob, f"MDt({tid})"))
            sub_res = self.method_msb(subprob)
            if sub_res is not None:
                return f"MDt({tid}) + {sub_res}"
        return None

    def method_mch(self, target: Expr) -> Optional[str]:
        """MCh (p. 51): Prove `target` (A -> C) by forward or backward chaining with T (T1 -> T2)."""
        if target.is_var or target.neg != 0 or target.conn != "->":
            return None
        view_variables_vv(target)
        left_p = target.left
        right_p = target.right

        for tid, thm in self.theorems:
            if thm.is_var or thm.neg != 0 or thm.conn != "->":
                continue
            view_variables_vv(thm)
            t_copy = thm.clone()
            left_t = t_copy.left
            right_t = t_copy.right

            # Forward chaining (SmF / McF): match left_t to left_p
            if (not self.use_csm) or test_similarity_csm(left_p, left_t, self.stats):
                p_copy = target.clone()
                if match_lmc(t_copy.left, t_copy, p_copy.left, p_copy, self.stats):
                    # Segment E (McR): check if right_t already matches right_p!
                    t_copy2 = t_copy.clone()
                    p_copy2 = p_copy.clone()
                    if match_lmc(p_copy2.right, p_copy2, t_copy2.right, t_copy2, self.stats):
                        return f"MCh_direct({tid})"
                    # Create new problem: t_copy.right -> p_copy.right
                    subprob = Expr(neg=0, is_var=False, free=False, conn="->",
                                   left=t_copy.right.clone(), right=p_copy.right.clone())
                    clear_free_flags(subprob)
                    self.P.append((subprob, f"MCh_fwd({tid})"))
                    sub_res = self.method_msb(subprob)
                    if sub_res is not None:
                        return f"MCh_fwd({tid}) + {sub_res}"
                    continue  # In P-868 p. 51, MSb 7 - - C branches to C (next theorem in T), skipping B (backward chaining)!

            # Backward chaining (SmB / McB): match right_t to right_p
            t_copy_b = thm.clone()
            view_variables_vv(t_copy_b)
            if (not self.use_csm) or test_similarity_csm(right_p, t_copy_b.right, self.stats):
                p_copy_b = target.clone()
                if match_lmc(t_copy_b.right, t_copy_b, p_copy_b.right, p_copy_b, self.stats):
                    subprob = Expr(neg=0, is_var=False, free=False, conn="->",
                                   left=p_copy_b.left.clone(), right=t_copy_b.left.clone())
                    clear_free_flags(subprob)
                    self.P.append((subprob, f"MCh_bwd({tid})"))
                    sub_res = self.method_msb(subprob)
                    if sub_res is not None:
                        return f"MCh_bwd({tid}) + {sub_res}"
        return None

    def prove_executive(self, problem: Expr) -> Tuple[bool, Optional[str], int]:
        """Ex (p. 50): Executive routine with MSb, MDt, MCh, work limit W, priority selection on P, and dedup on Q."""
        res = self.method_msb(problem)
        if res is not None:
            return True, res, 0

        curr = problem.clone()
        curr_trace = "root"
        depth = 0
        while True:
            res = self.method_mdt(curr)
            if res is not None:
                return True, f"{curr_trace} -> {res}", depth
            res = self.method_mch(curr)
            if res is not None:
                return True, f"{curr_trace} -> {res}", depth

            self.Q.append(curr)
            if self.work_w > self.work_limit:
                return False, None, depth

            # Select next subproblem from P (Ex(CK) + Ex(CX))
            picked = None
            while self.P:
                if self.priority_mode == "fifo":
                    best_idx = 0
                elif self.priority_mode == "min_k":
                    best_idx = 0
                    best_k = self._level_k(self.P[0][0])
                    for idx in range(1, len(self.P)):
                        k_i = self._level_k(self.P[idx][0])
                        # In P-868 p. 50 Ex(CK), `CKG 2 1 - C` branches only if K(2) > K(1),
                        # so when K(2) <= K(1) it overwrites 1 with 2 (selecting the last tied min-K item).
                        if k_i <= best_k:
                            best_k = k_i
                            best_idx = idx
                elif self.priority_mode == "max_k":
                    best_idx = 0
                    best_k = self._level_k(self.P[0][0])
                    for idx in range(1, len(self.P)):
                        k_i = self._level_k(self.P[idx][0])
                        if k_i > best_k:
                            best_k = k_i
                            best_idx = idx
                else:
                    raise ValueError(self.priority_mode)

                cand, cand_trace = self.P.pop(best_idx)
                if not any(compare_expressions_cx(cand, q_item) for q_item in self.Q):
                    picked = (cand, cand_trace)
                    break

            if picked is None:
                return False, None, depth
            curr, curr_trace = picked
            depth += 1

    @staticmethod
    def _level_k(e: Expr) -> int:
        view_variables_vv(e)
        return describe_d(e)[0]
