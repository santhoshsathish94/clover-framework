"""
Self-contained implementation of Newell & Simon (1956) RAND P-868 Logic Theory Machine (`lt1956_engine.py`),
plus parser for `dmoews/logic-theorist` IPL-I campaign output (E2, E3, E4).
"""
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Expr:
    """
    Represents a propositional expression element in LL (P-868 pp. 9-11):
      - neg (G): number of leading negation signs (-) on this node (>= 0)
      - is_var (V): True if variable, False if binary connective ('v' or '->')
      - free (F): True if free variable (available for substitution)
      - conn (C): 'v' or '->' (None if variable)
      - name (N): variable name ('p','q','r','s','A','B',...) or contracted unit name
      - left, right: child Expr nodes (None if variable)
      - unit (U): whether this node is currently viewed as a unit (for D=(K,J,H))
    """
    neg: int = 0
    is_var: bool = True
    free: bool = False
    conn: Optional[str] = None
    name: Optional[str] = None
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None
    unit: bool = False

    def clone(self) -> "Expr":
        return copy.deepcopy(self)

    def to_str(self) -> str:
        prefix = "~ " * self.neg
        if self.is_var:
            return f"{prefix}{self.name}"
        op = "\\/" if self.conn == "v" else "->"
        return f"{prefix}( {self.left.to_str()} {op} {self.right.to_str()} )"


def parse_pm_expr(text: str, free_vars: bool = False) -> Expr:
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def parse_item() -> Expr:
        nonlocal pos
        neg = 0
        while pos < len(tokens) and tokens[pos] == "~":
            neg += 1
            pos += 1
        if tokens[pos] == "(":
            pos += 1
            left = parse_item()
            op_tok = tokens[pos]
            pos += 1
            conn = "v" if op_tok in ("\\/", "v") else "->"
            right = parse_item()
            assert tokens[pos] == ")"
            pos += 1
            return Expr(neg=neg, is_var=False, free=False, conn=conn, left=left, right=right)
        else:
            vname = tokens[pos]
            pos += 1
            return Expr(neg=neg, is_var=True, free=free_vars, name=vname)

    root = parse_item()
    assert pos == len(tokens)
    return root


# --- Viewing routines VV and VCt (P-868 pp. 55-56) and Description D=(K,J,H) (pp. 58-59) ---

def view_variables_vv(e: Expr) -> None:
    """VV (p. 55): View variables as units."""
    e.unit = bool(e.is_var)
    if not e.is_var:
        view_variables_vv(e.left)
        view_variables_vv(e.right)


def view_contracted_vct(e: Expr, counter: List[int] = None) -> None:
    """
    VCt (p. 56): Make units of binary expressions (whose left and right children are both variables)
    and isolated variables.
    """
    if counter is None:
        counter = [1]
    if e.is_var:
        e.unit = True
        return
    if e.left.is_var and e.right.is_var:
        e.left.unit = False
        e.right.unit = False
        if not e.name:
            e.name = f"ct_{counter[0]}"
            counter[0] += 1
        e.unit = True
    else:
        e.unit = False
        if e.left.is_var:
            e.left.unit = True
        else:
            view_contracted_vct(e.left, counter)
        if e.right.is_var:
            e.right.unit = True
        else:
            view_contracted_vct(e.right, counter)


def describe_d(e: Optional[Expr]) -> Tuple[int, int, int]:
    """
    Computes D(e) = (K, J, H) (P-868 pp. 11, 58-59):
      K: number of levels down to units (1 if e is a unit)
      J: number of distinct unit names in e (strict subelements in NH/NJ, or 1 if e is a unit;
         note that in P-868 p. 32 DL(2.01)=(2,1,2), DR(2.01)=(1,1,1) for -p where -p is a single variable!)
      H: number of unit places in e
    """
    if e is None:
        return (0, 0, 0)

    def level_k(node: Optional[Expr]) -> int:
        if node is None:
            return 0
        if node.unit:
            return 1
        return 1 + max(level_k(node.left), level_k(node.right))

    unit_names: List[str] = []

    def collect_units(node: Optional[Expr]) -> None:
        if node is None:
            return
        if node.unit:
            unit_names.append(node.name or node.to_str())
            return
        collect_units(node.left)
        collect_units(node.right)

    collect_units(e)
    k_val = level_k(e)
    h_val = len(unit_names)
    j_val = len(set(unit_names))
    return (k_val, j_val, h_val)


def test_similarity_csm(x: Expr, y: Expr, stats: Dict[str, int]) -> bool:
    """
    CSm (p. 57): Compare left and right descriptions DL(x)==DL(y) and DR(x)==DR(y).
    In P-868 p. 57, CSm calls FL/FR on x and y; if a node is a variable, its left/right subelements are blank (0,0,0).
    """
    stats["csm_calls"] = stats.get("csm_calls", 0) + 1
    dl_x = describe_d(x.left) if not x.is_var else (0, 0, 0)
    dr_x = describe_d(x.right) if not x.is_var else (0, 0, 0)
    dl_y = describe_d(y.left) if not y.is_var else (0, 0, 0)
    dr_y = describe_d(y.right) if not y.is_var else (0, 0, 0)
    ok = (dl_x == dl_y) and (dr_x == dr_y)
    if ok:
        stats["csm_passed"] = stats.get("csm_passed", 0) + 1
    return ok


# --- Substitution LSb, Replacement LRp->v / LRpv->, and Matching LMc (pp. 53-55) ---

def contains_var_name(e: Expr, vname: str) -> bool:
    if e.is_var:
        return e.name == vname
    return contains_var_name(e.left, vname) or contains_var_name(e.right, vname)


def substitute_lsb(sub_expr: Expr, var_name: str, root: Expr, fresh_counter: List[int]) -> None:
    """
    LSb (p. 54): Substitute `sub_expr` for every free variable with name `var_name` inside `root`.
    If `sub_expr` itself contains `var_name`, first rename `var_name` in `root` to an unused name (segment G of LSb).
    """
    if contains_var_name(sub_expr, var_name):
        fresh_counter[0] += 1
        unused = f"_U{fresh_counter[0]}"
        rename_free_var(root, var_name, unused)
        var_name = unused

    def _replace(node: Expr) -> None:
        if node.is_var:
            if node.free and node.name == var_name:
                rep = sub_expr.clone()
                node.neg = node.neg + rep.neg
                node.is_var = rep.is_var
                node.free = rep.free
                node.conn = rep.conn
                node.name = rep.name
                node.left = rep.left
                node.right = rep.right
        else:
            _replace(node.left)
            _replace(node.right)

    _replace(root)


def rename_free_var(node: Expr, old_name: str, new_name: str) -> None:
    if node.is_var:
        if node.free and node.name == old_name:
            node.name = new_name
    else:
        rename_free_var(node.left, old_name, new_name)
        rename_free_var(node.right, old_name, new_name)


def replace_imp_to_or(node: Expr) -> bool:
    """LRp->v (p. 54): If C(x) == '->', replace (A -> B) with (~A v B)."""
    if node.is_var or node.conn != "->":
        return False
    node.conn = "v"
    node.left.neg += 1
    return True


def replace_or_to_imp(node: Expr, root_x: Expr, fresh_counter: List[int]) -> bool:
    """
    LRpv-> (p. 55): If C(x) == 'v', replace (~A v B) with (A -> B).
    If G(EL(x)) > 0, decrement G(EL(x)) and set C(x) = '->'.
    If G(EL(x)) == 0 and EL(x) is a FREE variable, substitute ~EL(x) for EL(x) across `root_x` (segment B of LRpv->, p. 35 & p. 55)
    and then set C(x) = '->' and decrement G(EL(x))!
    """
    if node.is_var or node.conn != "v":
        return False
    if node.left.neg > 0:
        node.conn = "->"
        node.left.neg -= 1
        return True
    if node.left.is_var and node.left.free:
        vname = node.left.name
        neg_var = Expr(neg=1, is_var=True, free=True, name=vname)
        substitute_lsb(neg_var, vname, root_x, fresh_counter)
        assert node.left.neg > 0
        node.conn = "->"
        node.left.neg -= 1
        return True
    return False


def match_lmc(x: Expr, root_x: Expr, y: Expr, root_y: Expr, stats: Dict[str, int], fresh_counter: List[int] = None) -> bool:
    """
    LMc (p. 53): Match X(x) ( inside root_x ) to X(y) ( inside root_y ).
    Follows all 6 segments of LMc on p. 53 of P-868:
      - Segment T: compare G(x) and G(y)
      - Segment Sby (A, B): if G(y) > G(x) or E(x) is a free variable, substitute y (with G(y)-G(x)) for E(x) in root_x
      - Segment Sbx (C, D): if G(x) > G(y) or E(y) is a free variable, substitute x (with G(x)-G(y)) for E(y) in root_y
      - Segment CN (E): if both variables with equal G and not free, check N(x) == N(y)
      - Segment Rp (F, G): if connectives differ, try LRp->v(x) or LRpv->(x) and recurse
      - Segment LMc: recurse on left children then right children
    """
    if fresh_counter is None:
        fresh_counter = [0]
        stats["lmc_top_calls"] = stats.get("lmc_top_calls", 0) + 1

    # CGG C L -> A (G(y) > G(x))
    if y.neg > x.neg:
        if not x.is_var or not x.free:
            return False
        sub = y.clone()
        sub.neg = y.neg - x.neg
        substitute_lsb(sub, x.name, root_x, fresh_counter)
        return True

    # CGG L C -> C (G(x) > G(y))
    if x.neg > y.neg:
        if not y.is_var or not y.free:
            return False
        sub = x.clone()
        sub.neg = x.neg - y.neg
        substitute_lsb(sub, y.name, root_y, fresh_counter)
        return True

    # Now G(x) == G(y)
    if x.is_var:
        # Label E: TF L -> B
        if x.free:
            sub = y.clone()
            sub.neg = 0
            substitute_lsb(sub, x.name, root_x, fresh_counter)
            return True
        if not y.is_var:
            return False
        if x.name == y.name:
            return True
        # Label D: -TF C -> H
        if not y.free:
            return False
        sub = x.clone()
        sub.neg = 0
        substitute_lsb(sub, y.name, root_y, fresh_counter)
        return True

    if y.is_var:
        # Label D: -TF C -> H
        if not y.free:
            return False
        sub = x.clone()
        sub.neg = 0
        substitute_lsb(sub, y.name, root_y, fresh_counter)
        return True

    # Both x and y are binary connectives with G(x) == G(y)
    if x.conn != y.conn:
        # Label F: try LRp->v on x, else LRpv-> on x
        if x.conn == "->":
            if not replace_imp_to_or(x):
                return False
        else:
            if not replace_or_to_imp(x, root_x, fresh_counter):
                return False
        return match_lmc(x, root_x, y, root_y, stats, fresh_counter)

    # Connectives agree: recurse on left then right
    if not match_lmc(x.left, root_x, y.left, root_y, stats, fresh_counter):
        return False
    if not match_lmc(x.right, root_x, y.right, root_y, stats, fresh_counter):
        return False
    return True
