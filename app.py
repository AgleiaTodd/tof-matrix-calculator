import os
import time
 
import gradio as gr
import matplotlib
 
matplotlib.use("Agg")
 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
 
from numba import njit, types
from numba.typed import Dict
 
# ============================================================
# TOWER OF FANTASY - PROBABILITY CALCULATORS
#
#   Tab 1: MATRICES      (exact dynamic programming)
#   Tab 2: RED NUCLEUS   (vectorized Monte Carlo, weapon copies A0-A6)
# ============================================================
 
 
# ############################################################
# ############################################################
#                    PART A - MATRICES
# ############################################################
# ############################################################
 
SSR_RATE = 0.017
CHIP_COST = 80
GUARANTEE = 40
MAX_STARS = 3
 
 
# ============================================================
# NUMBA DICTIONARY HELPER
# ============================================================
 
@njit
def add_probability(dictionary, key, probability):
 
    if probability == 0.0:
        return
 
    if key in dictionary:
        dictionary[key] += probability
 
    else:
        dictionary[key] = probability
 
 
# ============================================================
# INVENTORY ENCODING
# ============================================================
 
@njit
def encode_inventory(a, b, c, d, base):
 
    return (((a * base + b) * base + c) * base + d)
 
 
@njit
def sort_four(a, b, c, d):
 
    if a > b:
        a, b = b, a
 
    if c > d:
        c, d = d, c
 
    if a > c:
        a, c = c, a
 
    if b > d:
        b, d = d, b
 
    if b > c:
        b, c = c, b
 
    return a, b, c, d
 
 
# ============================================================
# PART 1: SSR / PITY / LIMITED COUNT DISTRIBUTION
# ============================================================
 
@njit
def calculate_limited_distribution(
    maximum_pulls,
    initial_counter,
    initial_d1,
    initial_d2,
    checkpoints
):
 
    # State dimensions:
    # 1. Pity counter (0 to 39)
    # 2. Previous SSR type 1 (S or L)
    # 3. Previous SSR type 2 (S or L)
    # 4. Number of limited SSRs obtained
 
    current = np.zeros(
        (40, 2, 2, maximum_pulls + 1),
        dtype=np.float64
    )
 
    following = np.zeros_like(current)
 
    current[
        initial_counter,
        initial_d1,
        initial_d2,
        0
    ] = 1.0
 
    saved = np.zeros(
        (len(checkpoints), maximum_pulls + 1),
        dtype=np.float64
    )
 
    checkpoint_index = 0
 
    for pull in range(1, maximum_pulls + 1):
 
        following.fill(0.0)
 
        for counter in range(40):
 
            # SSR guaranteed on the 40th pull.
 
            guaranteed = counter == GUARANTEE - 1
 
            if guaranteed:
 
                p_non_ssr = 0.0
                p_ssr = 1.0
                new_counter = 0
 
            else:
 
                p_non_ssr = 1.0 - SSR_RATE
                p_ssr = SSR_RATE
                new_counter = counter + 1
 
            for d1 in range(2):
 
                for d2 in range(2):
 
                    for limited_count in range(pull):
 
                        probability = current[
                            counter,
                            d1,
                            d2,
                            limited_count
                        ]
 
                        if probability == 0.0:
                            continue
 
                        # NON-SSR
 
                        if p_non_ssr > 0.0:
 
                            following[
                                new_counter,
                                d1,
                                d2,
                                limited_count
                            ] += (
                                probability * p_non_ssr
                            )
 
                        # SSR TYPE PROBABILITIES
 
                        if d1 == 1 and d2 == 1:
 
                            # LL: Standard guaranteed
 
                            p_standard = 1.0
                            p_limited = 0.0
 
                        elif d1 == 0 and d2 == 0:
 
                            # SS: Limited guaranteed
 
                            p_standard = 0.0
                            p_limited = 1.0
 
                        else:
 
                            # SL / LS: 50/50
 
                            p_standard = 0.5
                            p_limited = 0.5
 
                        # STANDARD SSR
 
                        if p_standard > 0.0:
 
                            following[
                                new_counter,
                                d2,
                                0,
                                limited_count
                            ] += (
                                probability
                                * p_ssr
                                * p_standard
                            )
 
                        # LIMITED SSR
 
                        if p_limited > 0.0:
 
                            following[
                                new_counter,
                                d2,
                                1,
                                limited_count + 1
                            ] += (
                                probability
                                * p_ssr
                                * p_limited
                            )
 
        current, following = following, current
 
        # SAVE REQUESTED CHECKPOINTS
 
        if (
            checkpoint_index < len(checkpoints)
            and pull == checkpoints[checkpoint_index]
        ):
 
            for k in range(pull + 1):
 
                total = 0.0
 
                for counter in range(40):
 
                    for d1 in range(2):
 
                        for d2 in range(2):
 
                            total += current[
                                counter,
                                d1,
                                d2,
                                k
                            ]
 
                saved[checkpoint_index, k] = total
 
            checkpoint_index += 1
 
    return saved
 
 
# ============================================================
# PART 2: MATRIX INVENTORY DISTRIBUTION
# ============================================================
 
def create_initial_inventory(initial_counts, base):
 
    a, b, c, d = initial_counts
 
    inventory_code = encode_inventory(
        a, b, c, d, base
    )
 
    states = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
 
    states[np.int64(inventory_code)] = 1.0
 
    return states
 
 
@njit
def advance_matrix_distribution(states, base):
 
    new_states = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )
 
    for inventory_code, probability in states.items():
 
        # DECODE INVENTORY
 
        temp = inventory_code
 
        d = temp % base
        temp //= base
 
        c = temp % base
        temp //= base
 
        b = temp % base
        temp //= base
 
        a = temp
 
        # Inventory is sorted: a <= b <= c <= d
 
        maximum = d
 
        # COUNT MATRICES TIED FOR MAXIMUM
 
        number_maximum = 0
 
        if a == maximum:
            number_maximum += 1
 
        if b == maximum:
            number_maximum += 1
 
        if c == maximum:
            number_maximum += 1
 
        if d == maximum:
            number_maximum += 1
 
        # The blocked matrix is selected uniformly
        # among all matrix types tied for the maximum.
        #
        # Matrix below the maximum: probability = 1/3
        # Matrix tied for the maximum:
        # probability = (number_maximum - 1) / (3 * number_maximum)
 
        for index in range(4):
 
            if index == 0:
                count = a
 
            elif index == 1:
                count = b
 
            elif index == 2:
                count = c
 
            else:
                count = d
 
            if count == maximum:
 
                matrix_probability = (
                    number_maximum - 1
                ) / (
                    3.0 * number_maximum
                )
 
            else:
 
                matrix_probability = 1.0 / 3.0
 
            if matrix_probability == 0.0:
                continue
 
            # ADD ONE COPY
 
            na, nb, nc, nd = a, b, c, d
 
            if index == 0:
                na += 1
 
            elif index == 1:
                nb += 1
 
            elif index == 2:
                nc += 1
 
            else:
                nd += 1
 
            # SORT INVENTORY
 
            na, nb, nc, nd = sort_four(
                na, nb, nc, nd
            )
 
            new_code = encode_inventory(
                na, nb, nc, nd, base
            )
 
            new_probability = (
                probability * matrix_probability
            )
 
            add_probability(
                new_states,
                new_code,
                new_probability
            )
 
    return new_states
 
 
# ============================================================
# CHIP OPTIMIZATION
# ============================================================
 
@njit
def calculate_inventory_objectives(
    states,
    base,
    purchases
):
 
    results = np.zeros(
        (len(purchases), 8),
        dtype=np.float64
    )
 
    for inventory_code, probability in states.items():
 
        # DECODE INVENTORY
 
        temp = inventory_code
 
        d = temp % base
        temp //= base
 
        c = temp % base
        temp //= base
 
        b = temp % base
        temp //= base
 
        a = temp
 
        # EVALUATE EVERY PURCHASE SCENARIO
 
        for p_index in range(len(purchases)):
 
            na, nb, nc, nd = a, b, c, d
 
            number_of_purchases = purchases[p_index]
 
            # Spend chips only AFTER all pulls.
            # Improve the weakest matrix first.
 
            for purchase in range(number_of_purchases):
 
                if na >= MAX_STARS + 1:
                    break
 
                na += 1
 
                na, nb, nc, nd = sort_four(
                    na, nb, nc, nd
                )
 
            # 3P AND 4P OBJECTIVES
 
            for stars in range(4):
 
                # 3P N: at least 3 matrices with N stars,
                # the fourth with at least N-1 stars.
 
                valid_3p = (
                    nb >= stars + 1
                    and na >= stars
                )
 
                # 4P N: all four matrices with at least N stars.
 
                valid_4p = (
                    na >= stars + 1
                )
 
                # A completed 4P automatically validates 3P.
 
                if valid_4p:
                    valid_3p = True
 
                if valid_3p:
                    results[p_index, stars] += probability
 
                if valid_4p:
                    results[p_index, 4 + stars] += probability
 
    return results
 
 
def build_matrix_objectives(
    maximum_limited,
    initial_counts,
    base,
    purchase_counts
):
 
    states = create_initial_inventory(
        initial_counts,
        base
    )
 
    objective_results = np.zeros(
        (
            maximum_limited + 1,
            len(purchase_counts),
            8
        ),
        dtype=np.float64
    )
 
    for limited_count in range(maximum_limited + 1):
 
        objective_results[limited_count] = (
            calculate_inventory_objectives(
                states,
                base,
                purchase_counts
            )
        )
 
        # Advance by exactly one limited SSR.
 
        if limited_count < maximum_limited:
 
            states = advance_matrix_distribution(
                states,
                base
            )
 
    return objective_results
 
 
# ============================================================
# PART 3: COMBINE SSR AND MATRIX DISTRIBUTIONS
# ============================================================
 
def combine_distributions(
    limited_distributions,
    matrix_objectives,
    checkpoints,
    initial_chips,
    purchase_index
):
 
    final_results = {}
 
    for checkpoint_index, pulls in enumerate(checkpoints):
 
        purchases = (
            initial_chips + pulls
        ) // CHIP_COST
 
        p_index = purchase_index[purchases]
 
        results = np.zeros(8, dtype=np.float64)
 
        limited_probs = limited_distributions[
            checkpoint_index
        ]
 
        for k in range(len(limited_probs)):
 
            probability_limited = limited_probs[k]
 
            if probability_limited == 0.0:
                continue
 
            results += (
                probability_limited
                * matrix_objectives[k, p_index]
            )
 
        final_results[pulls] = results
 
    return final_results
 
 
# ============================================================
# MATRIX MAIN CALCULATION
# ============================================================
 
def run_calculation(
    number_of_pulls,
    maximum_table_pulls,
    show_table,
    interval,
    initial_counter,
    last1,
    last2,
    initial_counts,
    initial_chips
):
 
    maximum_pulls = max(
        number_of_pulls,
        maximum_table_pulls if show_table else 0
    )
 
    # BUILD CHECKPOINTS
 
    checkpoints = set()
 
    if show_table:
 
        checkpoints.update(
            range(
                interval,
                maximum_table_pulls + 1,
                interval
            )
        )
 
        checkpoints.add(maximum_table_pulls)
 
    checkpoints.add(number_of_pulls)
 
    checkpoints = sorted(checkpoints)
 
    checkpoints_array = np.array(
        checkpoints,
        dtype=np.int64
    )
 
    # INVENTORY ENCODING
 
    base = (
        max(initial_counts)
        + maximum_pulls
        + 2
    )
 
    # INITIAL SSR HISTORY
 
    initial_d1 = 0 if last1 == "S" else 1
    initial_d2 = 0 if last2 == "S" else 1
 
    # CHIP PURCHASE COUNTS
 
    purchase_counts = sorted(
        set(
            (initial_chips + pulls) // CHIP_COST
            for pulls in checkpoints
        )
    )
 
    purchase_index = {
        count: index
        for index, count in enumerate(purchase_counts)
    }
 
    purchase_array = np.array(
        purchase_counts,
        dtype=np.int64
    )
 
    # PHASE 1: SSR DISTRIBUTION
 
    limited_distributions = calculate_limited_distribution(
        maximum_pulls,
        initial_counter,
        initial_d1,
        initial_d2,
        checkpoints_array
    )
 
    # MAXIMUM RELEVANT LIMITED COUNT
 
    nonzero_counts = np.where(
        np.any(
            limited_distributions > 0.0,
            axis=0
        )
    )[0]
 
    maximum_limited = int(nonzero_counts[-1])
 
    # PHASE 2: MATRIX DISTRIBUTION
 
    matrix_objectives = build_matrix_objectives(
        maximum_limited,
        initial_counts,
        base,
        purchase_array
    )
 
    # PHASE 3: COMBINE DISTRIBUTIONS
 
    final_results = combine_distributions(
        limited_distributions,
        matrix_objectives,
        checkpoints,
        initial_chips,
        purchase_index
    )
 
    return final_results, checkpoints
 
 
# ############################################################
# ############################################################
#                  PART B - RED NUCLEUS
# ############################################################
# ############################################################
 
RN_MAX_COPIES = 7  # A6 = 7 copies = max tier (anything above is grouped here)
RN_PITY_LIMIT = 80
RN_SSR_RATE = 0.0075
RN_SR_RATE = 0.12
RN_FLAME_GOLD_COST = 120
 
RN_MAX_PULLS = 1000
RN_MIN_SIMULATIONS = 1000
RN_MAX_SIMULATIONS = 500000
 
 
def rn_labels(french):
 
    none_label = "Aucune" if french else "None"
 
    return [none_label] + [f"A{i}" for i in range(RN_MAX_COPIES)]
 
 
def rn_simulate_distribution(
    total_pulls,
    simulations=100000,
    starting_pity=0,
    starting_last_ssrs=None,
    starting_flame_gold=0,
    seed=None
):
    """
    Returns probs with shape (total_pulls + 1, RN_MAX_COPIES + 1):
    probs[t][k] = probability of having EXACTLY k limited copies
    after t pulls (last column = RN_MAX_COPIES or more).
 
    All simulations run in parallel (vectorized over simulations).
    """
 
    rng = np.random.default_rng(seed)
    N = simulations
 
    # Last two SSRs: 0 = unknown, 1 = S (standard), 2 = L (limited)
    code = {"S": 1, "L": 2}
 
    last2 = np.zeros(N, dtype=np.int8)  # second-to-last
    last1 = np.zeros(N, dtype=np.int8)  # last
 
    if starting_last_ssrs and len(starting_last_ssrs) == 2:
        last2[:] = code[starting_last_ssrs[0]]
        last1[:] = code[starting_last_ssrs[1]]
 
    pity = np.full(N, starting_pity, dtype=np.int32)
    limited = np.zeros(N, dtype=np.int32)
    gold = np.full(N, starting_flame_gold, dtype=np.int32)
 
    sr_rate_cond = RN_SR_RATE / (1 - RN_SSR_RATE)
 
    counts = np.zeros(
        (total_pulls + 1, RN_MAX_COPIES + 1),
        dtype=np.int64
    )
 
    def record(t):
 
        copies = np.minimum(
            limited + gold // RN_FLAME_GOLD_COST,
            RN_MAX_COPIES
        )
 
        counts[t] = np.bincount(
            copies,
            minlength=RN_MAX_COPIES + 1
        )
 
    record(0)
 
    for t in range(1, total_pulls + 1):
 
        gold += 1  # +1 base flame gold per pull
 
        pity_triggered = pity == RN_PITY_LIMIT - 1
        ssr = pity_triggered | (rng.random(N) < RN_SSR_RATE)
 
        # Hidden rule: SS -> L guaranteed, LL -> S guaranteed, else 50/50
        forced_L = (last1 == 1) & (last2 == 1)
        forced_S = (last1 == 2) & (last2 == 2)
        coin_L = rng.random(N) < 0.5
 
        is_L = np.where(
            forced_L,
            True,
            np.where(forced_S, False, coin_L)
        )
 
        got_L = ssr & is_L
        got_S = ssr & ~is_L
 
        limited += got_L
        gold += 10 * got_S  # +10 flame gold for a standard SSR dupe
 
        # SSR history update (only when an SSR was pulled)
        new_type = np.where(is_L, 2, 1).astype(np.int8)
        last2 = np.where(ssr, last1, last2)
        last1 = np.where(ssr, new_type, last1)
 
        # SR (only among non-SSR pulls)
        sr = (~ssr) & (rng.random(N) < sr_rate_cond)
        gold += sr  # +1 flame gold for an SR dupe
 
        # Pity resets only if the SSR came from the pity itself
        pity = np.where(pity_triggered, 0, pity + 1)
 
        record(t)
 
    return counts / N
 
 
def rn_at_least(probs):
    """P(>= k copies) for each k, from exact probabilities."""
 
    return np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]
 
 
def rn_checkpoints(total_pulls, step):
 
    pulls = list(range(step, total_pulls + 1, step))
 
    if not pulls or pulls[-1] != total_pulls:
        pulls.append(total_pulls)
 
    return pulls
 
 
# ============================================================
# RED NUCLEUS CALLBACK
# ============================================================
 
def calculate_red_nucleus(
    language,
    pulls,
    simulations,
    pity,
    history,
    flame_gold,
    table_mode,
    table_type
):
 
    french = language == "Français"
 
    # VALIDATION
 
    pulls = int(pulls or 0)
    simulations = int(simulations or 0)
    pity = int(pity or 0)
    flame_gold = int(flame_gold or 0)
 
    if not 1 <= pulls <= RN_MAX_PULLS:
        raise gr.Error(
            f"Le nombre de tirages doit être entre 1 et {RN_MAX_PULLS}."
            if french else
            f"Number of pulls must be between 1 and {RN_MAX_PULLS}."
        )
 
    if not RN_MIN_SIMULATIONS <= simulations <= RN_MAX_SIMULATIONS:
        raise gr.Error(
            f"Le nombre de simulations doit être entre {RN_MIN_SIMULATIONS} et {RN_MAX_SIMULATIONS}."
            if french else
            f"Number of simulations must be between {RN_MIN_SIMULATIONS} and {RN_MAX_SIMULATIONS}."
        )
 
    if not 0 <= pity <= RN_PITY_LIMIT - 1:
        raise gr.Error(
            f"Le pity doit être entre 0 et {RN_PITY_LIMIT - 1}."
            if french else
            f"Pity must be between 0 and {RN_PITY_LIMIT - 1}."
        )
 
    if flame_gold < 0:
        raise gr.Error(
            "Les flame gold doivent être positifs."
            if french else
            "Flame gold must be non-negative."
        )
 
    history = (history or "").strip().upper()
 
    if len(history) == 2 and all(c in "SL" for c in history):
        starting_last_ssrs = list(history)
    else:
        starting_last_ssrs = []
 
    # SIMULATION
 
    start = time.perf_counter()
 
    probs = rn_simulate_distribution(
        pulls,
        simulations,
        starting_pity=pity,
        starting_last_ssrs=starting_last_ssrs,
        starting_flame_gold=flame_gold
    )
 
    elapsed = time.perf_counter() - start
 
    cumul = rn_at_least(probs)
    labels = rn_labels(french)
 
    exact_final = probs[pulls]
    cumul_final = cumul[pulls]
 
    # FINAL SUMMARY
 
    if french:
 
        output = f"""
# Résultats Red Nucleus
 
## Probabilités après {pulls} tirages
 
| Palier | Exactement | Au moins |
|:---|---:|---:|
"""
 
    else:
 
        output = f"""
# Red Nucleus results
 
## Probabilities after {pulls} pulls
 
| Tier | Exactly | At least |
|:---|---:|---:|
"""
 
    for k in range(RN_MAX_COPIES + 1):
 
        if k == 0:
            name = labels[0] + (" (0 copie)" if french else " (0 copies)")
        elif k == 1:
            name = f"{labels[k]} (1 copie)" if french else f"{labels[k]} (1 copy)"
        elif k == RN_MAX_COPIES:
            name = (
                f"{labels[k]} ({k} copies ou plus)"
                if french else
                f"{labels[k]} ({k} copies or more)"
            )
        else:
            name = (
                f"{labels[k]} ({k} copies)"
                if french else
                f"{labels[k]} ({k} copies)"
            )
 
        output += (
            f"| **{name}** | "
            f"{exact_final[k] * 100:.2f}% | "
            f"{cumul_final[k] * 100:.2f}% |\n"
        )
 
    # RESOURCES
 
    if french:
 
        output += f"""
---
 
## Paramètres
 
- Tirages : **{pulls}**
- Simulations : **{simulations:,}**
- Pity initial : **{pity}/{RN_PITY_LIMIT}**
- Derniers SSR : **{''.join(starting_last_ssrs) if starting_last_ssrs else 'inconnu'}**
- Flame gold initiaux : **{flame_gold}**
- Temps de calcul : **{elapsed:.2f} s**
 
*Estimation Monte Carlo : avec {simulations:,} simulations, la marge d'erreur est d'environ ±{100 * 1.96 * 0.5 / np.sqrt(simulations):.2f} point(s) au maximum.*
"""
 
    else:
 
        output += f"""
---
 
## Parameters
 
- Pulls: **{pulls}**
- Simulations: **{simulations:,}**
- Starting pity: **{pity}/{RN_PITY_LIMIT}**
- Last SSRs: **{''.join(starting_last_ssrs) if starting_last_ssrs else 'unknown'}**
- Starting flame gold: **{flame_gold}**
- Calculation time: **{elapsed:.2f} s**
 
*Monte Carlo estimate: with {simulations:,} simulations, the margin of error is about ±{100 * 1.96 * 0.5 / np.sqrt(simulations):.2f} percentage point(s) at most.*
"""
 
    # PROGRESSION TABLE AND GRAPH
 
    fig = None
 
    step = {
        "Every pull / Chaque tirage": 1,
        "Every 10 pulls / Tous les 10 tirages": 10
    }.get(table_mode)
 
    if step is not None:
 
        use_exact = table_type.startswith("Exact")
 
        data = probs if use_exact else cumul
 
        checkpoints = rn_checkpoints(pulls, step)
 
        if french:
            kind = "exacte" if use_exact else "d'en avoir au moins"
            output += "\n---\n\n# Tableau de progression\n\n"
            output += f"Probabilité **{kind}** pour chaque palier, aux tirages indiqués.\n\n"
        else:
            kind = "exact" if use_exact else "at-least"
            output += "\n---\n\n# Probability progression\n\n"
            output += f"**{kind.capitalize()}** probability for each tier at the given pull counts.\n\n"
 
        pull_header = "Tirages" if french else "Pulls"
 
        output += (
            f"| {pull_header} | "
            + " | ".join(labels)
            + " |\n"
        )
 
        output += "|---:|" + "---:|" * len(labels) + "\n"
 
        for t in checkpoints:
 
            output += (
                f"| {t} | "
                + " | ".join(
                    f"{data[t][k] * 100:.2f}%"
                    for k in range(RN_MAX_COPIES + 1)
                )
                + " |\n"
            )
 
        # GRAPH (A0 to A6)
 
        fig = Figure(figsize=(12, 6))
        ax = fig.subplots()
 
        colors = [
            "#E5B94F",
            "#F28C45",
            "#E65D68",
            "#BB83E5",
            "#40BFD2",
            "#468FE0",
            "#49B88A"
        ]
 
        x_values = [0] + checkpoints if checkpoints[0] != 0 else checkpoints
 
        for k in range(1, RN_MAX_COPIES + 1):
 
            y_values = [data[t][k] * 100 for t in x_values]
 
            ax.plot(
                x_values,
                y_values,
                label=labels[k],
                color=colors[k - 1],
                linewidth=2,
                marker="o" if step > 1 else None,
                markersize=3
            )
 
        if french:
            title = (
                "Probabilité exacte de chaque palier"
                if use_exact else
                "Probabilité d'avoir au moins chaque palier"
            )
            xlabel = "Nombre de tirages"
            ylabel = "Probabilité (%)"
        else:
            title = (
                "Exact probability of each tier"
                if use_exact else
                "Probability of reaching at least each tier"
            )
            xlabel = "Number of pulls"
            ylabel = "Probability (%)"
 
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", ncol=2, fontsize=9)
        fig.tight_layout()
 
    # RULES
 
    if french:
 
        output += f"""
---
 
### Rappel des règles
 
- **A0** = 1 copie de l'arme limitée, **A6** = 7 copies ou plus (palier maximal).
- **Exactement** : probabilité d'avoir précisément ce nombre de copies. **Au moins** : ce nombre de copies ou plus.
- Pity à {RN_PITY_LIMIT} tirages (non remis à zéro si un SSR sort hors pity).
- Règle cachée : après SS → limité garanti ; après LL → standard garanti.
- Flame gold : +1 par tirage, +1 par SR, +10 par SSR standard ; {RN_FLAME_GOLD_COST} flame gold = 1 copie en boutique.
"""
 
    else:
 
        output += f"""
---
 
### Rules
 
- **A0** = 1 copy of the limited weapon, **A6** = 7 copies or more (max tier).
- **Exactly**: probability of having precisely that many copies. **At least**: that many copies or more.
- Pity at {RN_PITY_LIMIT} pulls (not reset if an SSR is pulled outside the pity).
- Hidden rule: after SS → limited guaranteed; after LL → standard guaranteed.
- Flame gold: +1 per pull, +1 per SR, +10 per standard SSR; {RN_FLAME_GOLD_COST} flame gold = 1 shop copy.
"""
 
    return output, fig
 
 
# ############################################################
# ############################################################
#                      PART C - WEB SITE
# ############################################################
# ############################################################
 
CUSTOM_CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
    background: radial-gradient(
        ellipse at 50% 0%,
        #263c50 0%,
        #111925 55%,
        #0b111b 100%
    );
    color: #edf4fa !important;
}
 
/* HEADER */
 
#main-header {
    text-align: center;
    padding: 35px 15px 25px;
    border: 1px solid #52677e;
    border-radius: 18px;
    background: linear-gradient(135deg, #1c2b3c, #101a27);
    margin-bottom: 20px;
}
 
#main-header h1 {
    font-size: 35px;
    letter-spacing: 3px;
    color: #f2d58c !important;
}
 
#main-header h2 {
    color: #75e5f2 !important;
    font-size: 19px;
    letter-spacing: 5px;
}
 
#main-header p {
    color: #c1d0de !important;
}
 
/* SECTION TITLES */
 
.section-title,
.section-title h2 {
    color: #f2d58c !important;
    border-bottom: 1px solid #52677e;
    padding-bottom: 8px;
}
 
/* INPUT PANELS */
 
.input-panel {
    background: #1c2b3c !important;
    border: 1px solid #52677e;
    border-radius: 14px;
    padding: 15px;
}
 
/* LABELS */
 
.gradio-container label,
.gradio-container .wrap label,
.gradio-container .prose,
.gradio-container p {
    color: #edf4fa !important;
}
 
/* TABS */
 
.gradio-container button[role="tab"] {
    color: #e5d5a9 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}
 
.gradio-container button[role="tab"][aria-selected="true"] {
    color: #f2d58c !important;
    border-color: #e6c778 !important;
}
 
/* INPUT FIELDS - SOFT GREY */
 
.gradio-container input:not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea {
    background-color: #cbd0d5 !important;
    color: #172638 !important;
    border: 1px solid #929da8 !important;
    border-radius: 7px !important;
}
 
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #626d77 !important;
    opacity: 1 !important;
}
 
/* DROPDOWN */
 
.gradio-container [role="combobox"] {
    background: #cbd0d5 !important;
    color: #172638 !important;
    border: 1px solid #929da8 !important;
    border-radius: 7px !important;
}
 
.gradio-container [role="listbox"],
.gradio-container [role="option"] {
    background: #cbd0d5 !important;
    color: #172638 !important;
}
 
.gradio-container [role="combobox"] input {
    background: transparent !important;
    color: #172638 !important;
}
 
/* CALCULATE BUTTONS */
 
#calculate-button,
#rn-calculate-button {
    background: linear-gradient(
        90deg,
        #b48b43,
        #e6c778
    ) !important;
    color: #111925 !important;
    font-weight: bold !important;
    font-size: 17px !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 55px;
}
 
/* RESULTS PANELS (matrices + Red Nucleus) */
 
.results-box {
    background: #d9dde1 !important;
    color: #172638 !important;
    border: 2px solid #d2b36d !important;
    border-radius: 14px !important;
    padding: 25px !important;
}
 
.results-box,
.results-box p,
.results-box li,
.results-box strong,
.results-box em,
.results-box span {
    color: #172638 !important;
}
 
.results-box h1,
.results-box h2,
.results-box h3 {
    color: #183d58 !important;
    font-weight: 800 !important;
}
 
/* RESULTS TABLE */
 
.results-box table {
    width: 100% !important;
    border-collapse: collapse !important;
    background: #d9dde1 !important;
}
 
.results-box th {
    background: #c2ccd5 !important;
    color: #183d58 !important;
    padding: 12px !important;
    border: 1px solid #aebfce !important;
}
 
.results-box td {
    background: #d9dde1 !important;
    color: #172638 !important;
    padding: 12px !important;
    border: 1px solid #b8c1ca !important;
}
 
.results-box tr:nth-child(even) td {
    background: #cbd2d9 !important;
}
 
/* LINKS */
 
.results-box a {
    color: #087eaa !important;
}
 
/* CHECKBOX */
 
#progress-checkbox {
    background: #1c2b3c !important;
    border: 1px solid #52677e !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
 
#progress-checkbox input[type="checkbox"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;
    accent-color: #e6c778 !important;
    width: 22px !important;
    height: 22px !important;
    cursor: pointer !important;
    opacity: 1 !important;
}
 
#progress-checkbox input[type="checkbox"]:checked {
    accent-color: #e6c778 !important;
}
 
#progress-checkbox label,
#progress-checkbox span,
#progress-checkbox p {
    color: #f2d58c !important;
    font-weight: 600 !important;
    cursor: pointer !important;
}
 
#progress-checkbox [role="checkbox"] {
    border: 2px solid #e6c778 !important;
    background: #26394c !important;
    opacity: 1 !important;
}
 
#progress-checkbox [role="checkbox"][aria-checked="true"] {
    background: #b48b43 !important;
    border-color: #f2d58c !important;
}
 
#progress-checkbox [role="checkbox"][aria-checked="true"] svg {
    color: #101925 !important;
    stroke: #101925 !important;
}
 
/* RADIO BUTTONS */
 
.gradio-container input[type="radio"] {
    accent-color: #e6c778 !important;
}
 
/* GENERAL SETTINGS */
 
.gradio-container,
.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap {
    color: #edf4fa !important;
}
 
.gradio-container label {
    color: #e5d5a9 !important;
    font-weight: 600 !important;
}
 
footer {
    display: none !important;
}
"""
 
 
# ============================================================
# MATRIX CALLBACK
# ============================================================
 
def calculate_web(
    language,
    pulls,
    chips,
    pity,
    last1,
    last2,
    A,
    B,
    C,
    D,
    show_progress
):
 
    # VALIDATION
 
    pulls = int(pulls or 0)
    chips = int(chips or 0)
    pity = int(pity or 0)
 
    counts = [int(x or 0) for x in (A, B, C, D)]
 
    if pulls < 1:
        raise gr.Error("Number of pulls must be at least 1.")
 
    if not 0 <= pity <= 39:
        raise gr.Error("Pity must be between 0 and 39.")
 
    if chips < 0 or any(x < 0 for x in counts):
        raise gr.Error("All inventory and chip values must be non-negative.")
 
    initial_counts = tuple(sorted(counts))
 
    # EXACT CALCULATION ENGINE
 
    if show_progress:
 
        results, checkpoints = run_calculation(
            pulls,
            pulls,
            True,
            20,
            pity,
            last1,
            last2,
            initial_counts,
            chips
        )
 
    else:
 
        results, checkpoints = run_calculation(
            pulls,
            0,
            False,
            20,
            pity,
            last1,
            last2,
            initial_counts,
            chips
        )
 
    values = results[pulls]
 
    purchases = (chips + pulls) // CHIP_COST
    remaining_chips = (chips + pulls) % CHIP_COST
 
    # LANGUAGE
 
    french = language == "Français"
 
    if french:
 
        output = """
# Résultats des probabilités
 
## Objectifs 3 pièces
 
| Objectif | Probabilité |
|:---|---:|
"""
 
    else:
 
        output = """
# Probability results
 
## 3-piece objectives
 
| Objective | Probability |
|:---|---:|
"""
 
    # FINAL 3P RESULTS
 
    for stars in range(4):
 
        output += (
            f"| **3P {stars}★** | "
            f"**{values[stars] * 100:.6f}%** |\n"
        )
 
    # FINAL 4P RESULTS
 
    if french:
        output += "\n## Objectifs 4 pièces\n\n"
    else:
        output += "\n## 4-piece objectives\n\n"
 
    output += "| Objective | Probability |\n|:---|---:|\n"
 
    for stars in range(4):
 
        output += (
            f"| **4P {stars}★** | "
            f"**{values[4 + stars] * 100:.6f}%** |\n"
        )
 
    # RESOURCES
 
    if french:
 
        output += f"""
---
 
## Informations sur les ressources
 
- Tirages disponibles : **{pulls}**
- Chips initiales : **{chips}**
- Chips totales : **{chips + pulls}**
- Achats possibles : **{purchases}**
- Chips restantes : **{remaining_chips}**
"""
 
    else:
 
        output += f"""
---
 
## Resource information
 
- Available pulls: **{pulls}**
- Initial chips: **{chips}**
- Total chips: **{chips + pulls}**
- Available purchases: **{purchases}**
- Remaining chips: **{remaining_chips}**
"""
 
    # PROGRESSION TABLE AND GRAPH
 
    fig = None
 
    if show_progress:
 
        if french:
            output += "\n---\n\n# Tableau de progression\n\n"
            output += (
                "Probabilités calculées tous les 20 tirages, "
                "avec le dernier palier demandé.\n\n"
            )
        else:
            output += "\n---\n\n# Probability progression\n\n"
            output += (
                "Probabilities calculated every 20 pulls, "
                "including the requested final checkpoint.\n\n"
            )
 
        output += (
            "| Pulls | 3P 0★ | 3P 1★ | 3P 2★ | 3P 3★ "
            "| 4P 0★ | 4P 1★ | 4P 2★ | 4P 3★ |\n"
        )
 
        output += (
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
 
        for checkpoint in checkpoints:
 
            row = results[checkpoint]
 
            output += f"| {checkpoint} "
 
            for probability in row:
                output += f"| {probability * 100:.4f}% "
 
            output += "|\n"
 
        # GRAPH
 
        fig, ax = plt.subplots(figsize=(12, 6))
 
        colors = [
            "#E5B94F",
            "#F28C45",
            "#E65D68",
            "#BB83E5",
            "#40BFD2",
            "#468FE0",
            "#49B88A",
            "#A4C75A"
        ]
 
        labels = [
            "3P 0★",
            "3P 1★",
            "3P 2★",
            "3P 3★",
            "4P 0★",
            "4P 1★",
            "4P 2★",
            "4P 3★"
        ]
 
        x_values = checkpoints
 
        for index in range(8):
 
            y_values = [
                results[p][index] * 100
                for p in checkpoints
            ]
 
            ax.plot(
                x_values,
                y_values,
                label=labels[index],
                color=colors[index],
                linewidth=2,
                marker="o",
                markersize=3
            )
 
        ax.set_title(
            "Evolution of matrix probabilities"
            if not french
            else "Évolution des probabilités des matrices",
            fontsize=14,
            fontweight="bold"
        )
 
        ax.set_xlabel(
            "Number of pulls"
            if not french
            else "Nombre de tirages"
        )
 
        ax.set_ylabel(
            "Probability (%)"
            if not french
            else "Probabilité (%)"
        )
 
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)
 
        ax.grid(
            True,
            linestyle="--",
            alpha=0.35
        )
 
        ax.legend(
            loc="upper left",
            ncol=2,
            fontsize=9
        )
 
        fig.tight_layout()
 
        plt.close(fig)
 
    # RULE EXPLANATION
 
    if french:
 
        output += """
---
 
### Rappel des règles
 
- **3P N★** : trois matrices différentes avec au moins N étoiles et la quatrième avec au moins N-1 étoiles.
- **4P N★** : les quatre matrices différentes avec au moins N étoiles.
- Les chips sont dépensées après tous les tirages.
- Le blocage dynamique est pris en compte.
"""
 
    else:
 
        output += """
---
 
### Rules
 
- **3P N★**: three different matrices with at least N stars and the fourth with at least N-1 stars.
- **4P N★**: all four different matrices with at least N stars.
- Chips are spent after all pulls.
- Dynamic matrix blocking is taken into account.
"""
 
    return output, fig
 
 
# ============================================================
# BUILD WEB INTERFACE
# ============================================================
 
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="amber",
        neutral_hue="slate"
    ),
    css=CUSTOM_CSS,
    title="Tower of Fantasy - Probability Calculators"
) as demo:
 
    # HEADER
 
    gr.HTML("""
    <div id="main-header">
        <h1>PROBABILITY CALCULATORS</h1>
        <h2>TOWER OF FANTASY</h2>
        <p>MATRICES &amp; RED NUCLEUS WEAPON BANNER</p>
    </div>
    """)
 
    # LANGUAGE (shared by both calculators)
 
    language = gr.Dropdown(
        choices=["English", "Français"],
        value="English",
        label="Language / Langue",
        interactive=True
    )
 
    with gr.Tabs():
 
        # ====================================================
        # TAB 1: MATRICES
        # ====================================================
 
        with gr.TabItem("🔷 Matrices"):
 
            gr.Markdown(
                "## YOUR PULLING SITUATION",
                elem_classes=["section-title"]
            )
 
            with gr.Row():
 
                with gr.Column(elem_classes=["input-panel"]):
 
                    pulls = gr.Number(
                        value=320,
                        precision=0,
                        label="Number of pulls / Nombre de tirages"
                    )
 
                    chips = gr.Number(
                        value=0,
                        precision=0,
                        label="Initial chips / Chips initiales"
                    )
 
                with gr.Column(elem_classes=["input-panel"]):
 
                    pity = gr.Number(
                        value=0,
                        precision=0,
                        label="Current pity counter (0-39)"
                    )
 
                    gr.Markdown("### Previous SSR history")
 
                    with gr.Row():
 
                        last1 = gr.Dropdown(
                            choices=["S", "L"],
                            value="S",
                            label="Previous SSR #1"
                        )
 
                        last2 = gr.Dropdown(
                            choices=["S", "L"],
                            value="L",
                            label="Previous SSR #2"
                        )
 
            gr.Markdown(
                "## YOUR MATRIX INVENTORY",
                elem_classes=["section-title"]
            )
 
            gr.Markdown(
                "Enter the total number of copies owned for each matrix."
            )
 
            with gr.Row():
 
                A = gr.Number(value=0, precision=0, label="Matrix A")
                B = gr.Number(value=0, precision=0, label="Matrix B")
                C = gr.Number(value=0, precision=0, label="Matrix C")
                D = gr.Number(value=0, precision=0, label="Matrix D")
 
            gr.Markdown(
                "## PROGRESSION OPTIONS",
                elem_classes=["section-title"]
            )
 
            show_progress = gr.Checkbox(
                value=True,
                label=(
                    "Display probability table and graph every 20 pulls "
                    "/ Afficher le tableau et le graphique tous les 20 tirages"
                ),
                interactive=True,
                elem_id="progress-checkbox"
            )
 
            calculate_button = gr.Button(
                "CALCULATE PROBABILITIES",
                elem_id="calculate-button",
                variant="primary"
            )
 
            gr.Markdown(
                "## PROBABILITY RESULTS",
                elem_classes=["section-title"]
            )
 
            results_output = gr.Markdown(
                value="Your results will appear here.",
                elem_classes=["results-box"]
            )
 
            graph_output = gr.Plot(
                label="Probability progression"
            )
 
            calculate_button.click(
                fn=calculate_web,
                inputs=[
                    language,
                    pulls,
                    chips,
                    pity,
                    last1,
                    last2,
                    A,
                    B,
                    C,
                    D,
                    show_progress
                ],
                outputs=[
                    results_output,
                    graph_output
                ]
            )
 
        # ====================================================
        # TAB 2: RED NUCLEUS
        # ====================================================
 
        with gr.TabItem("🔥 Red Nucleus"):
 
            gr.Markdown(
                "## YOUR PULLING SITUATION / VOTRE SITUATION",
                elem_classes=["section-title"]
            )
 
            with gr.Row():
 
                with gr.Column(elem_classes=["input-panel"]):
 
                    rn_pulls = gr.Number(
                        value=160,
                        precision=0,
                        label="Number of pulls / Nombre de tirages"
                    )
 
                    rn_simulations = gr.Number(
                        value=100000,
                        precision=0,
                        label="Simulations (more = more precise / plus = plus précis)"
                    )
 
                    rn_flame_gold = gr.Number(
                        value=0,
                        precision=0,
                        label="Current flame gold / Flame gold actuels"
                    )
 
                with gr.Column(elem_classes=["input-panel"]):
 
                    rn_pity = gr.Number(
                        value=0,
                        precision=0,
                        label=f"Current pity counter (0-{RN_PITY_LIMIT - 1}) / Compteur de pity"
                    )
 
                    rn_history = gr.Dropdown(
                        choices=["Unknown / Inconnu", "SS", "SL", "LS", "LL"],
                        value="Unknown / Inconnu",
                        label="Last two SSRs (S = standard, L = limited) / Deux derniers SSR"
                    )
 
            gr.Markdown(
                "## TABLE OPTIONS / OPTIONS DU TABLEAU",
                elem_classes=["section-title"]
            )
 
            with gr.Row():
 
                rn_table_mode = gr.Radio(
                    choices=[
                        "No table / Pas de tableau",
                        "Every 10 pulls / Tous les 10 tirages",
                        "Every pull / Chaque tirage"
                    ],
                    value="Every 10 pulls / Tous les 10 tirages",
                    label="Progression table and graph / Tableau et graphique"
                )
 
                rn_table_type = gr.Radio(
                    choices=[
                        "Exact probability / Probabilité exacte",
                        "At least / Au moins"
                    ],
                    value="At least / Au moins",
                    label="Table type / Type de tableau"
                )
 
            rn_button = gr.Button(
                "CALCULATE PROBABILITIES",
                elem_id="rn-calculate-button",
                variant="primary"
            )
 
            gr.Markdown(
                "## PROBABILITY RESULTS",
                elem_classes=["section-title"]
            )
 
            rn_results_output = gr.Markdown(
                value="Your results will appear here.",
                elem_classes=["results-box"]
            )
 
            rn_graph_output = gr.Plot(
                label="Probability progression"
            )
 
            def _rn_wrapper(
                language,
                pulls,
                simulations,
                pity,
                history,
                flame_gold,
                table_mode,
                table_type
            ):
 
                history_clean = (
                    history
                    if history in ("SS", "SL", "LS", "LL")
                    else ""
                )
 
                return calculate_red_nucleus(
                    language,
                    pulls,
                    simulations,
                    pity,
                    history_clean,
                    flame_gold,
                    table_mode,
                    table_type
                )
 
            rn_button.click(
                fn=_rn_wrapper,
                inputs=[
                    language,
                    rn_pulls,
                    rn_simulations,
                    rn_pity,
                    rn_history,
                    rn_flame_gold,
                    rn_table_mode,
                    rn_table_type
                ],
                outputs=[
                    rn_results_output,
                    rn_graph_output
                ]
            )
 
    # FOOTER
 
    gr.HTML("""
    <div style="
        text-align:center;
        padding:25px 5px;
        color:#8297aa;
        font-size:11px;
        letter-spacing:1px;
    ">
        COMMUNITY FAN-MADE TOOL<br>
        TOWER OF FANTASY PROBABILITY CALCULATORS
    </div>
    """)
 
 
# ============================================================
# LAUNCH PUBLIC WEBSITE
# ============================================================
 
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=False,
    show_error=True
)
 
