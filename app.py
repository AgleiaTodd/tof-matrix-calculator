import numpy as np
import time

from numba import njit
from numba.typed import Dict
from numba import types


# ============================================================
# TOWER OF FANTASY - MATRIX PROBABILITY CALCULATOR
# OPTIMIZED EXACT DYNAMIC PROGRAMMING
# ============================================================

SSR_RATE = 0.017
CHIP_COST = 80
GUARANTEE = 40
MAX_STARS = 3


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_integer(message, minimum=0):

    while True:
        try:
            value = int(input(message))

            if value < minimum:
                print(f"Enter a value >= {minimum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid integer.")


def ask_last_ssr_types():

    while True:

        history = input(
            "Last two SSR types (SS, SL, LS or LL): "
        ).strip().upper()

        if len(history) == 2 and all(
            c in "SL" for c in history
        ):
            return history[0], history[1]

        print("Invalid input. Use SS, SL, LS or LL.")


def ask_initial_matrices():

    counts = []

    print("\nEnter the number of copies owned:")

    for matrix in ("A", "B", "C", "D"):

        count = ask_integer(
            f"{matrix}: ",
            minimum=0
        )

        counts.append(count)

    # Matrix identities are interchangeable.
    # Sorting is valid because all four types
    # have identical probabilities.

    return tuple(sorted(counts))


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
    #
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

    # --------------------------------------------------------
    # PROCESS EVERY PULL
    # --------------------------------------------------------

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

                        # ------------------------------------
                        # NON-SSR
                        # ------------------------------------

                        if p_non_ssr > 0.0:

                            following[
                                new_counter,
                                d1,
                                d2,
                                limited_count
                            ] += (
                                probability * p_non_ssr
                            )

                        # ------------------------------------
                        # SSR TYPE PROBABILITIES
                        # ------------------------------------

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

                        # ------------------------------------
                        # STANDARD SSR
                        # ------------------------------------

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

                        # ------------------------------------
                        # LIMITED SSR
                        # ------------------------------------

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

        # ----------------------------------------------------
        # SAVE REQUESTED CHECKPOINTS
        # ----------------------------------------------------

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


# ============================================================
# DYNAMIC MATRIX BLOCKING
# ============================================================

@njit
def advance_matrix_distribution(states, base):

    new_states = Dict.empty(
        key_type=types.int64,
        value_type=types.float64
    )

    for inventory_code, probability in states.items():

        # ----------------------------------------------------
        # DECODE INVENTORY
        # ----------------------------------------------------

        temp = inventory_code

        d = temp % base
        temp //= base

        c = temp % base
        temp //= base

        b = temp % base
        temp //= base

        a = temp

        # Inventory is sorted:
        # a <= b <= c <= d

        maximum = d

        # ----------------------------------------------------
        # COUNT MATRICES TIED FOR MAXIMUM
        # ----------------------------------------------------

        number_maximum = 0

        if a == maximum:
            number_maximum += 1

        if b == maximum:
            number_maximum += 1

        if c == maximum:
            number_maximum += 1

        if d == maximum:
            number_maximum += 1

        # ----------------------------------------------------
        # BLOCKING PROBABILITIES
        # ----------------------------------------------------

        # The blocked matrix is selected uniformly
        # among all matrix types tied for the maximum.
        #
        # For a matrix below the maximum:
        # acquisition probability = 1/3
        #
        # For a matrix tied for the maximum:
        # acquisition probability =
        # (number_maximum - 1)
        # / (3 * number_maximum)

        # ----------------------------------------------------
        # PROCESS EACH POSSIBLE ACQUISITION
        # ----------------------------------------------------

        for index in range(4):

            if index == 0:
                count = a

            elif index == 1:
                count = b

            elif index == 2:
                count = c

            else:
                count = d

            # -----------------------------------------------
            # PROBABILITY OF OBTAINING THIS MATRIX
            # -----------------------------------------------

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

            # -----------------------------------------------
            # ADD ONE COPY
            # -----------------------------------------------

            na, nb, nc, nd = a, b, c, d

            if index == 0:
                na += 1

            elif index == 1:
                nb += 1

            elif index == 2:
                nc += 1

            else:
                nd += 1

            # -----------------------------------------------
            # SORT INVENTORY
            # -----------------------------------------------

            na, nb, nc, nd = sort_four(
                na, nb, nc, nd
            )

            new_code = encode_inventory(
                na, nb, nc, nd, base
            )

            # -----------------------------------------------
            # UPDATE PROBABILITY
            # -----------------------------------------------

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

        # ----------------------------------------------------
        # DECODE INVENTORY
        # ----------------------------------------------------

        temp = inventory_code

        d = temp % base
        temp //= base

        c = temp % base
        temp //= base

        b = temp % base
        temp //= base

        a = temp

        # ----------------------------------------------------
        # EVALUATE EVERY PURCHASE SCENARIO
        # ----------------------------------------------------

        for p_index in range(len(purchases)):

            na, nb, nc, nd = a, b, c, d

            number_of_purchases = purchases[p_index]

            # Spend chips only AFTER all pulls.
            #
            # Improve the weakest matrix first.
            # This completes each star level progressively.

            for purchase in range(number_of_purchases):

                if na >= MAX_STARS + 1:
                    break

                na += 1

                na, nb, nc, nd = sort_four(
                    na, nb, nc, nd
                )

            # ------------------------------------------------
            # CALCULATE 3P AND 4P OBJECTIVES
            # ------------------------------------------------

            for stars in range(4):

                # 3P N:
                #
                # At least 3 different matrices must have
                # at least N stars.
                #
                # The fourth matrix must have at least
                # N-1 stars.

                valid_3p = (
                    nb >= stars + 1
                    and na >= stars
                )

                # 4P N:
                #
                # All four different matrices must have
                # at least N stars.

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


# ============================================================
# BUILD MATRIX DISTRIBUTIONS
# ============================================================

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

        # Evaluate current inventory distribution.

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
# DISPLAY
# ============================================================

def display_header():

    print("\n")
    print("=" * 125)
    print("PROBABILITY TABLE")
    print("=" * 125)

    print(
        f"{'Pulls':>6} | "
        f"{'3P 0*':>10} | "
        f"{'3P 1*':>10} | "
        f"{'3P 2*':>10} | "
        f"{'3P 3*':>10} | "
        f"{'4P 0*':>10} | "
        f"{'4P 1*':>10} | "
        f"{'4P 2*':>10} | "
        f"{'4P 3*':>10}"
    )

    print("-" * 125)


def display_row(pulls, results):

    print(
        f"{pulls:6d} | "
        + " | ".join(
            f"{value * 100:9.5f}%"
            for value in results
        )
    )


def display_final_results(
    results,
    pulls,
    initial_chips
):

    print("\n")
    print("=" * 75)
    print("FINAL RESULTS")
    print("=" * 75)

    print("\n3-PIECE SET PROBABILITIES")

    for stars in range(4):

        print(
            f"3P {stars}*: "
            f"{results[stars] * 100:.8f}%"
        )

    print("\n4-PIECE SET PROBABILITIES")

    for stars in range(4):

        print(
            f"4P {stars}*: "
            f"{results[4 + stars] * 100:.8f}%"
        )

    total_chips = initial_chips + pulls

    print("\n" + "-" * 75)

    print(f"Pulls: {pulls}")
    print(f"Initial chips: {initial_chips}")
    print(f"Total chips: {total_chips}")
    print(f"Purchases: {total_chips // CHIP_COST}")
    print(f"Remaining chips: {total_chips % CHIP_COST}")

    print("=" * 75)


# ============================================================
# MAIN CALCULATION
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

    # --------------------------------------------------------
    # BUILD CHECKPOINTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INVENTORY ENCODING
    # --------------------------------------------------------

    base = (
        max(initial_counts)
        + maximum_pulls
        + 2
    )

    # --------------------------------------------------------
    # INITIAL SSR HISTORY
    # --------------------------------------------------------

    initial_d1 = 0 if last1 == "S" else 1
    initial_d2 = 0 if last2 == "S" else 1

    # --------------------------------------------------------
    # CHIP PURCHASE COUNTS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PHASE 1: SSR DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nPhase 1/3: Calculating SSR and pity probabilities..."
    )

    limited_distributions = calculate_limited_distribution(
        maximum_pulls,
        initial_counter,
        initial_d1,
        initial_d2,
        checkpoints_array
    )

    # --------------------------------------------------------
    # FIND MAXIMUM RELEVANT LIMITED COUNT
    # --------------------------------------------------------

    nonzero_counts = np.where(
        np.any(
            limited_distributions > 0.0,
            axis=0
        )
    )[0]

    maximum_limited = int(nonzero_counts[-1])

    print(
        f"Maximum limited SSR count: {maximum_limited}"
    )

    # --------------------------------------------------------
    # PHASE 2: MATRIX DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nPhase 2/3: Calculating matrix distributions..."
    )

    matrix_objectives = build_matrix_objectives(
        maximum_limited,
        initial_counts,
        base,
        purchase_array
    )

    # --------------------------------------------------------
    # PHASE 3: COMBINE DISTRIBUTIONS
    # --------------------------------------------------------

    print(
        "\nPhase 3/3: Combining probabilities..."
    )

    final_results = combine_distributions(
        limited_distributions,
        matrix_objectives,
        checkpoints,
        initial_chips,
        purchase_index
    )

    return final_results, checkpoints


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 75)
    print(
        "TOWER OF FANTASY - MATRIX PROBABILITY CALCULATOR"
    )
    print(
        "FACTORIZED EXACT DYNAMIC PROGRAMMING"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    number_of_pulls = ask_integer(
        "\nNumber of pulls available: ",
        minimum=1
    )

    initial_chips = ask_integer(
        "Number of chips already owned: ",
        minimum=0
    )

    initial_counter = ask_integer(
        "Current pity counter (0-39): ",
        minimum=0
    )

    if initial_counter > 39:

        print(
            "Counter must be between 0 and 39."
        )

        return

    last1, last2 = ask_last_ssr_types()

    initial_counts = ask_initial_matrices()

    # --------------------------------------------------------
    # OPTIONAL TABLE
    # --------------------------------------------------------

    answer = input(
        "\nGenerate a probability table? (y/n): "
    ).strip().lower()

    show_table = answer == "y"

    maximum_table_pulls = 0
    interval = 20

    if show_table:

        maximum_table_pulls = ask_integer(
            "Maximum number of pulls in the table: ",
            minimum=1
        )

        interval = ask_integer(
            "Display interval (20 recommended): ",
            minimum=1
        )

    # --------------------------------------------------------
    # DISPLAY PARAMETERS
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("RECORDED PARAMETERS")
    print("=" * 75)

    print(f"Pulls: {number_of_pulls}")
    print(f"Initial chips: {initial_chips}")
    print(f"Pity: {initial_counter}/40")
    print(f"Last SSRs: {last1}{last2}")

    print("\nInitial inventory:")

    for matrix, count in zip(
        ("A", "B", "C", "D"),
        initial_counts
    ):

        print(f"{matrix}: {count} copies")

    if show_table:

        print(
            f"\nTable interval: {interval}"
        )

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    print("\nStarting calculation...")

    start = time.perf_counter()

    results, checkpoints = run_calculation(
        number_of_pulls,
        maximum_table_pulls,
        show_table,
        interval,
        initial_counter,
        last1,
        last2,
        initial_counts,
        initial_chips
    )

    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # DISPLAY TABLE
    # --------------------------------------------------------

    if show_table:

        display_header()

        for pulls in checkpoints:

            if pulls <= maximum_table_pulls:

                display_row(
                    pulls,
                    results[pulls]
                )

        print("=" * 125)

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    display_final_results(
        results[number_of_pulls],
        number_of_pulls,
        initial_chips
    )

    print(
        f"\nTotal calculation time: {elapsed:.2f} seconds"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
!pip install -q gradio numba

import gradio as gr
import matplotlib.pyplot as plt

# ============================================================
# TOWER OF FANTASY - MATRIX CALCULATOR
# ============================================================

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

/* ============================================================
   INPUT FIELDS - SOFT GREY
   ============================================================ */

.gradio-container input:not([type="checkbox"]),
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

/* CALCULATE BUTTON */

#calculate-button {
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

/* RESULTS PANEL */

#results-panel {
    background: #d9dde1 !important;
    color: #172638 !important;
    border: 2px solid #d2b36d !important;
    border-radius: 14px !important;
    padding: 25px !important;
}

#results-panel,
#results-panel p,
#results-panel li,
#results-panel strong,
#results-panel em,
#results-panel span {
    color: #172638 !important;
}

#results-panel h1,
#results-panel h2,
#results-panel h3 {
    color: #183d58 !important;
    font-weight: 800 !important;
}

/* RESULTS TABLE */

#results-panel table {
    width: 100% !important;
    border-collapse: collapse !important;
    background: #d9dde1 !important;
}

#results-panel th {
    background: #c2ccd5 !important;
    color: #183d58 !important;
    padding: 12px !important;
    border: 1px solid #aebfce !important;
}

#results-panel td {
    background: #d9dde1 !important;
    color: #172638 !important;
    padding: 12px !important;
    border: 1px solid #b8c1ca !important;
}

#results-panel tr:nth-child(even) td {
    background: #cbd2d9 !important;
}

/* LINKS */

#results-panel a {
    color: #087eaa !important;
}

/* ============================================================
   CHECKBOX
   ============================================================ */

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
# CALCULATION CALLBACK
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

    # ========================================================
    # PROGRESSION TABLE AND GRAPH
    # ========================================================

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

        # BUILD PROGRESSION TABLE

        for checkpoint in checkpoints:

            row = results[checkpoint]

            output += f"| {checkpoint} "

            for probability in row:
                output += f"| {probability * 100:.4f}% "

            output += "|\n"

        # ====================================================
        # BUILD GRAPH
        # ====================================================

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

    # ========================================================
    # RULE EXPLANATION
    # ========================================================

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
    title="Tower of Fantasy - Matrix Calculator"
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML("""
    <div id="main-header">
        <h1>MATRIX PROBABILITY</h1>
        <h2>TOWER OF FANTASY</h2>
        <p>LIMITED MATRIX CALCULATOR</p>
    </div>
    """)

    # ========================================================
    # LANGUAGE
    # ========================================================

    language = gr.Dropdown(
        choices=["English", "Français"],
        value="English",
        label="Language / Langue",
        interactive=True
    )

    # ========================================================
    # PULLING PARAMETERS
    # ========================================================

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

    # ========================================================
    # MATRIX INVENTORY
    # ========================================================

    gr.Markdown(
        "## YOUR MATRIX INVENTORY",
        elem_classes=["section-title"]
    )

    gr.Markdown(
        "Enter the total number of copies owned for each matrix."
    )

    with gr.Row():

        A = gr.Number(
            value=0,
            precision=0,
            label="Matrix A"
        )

        B = gr.Number(
            value=0,
            precision=0,
            label="Matrix B"
        )

        C = gr.Number(
            value=0,
            precision=0,
            label="Matrix C"
        )

        D = gr.Number(
            value=0,
            precision=0,
            label="Matrix D"
        )

    # ========================================================
    # PROGRESSION OPTION
    # ========================================================

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

    # ========================================================
    # CALCULATE BUTTON
    # ========================================================

    calculate_button = gr.Button(
        "CALCULATE PROBABILITIES",
        elem_id="calculate-button",
        variant="primary"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    gr.Markdown(
        "## PROBABILITY RESULTS",
        elem_classes=["section-title"]
    )

    results_output = gr.Markdown(
        value="Your results will appear here.",
        elem_id="results-panel"
    )

    graph_output = gr.Plot(
        label="Probability progression"
    )

    # ========================================================
    # CALCULATION EVENT
    # ========================================================

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

    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML("""
    <div style="
        text-align:center;
        padding:25px 5px;
        color:#8297aa;
        font-size:11px;
        letter-spacing:1px;
    ">
        COMMUNITY FAN-MADE TOOL<br>
        TOWER OF FANTASY MATRIX PROBABILITY CALCULATOR
    </div>
    """)


# ============================================================
# LAUNCH PUBLIC WEBSITE
# ============================================================

demo.launch(
    share=True,
    show_error=True
)
