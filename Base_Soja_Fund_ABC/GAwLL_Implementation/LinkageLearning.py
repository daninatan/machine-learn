import random
from typing import List
import numpy as np

class LinkageLearning:
    """
    Implements Linkage Learning mechanisms to estimate variable interactions.

    By analyzing performance changes when specific genes (g, h) are flipped,
    this class builds an empirical model of the search space's topology.
    """

    def __init__(self, evig, importance, eval_perf_batch):
        """
        Initializes the Linkage Learning module.

        Args:
            evig: Instance of eVIG to store interactions.
            importance: Instance of Importance to store variable impact.
            eval_perf_batch: Callable that evaluates a batch of chromosomes.
        """
        self.evig = evig
        self.importance = importance
        self.eval_perf = eval_perf_batch

    def update_linkage_model(self, g: int, h: int, fx: float, fxg: float, fxh: float, fxgh: float) -> None:
        """
        Estimates VImp and VInt using the performance differences.

        The formula used for interaction is:
        Δ = fxgh - fxg - fxh + fx

        Args:
            g, h: Indices of the targeted variables.
            fx: Base performance.
            fxg, fxh: Performance with one gene flipped.
            fxgh: Performance with both genes flipped.
        """
        EPS2 = 1.0e-10

        # Safety check: avoid estimation if performance is negligible
        if any(val < EPS2 for val in [fx, fxg, fxh, fxgh]):
            return

        # --- Variable Importance (VImp) ---
        # Impact of g alone
        df_g = fxg - fx
        if abs(df_g) > EPS2:
            self.importance.add_importance(g, df_g)

        # Impact of h alone
        df_h = fxh - fx
        if abs(df_h) > EPS2:
            self.importance.add_importance(h, df_h)

        # --- Variable Interaction (VInt) ---
        # Interaction is the difference between the joint impact and sum of individual impacts
        df_interaction = fxgh - fxg - fxh + fx

        if abs(df_interaction) > EPS2:
            # Symmetrically update the eVIG matrix
            self.evig.add_edge(g, h, df_interaction)

            # Conditional Importance
            df_g_cond = fxgh - fxh
            if abs(df_g_cond) > EPS2:
                self.importance.add_importance(g, df_g_cond)

            df_h_cond = fxgh - fxg
            if abs(df_h_cond) > EPS2:
                self.importance.add_importance(h, df_h_cond)

    def mutation_ll(self, parent) -> List[np.ndarray]:
        """
        Performs mutation guided by Linkage Learning.
        Generates 3 offspring to probe the search space for interactions.

        Returns:
            List[np.ndarray]: [xg_chrom, xh_chrom, xgh_chrom] for the GA engine.
        """
        parent_chrom = parent.chromosome
        chrom_size = len(parent_chrom)
        fx = parent.fitness

        # 1. Random Selection of two distinct genes
        g, h = random.sample(range(chrom_size), 2)

        # 2. Generating perturbed chromosomes
        # Using boolean NOT for fast flipping
        xg_chrom = parent_chrom.copy()
        xg_chrom[g] = not xg_chrom[g]

        xh_chrom = parent_chrom.copy()
        xh_chrom[h] = not xh_chrom[h]

        xgh_chrom = xg_chrom.copy()
        xgh_chrom[h] = not xgh_chrom[h]

        # 3. Batch Evaluation
        # We send all 3 to the batch evaluator at once
        fxg, fxh, fxgh = self.eval_perf([xg_chrom, xh_chrom, xgh_chrom])

        # 4. Estimation Model Mapping
        # Maps the 4 states (00, 01, 10, 11).
        g_val, h_val = parent_chrom[g], parent_chrom[h]

        if not g_val and not h_val:
            self.update_linkage_model(g, h, fx, fxg, fxh, fxgh)
        elif g_val and not h_val:
            self.update_linkage_model(g, h, fxg, fx, fxgh, fxh)
        elif not g_val and h_val:
            self.update_linkage_model(g, h, fxh, fxgh, fx, fxg)
        else:
            self.update_linkage_model(g, h, fxgh, fxh, fxg, fx)

        return [xg_chrom, xh_chrom, xgh_chrom]