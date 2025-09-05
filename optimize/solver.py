import numpy as np
from optimize.group_update import update_group

def sparse_group_lasso_solver(x_groups,y,lambda1,lambda2,max_iter=100,tol=1e-4):
    """
    Full coordinate descent solver for the Sparse Group Lasso

    Parameters:
        X_groups : list of np.ndarray
            List of group design matrices (each: n_samples x k_ℓ)
        y : np.ndarray
            Target vector (n_samples,)
        lambda1 : float
            Group-level regularization strength
        lambda2 : float
            L1 feature-level regularization strength
        max_iter : int
            Maximum number of coordinate descent iterations
        tol : float
            Convergence threshold (on full beta)

    Returns:
        beta_groups : list of np.ndarray
            List of optimized coefficient vectors per group

    """

    L = len(x_groups)

    n = y.shape[0]

    #Step1: Initialize all beta_l = 0
    beta_groups = [np.zeros(x_l.shape[1]) for x_l in x_groups]

    for iteration in range(max_iter):
        beta_old = np.concatenate(beta_groups)

        #Step2: Loop over each group
        for l in range(L):

            # Reconstruct current residual: r = y - sum_{k ≠ ℓ} X_k β_k
            r = y.copy()

            for k in range(L):
                if k != l:
                    r -= x_groups[k] @ beta_groups[k]

            #Step3: Update current group:
            beta_groups[l] = update_group(x_groups[l],beta_groups[l],r,lambda1,lambda2)

        #Step4: Check convergence
        beta_new = np.concatenate(beta_groups)

        if np.linalg.norm(beta_new - beta_old) < tol:
            print(f"Converged after {iteration+1} iterations.")
            break

    return beta_groups    