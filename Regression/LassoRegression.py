

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    """
    Purpose: Compute the descent step for one feature
    Input  : Feature index, normalized feature matrix, output,
             feature weights and L1_penalty
    Output : Descent step for feature
    """
    predictions = feature_matrix.dot(weights)
    rho = (feature_matrix[:, i].T).dot(output - predictions + (weights[i] * feature_matrix[:, i]))
    if i==0:
        new_weight = rho
    elif rho < (-l1_penalty/2.0):
        new_weight = rho + (l1_penalty/2.0)
    elif rho > (l1_penalty/2.0):
        new_weight = rho - (l1_penalty/2.0)
    else:
        new_weight = 0.0
    return new_weight