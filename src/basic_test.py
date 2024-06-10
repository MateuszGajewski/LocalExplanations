from sklearn import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import cvxpy as cp
import shap
import lime


def explain_shap(clf, X):
    ex = shap.TreeExplainer(clf)
    shap_values = ex.shap_values(X)
    sum = 0
    for i in shap_values:
        sum += abs(i)
    print(shap_values / sum)


def explain_lime(clf, X, X_train):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        verbose=True,
        mode="regression",
    )
    exp = explainer.explain_instance(X, clf.predict, num_features=13)
    print(exp.as_list())

    return
    for i in shap_values:
        sum += abs(i)
    print(shap_values / sum)


def solve_for_l2(A_eq, b_eq, A_lb, b_lb, e):
    n_variables = A_eq.shape[1]

    x = cp.Variable(n_variables)
    objective = cp.Minimize(cp.norm2(x))

    constraints = [A_eq @ x == b_eq, (A_lb @ x + e * np.ones(len(A_lb))) >= b_lb]
    # constraints += [e >= 0]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.SolverError:
        raise ValueError("Could not solve linear program") from err
    return x.value


def build_constraints(A_eq, b_eq, A_lb, b_lb):
    n_variables = A_eq.shape[1]

    x = cp.Variable(n_variables)
    e = cp.Variable()

    objective = cp.Minimize(e)
    # objective = cp.Minimize(cp.norm2(x))

    constraints = [A_eq @ x == b_eq, (A_lb @ x + e * np.ones(len(A_lb))) >= b_lb]
    constraints += [e >= 0]
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve()
    except cp.SolverError:
        raise ValueError("Could not solve linear program") from err

    subsidy = e.value.item()
    x = solve_for_l2(A_eq, b_eq, A_lb, b_lb, subsidy)
    return x, subsidy


def generate_points(clf: object, X: np.ndarray):
    b_eq = clf.predict([X])
    A_eq = np.ones((1, len(X)))

    A_lb = np.ones((len(X) ** 2, len(X)))
    b_lb = np.zeros((len(X) ** 2))
    indx = 0
    for i in range(len(X)):
        for j in range(len(X)):
            avg = 0
            for _ in range(0, 100):
                X_ = X.copy()
                X_[i] += np.random.randn() * 0.1
                if i != j:
                    X_[j] += np.random.randn() * 0.1
                avg += clf.predict([X_])
            avg /= 100
            A_lb[indx, i] = 0
            A_lb[indx, j] = 0
            b_lb[indx] = avg[0]  # clf.predict([X_])
            indx += 1

    x, e = build_constraints(A_eq, b_eq, A_lb, b_lb)
    sum_ = sum([abs(i) for i in x])
    print(x)# / sum_)
    print(e)


def generate_points_separate(clf: object, X: np.ndarray):
    n_points = 5
    b_eq = clf.predict([X])
    A_eq = np.ones((1, len(X)))

    A_lb = np.ones((n_points * len(X) ** 2, len(X)))
    b_lb = np.zeros((n_points * len(X) ** 2))
    indx = 0
    for i in range(len(X)):
        for j in range(len(X)):
            for _ in range(0, n_points):
                X_ = X.copy()
                X_[i] += np.random.randn() * 0.01
                if i != j:
                    X_[j] += np.random.randn() * 0.01
                A_lb[indx, i] = 0
                A_lb[indx, j] = 0
                b_lb[indx] = clf.predict([X_])[0]
                indx += 1

    x, e = build_constraints(A_eq, b_eq, A_lb, b_lb)
    sum_ = sum([abs(i) for i in x])
    print(x / sum_)
    print(e)


def train_tree():
    X, y = load_diabetes(return_X_y=True)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X, y)
    # print(X)
    explain_lime(regressor, X[3], X)
    generate_points(regressor, X[3])
    explain_shap(regressor, X[3])


if __name__ == "__main__":
    train_tree()
