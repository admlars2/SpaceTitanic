RANDOM_SEED = 42
N_SPLITS = 5
CABIN_BIN_SIZE = 50
AGE_BINS = [0, 5, 12, 18, 25, 35, 45, 55, 65, 200]

GRADIENTBOOST_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_leaf_nodes": 31,
    "max_iter": 600,
    "min_samples_leaf": 20,
    "l2_regularization": 0.2,
    "early_stopping": True,
}

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Accuracy",
    "depth": 10,
    "learning_rate": 0.04,
    "iterations": 2000,
    "l2_leaf_reg": 4.0,
    "subsample": 0.8,
    "rsm": 0.8,
    "bagging_temperature": 1.0,
    "random_seed": RANDOM_SEED,
    "verbose": False,
}

GROUP_MODE_COLS = ["HomePlanet", "Destination", "CabinDeck", "CabinSide"]
