"""
Created on May 05, 2015

@author: Katharina Eggensperger
"""
import functools
from sklearn.ensemble import RandomForestRegressor
from epm.models import bootstrapModel, external_rfr, external_rfr_predict,\
    external_rfr_quantile

# This is a collection of all possible models

RNG = 0

basic_models = {"rf": functools.partial(RandomForestRegressor, n_estimators=10,
                                        random_state=RNG),
                "rfr": functools.partial(external_rfr.ExternalRFR,
                                         num_trees=48,
                                         seed=RNG,
                                         do_bootstrapping=False,
                                         frac_points_per_tree=0.635453473468,
                                         max_depth=26,
                                         max_num_nodes=16678,
                                         min_samples_in_leaf=1,
                                         min_samples_to_split=5,
                                         ratio_features=0.284881375208),
                "rfr_predict": functools.partial(
                        external_rfr_predict.ExternalRFRPredict,
                        num_trees=48,
                        seed=RNG,
                        compute_oob_error=True,
                        do_bootstrapping=False,
                        frac_points_per_tree=0.635453473468,
                        max_depth=26,
                        max_num_nodes=16678,
                        min_samples_in_leaf=1,
                        min_samples_to_split=5,
                        ratio_features=0.284881375208),
                "rfr_quantile": functools.partial(
                        external_rfr_quantile.ExternalRFRQuantile,
                        num_trees=48,
                        seed=RNG,
                        do_bootstrapping=False,
                        frac_points_per_tree=0.8, #0.635453473468,
                        max_depth=26,
                        max_num_nodes=50000, #16678,
                        min_samples_in_leaf=1,
                        min_samples_to_split=5,
                        epsilon_purity=1e-8,
                        ratio_features=0.284881375208),
                }

bootstrap_settings = {"n_bootstrap_samples": 50,
                      "bootstrap_size": 0.7,
                      "debug": False,
                      "rng": RNG}

# keys = [name, predict model, imputation model
model_dict = {
    "rf": ("RandomForest", basic_models["rf"],
           functools.partial(bootstrapModel.bootstrapModel,
                             base_estimator=basic_models["rf"],
                             **bootstrap_settings)),
    "rfr": ("rfr", basic_models["rfr_predict"], basic_models["rfr"]),
    "rfrq": ("rfrq", basic_models["rfr_quantile"], basic_models["rfr"]),

}
