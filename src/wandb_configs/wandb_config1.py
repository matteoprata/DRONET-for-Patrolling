
import src.utilities.config as co

sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',

    'metric': {'goal': 'maximize',
               'name': 'cumulative_reward'},

    'parameters': {"n_epochs":         {'values': [200]},
                   "n_episodes":       {'values': [50]},
                   "episode_duration": {'values': [1 * co.Time.HOUR.value]},
                   "learning_rate":    {'min': 0.0005, 'max': 0.005},
                   "discount_factor":  {'values': [1]},
                   "n_hidden_neurons_lv1":  {'values': [10]},
                   "n_hidden_neurons_lv2":  {'values': [0]},
                   "n_hidden_neurons_lv3":  {'values': [0]},
                   "swap_models_every_decision": {'values': [500]},
                   "battery":                    {'values': [120 * co.Time.MIN.value]},
                   "is_allow_self_loop":         {'values': [0]}
                   }
}
