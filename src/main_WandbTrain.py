
# def main_sweep():
#     """ Run sweeps to monitor models performances live. """
#     sweep_id = wandb.sweep(sweep=sweep_configuration, project=cst.PROJECT_NAME)
#     wandb.agent(sweep_id, function=__run_sweep)
#
# def __run_sweep():
#     """ the place where to run simulations and experiments. """
#
#     try:
#         with wandb.init() as wandb_instance:
#             wandb_config = wandb_instance.config
#
#             learning = conf.LEARNING_PARAMETERS
#
#             learning[cst.HyperParameters.LR.value] = wandb_config[cst.HyperParameters.LR.value]
#             learning[cst.HyperParameters.DISCOUNT_FACTOR.value] = wandb_config[cst.HyperParameters.DISCOUNT_FACTOR.value]
#             learning[cst.HyperParameters.SWAP_MODELS.value] = wandb_config[cst.HyperParameters.SWAP_MODELS.value]
#
#             learning[cst.HyperParameters.MLP_HID1.value] = wandb_config[cst.HyperParameters.MLP_HID1.value]
#             learning[cst.HyperParameters.MLP_HID2.value] = wandb_config[cst.HyperParameters.MLP_HID2.value]
#             learning[cst.HyperParameters.MLP_HID3.value] = wandb_config[cst.HyperParameters.MLP_HID3.value]
#
#             conf.IS_ALLOW_SELF_LOOP = wandb_config[cst.HyperParameters.IS_SELF_LOOP.value]
#
#             sim = PatrollingSimulator(learning=learning,
#                                       drone_max_battery=wandb_config[cst.HyperParameters.BATTERY.value],
#                                       n_epochs=wandb_config[cst.HyperParameters.N_EPOCHS.value],
#                                       n_episodes=wandb_config[cst.HyperParameters.N_EPISODES.value],
#                                       episode_duration=wandb_config[cst.HyperParameters.DURATION_EPISODE.value],
#                                       wandb=wandb_instance)
#             sim.run()
#
#     except Exception as e:
#         # exit gracefully, so wandb logs the problem
#         print(traceback.print_exc(), file=sys.stderr)
#         exit(1)