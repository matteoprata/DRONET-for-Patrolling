
python -m src.main -de REW_1_PEN_1_SWA_500_LR_0001_DF_98 -rew 1 -pe -1 &
python -m src.main -de REW_1_PEN_5_SWA_500_LR_0001_DF_98 -rew 1 -pe -5 &
python -m src.main -de REW_1_PEN_10_SWA_500_LR_0001_DF_98 -rew 1 -pe -10 &
python -m src.main -de REW_0_PEN_1_SWA_500_LR_0001_DF_98 -rew 0 -pe -1 &
python -m src.main -de REW_0_PEN_5_SWA_500_LR_0001_DF_98 -rew 0 -pe -5 &
python -m src.main -de REW_0_PEN_10_SWA_500_LR_0001_DF_98 -rew 0 -pe -10 &

python -m src.main -de REW_1_PEN_5_SWA_1000_LR_0001_DF_98 -rew 1 -pe -5 -sw 1000 &
python -m src.main -de REW_1_PEN_5_SWA_2000_LR_0001_DF_98 -rew 1 -pe -5 -sw 2000 &

python -m src.main -de REW_1_PEN_5_SWA_500_LR_001_DF_98 -rew 1 -pe -5 -lr 0.001 &
python -m src.main -de REW_1_PEN_5_SWA_500_LR_0001_DF_50 -rew 1 -pe -5 -df 0.50 &
python -m src.main -de REW_1_PEN_5_SWA_500_LR_0001_DF_75 -rew 1 -pe -5 -df 0.75 &
python -m src.main -de REW_1_PEN_5_SWA_500_LR_0001_DF_85 -rew 1 -pe -5 -df 0.85 &
python -m src.main -de REW_1_PEN_5_SWA_500_LR_0001_DF_90 -rew 1 -pe -5 -df 0.90