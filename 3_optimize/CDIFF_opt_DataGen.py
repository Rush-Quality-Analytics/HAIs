import HAI_optimize

obs_days = 'CDIFF patient days'
pred_cases = 'CDIFF Predicted Cases'
obs_cases = 'CDIFF Observed Cases'
hai = 'CDIFF'
z_ran = [1, 40000]
pi_ran = [0.00001, 0.01]

HAI_optimize.optimize(obs_days, pred_cases, obs_cases, hai, z_ran, pi_ran)
