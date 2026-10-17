# Task C baselines

Written by `src/models/task_c_baselines.py` **before any C1 or C2 model was trained**. Splits by the
row's quarter: train 2018-2021, validate 2022, test 2023Q1-2026Q1 (the last quarter with a label). Seed
funds only unless the split says `holdout`. The embargo (see `src/models/task_c_common.py`) removed
39,803 train rows whose labels became public after the first 2022 prediction date.

## Frames

| frame | rows | funds | first_q | last_q | max_label_known_at | first_pred_date |
|---|---|---|---|---|---|---|
| train | 465781 | 77 | 2018Q1 | 2021Q4 | 2022-05-04 | 2018-05-11 |
| val | 161344 | 77 | 2022Q1 | 2022Q4 | 2023-05-25 | 2022-05-04 |
| trainval | 627992 | 77 | 2018Q1 | 2022Q4 | 2023-05-09 | 2018-05-11 |
| test | 488024 | 77 | 2023Q1 | 2026Q1 | 2026-08-17 | 2023-05-09 |
| test_holdout | 90420 | 16 | 2023Q1 | 2026Q1 | 2026-08-14 | 2023-05-10 |

## Class shares

Hold is the *smallest* class, not the 50-60% the plan guessed: these are active managers with 10% bands,
so "predict hold" is a weak baseline and accuracy is meaningless. Macro-F1 and log-loss are the metrics.

| frame | exit | trim | hold | add |
|---|---|---|---|---|
| test | 0.194 | 0.301 | 0.181 | 0.325 |
| train | 0.213 | 0.311 | 0.176 | 0.300 |
| val | 0.201 | 0.285 | 0.207 | 0.307 |

By style (seed funds, all quarters 2018-2026Q1):

| style | rows | funds | exit | trim | hold | add |
|---|---|---|---|---|---|---|
| multi_manager | 528269 | 7 | 0.251 | 0.327 | 0.084 | 0.338 |
| quant | 422439 | 5 | 0.122 | 0.329 | 0.209 | 0.341 |
| event_driven | 57587 | 12 | 0.369 | 0.155 | 0.291 | 0.185 |
| activist | 57167 | 15 | 0.123 | 0.143 | 0.600 | 0.133 |
| growth_tech | 33517 | 18 | 0.298 | 0.251 | 0.236 | 0.216 |
| discretionary_other | 28043 | 3 | 0.237 | 0.281 | 0.192 | 0.290 |
| healthcare | 21123 | 8 | 0.205 | 0.139 | 0.502 | 0.155 |
| value_concentrated | 6807 | 9 | 0.145 | 0.166 | 0.563 | 0.126 |

## C1 baselines

`ret_clean` drops rows whose stock's return over the quarter of the move is a 0% deal fallback or
missing (`ret_source`); `excl_multi_quant` drops multi-manager and quant funds. Persistence's log-loss is
from probability 1 on hold, clipped at 1e-6.

| split | subset | baseline | n | macro_f1 | log_loss | accuracy | exit_auc | R_exit | R_trim | R_hold | R_add |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | all | persistence | 161344 | 0.086 | 10.952 | 0.207 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| val | all | class_prior | 161344 | 0.111 | 1.373 | 0.285 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| val | all | fund_prior | 161344 | 0.364 | 1.298 | 0.359 | 0.635 | 0.453 | 0.360 | 0.394 | 0.272 |
| val | all | fund_last_q | 161344 | 0.350 | 1.331 | 0.352 | 0.627 | 0.270 | 0.304 | 0.453 | 0.382 |
| val | ret_clean | persistence | 157441 | 0.087 | 10.891 | 0.212 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| val | ret_clean | class_prior | 157441 | 0.113 | 1.368 | 0.292 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| val | ret_clean | fund_prior | 157441 | 0.366 | 1.293 | 0.360 | 0.643 | 0.475 | 0.360 | 0.392 | 0.272 |
| val | ret_clean | fund_last_q | 157441 | 0.353 | 1.319 | 0.356 | 0.638 | 0.283 | 0.304 | 0.452 | 0.382 |
| val | excl_multi_quant | persistence | 31179 | 0.152 | 7.790 | 0.436 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| val | excl_multi_quant | class_prior | 31179 | 0.070 | 1.507 | 0.162 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| val | excl_multi_quant | fund_prior | 31179 | 0.310 | 1.253 | 0.454 | 0.624 | 0.423 | 0.005 | 0.745 | 0.189 |
| val | excl_multi_quant | fund_last_q | 31179 | 0.351 | 1.398 | 0.453 | 0.602 | 0.264 | 0.186 | 0.749 | 0.212 |
| test | all | persistence | 488024 | 0.077 | 11.317 | 0.181 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| test | all | class_prior | 488024 | 0.116 | 1.355 | 0.301 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| test | all | fund_prior | 488024 | 0.373 | 1.271 | 0.369 | 0.678 | 0.472 | 0.354 | 0.364 | 0.323 |
| test | all | fund_last_q | 488024 | 0.380 | 1.267 | 0.381 | 0.688 | 0.327 | 0.388 | 0.333 | 0.434 |
| test | ret_clean | persistence | 482776 | 0.077 | 11.300 | 0.182 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| test | ret_clean | class_prior | 482776 | 0.117 | 1.353 | 0.304 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| test | ret_clean | fund_prior | 482776 | 0.374 | 1.268 | 0.370 | 0.684 | 0.485 | 0.354 | 0.362 | 0.323 |
| test | ret_clean | fund_last_q | 482776 | 0.381 | 1.264 | 0.383 | 0.693 | 0.336 | 0.388 | 0.330 | 0.434 |
| test | excl_multi_quant | persistence | 80761 | 0.143 | 8.294 | 0.400 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| test | excl_multi_quant | class_prior | 80761 | 0.079 | 1.477 | 0.187 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| test | excl_multi_quant | fund_prior | 80761 | 0.340 | 1.226 | 0.471 | 0.702 | 0.475 | 0.013 | 0.802 | 0.215 |
| test | excl_multi_quant | fund_last_q | 80761 | 0.374 | 1.247 | 0.484 | 0.720 | 0.432 | 0.203 | 0.821 | 0.099 |
| test_holdout | all | persistence | 90420 | 0.071 | 11.514 | 0.167 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| test_holdout | all | class_prior | 90420 | 0.110 | 1.365 | 0.283 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| test_holdout | all | fund_prior | 90420 | 0.243 | 1.370 | 0.291 | 0.540 | 0.333 | 0.000 | 0.302 | 0.516 |
| test_holdout | all | fund_last_q | 90420 | 0.319 | 1.362 | 0.339 | 0.576 | 0.285 | 0.179 | 0.281 | 0.557 |
| test_holdout | ret_clean | persistence | 89327 | 0.072 | 11.507 | 0.167 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| test_holdout | ret_clean | class_prior | 89327 | 0.111 | 1.362 | 0.286 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| test_holdout | ret_clean | fund_prior | 89327 | 0.243 | 1.371 | 0.291 | 0.539 | 0.336 | 0.000 | 0.301 | 0.516 |
| test_holdout | ret_clean | fund_last_q | 89327 | 0.319 | 1.361 | 0.340 | 0.574 | 0.284 | 0.180 | 0.281 | 0.557 |
| test_holdout | excl_multi_quant | persistence | 10133 | 0.129 | 9.012 | 0.348 | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| test_holdout | excl_multi_quant | class_prior | 10133 | 0.080 | 1.475 | 0.190 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| test_holdout | excl_multi_quant | fund_prior | 10133 | 0.243 | 1.337 | 0.394 | 0.681 | 0.271 | 0.000 | 0.877 | 0.054 |
| test_holdout | excl_multi_quant | fund_last_q | 10133 | 0.282 | 1.491 | 0.408 | 0.644 | 0.376 | 0.052 | 0.800 | 0.053 |

## C2 baselines (new buys, top 20)

Per fund-quarter with at least one new buy, equal weight. `reachable` = share of new buys held in any
book visible on the prediction date (the most any recommender over those books could recall).

| split | cohort | subset | method | fund_quarters | recall@20 | ndcg@20 | hit_rate | reachable |
|---|---|---|---|---|---|---|---|---|
| test | holdout | all | cluster_popularity | 190 | 0.013 | 0.073 | 0.395 | 0.916 |
| test | holdout | all | popularity | 190 | 0.013 | 0.073 | 0.395 | 0.916 |
| test | holdout | all | reentry | 190 | 0.049 | 0.141 | 0.574 | 0.916 |
| test | holdout | excl_multi_quant | cluster_popularity | 138 | 0.014 | 0.017 | 0.196 | 0.914 |
| test | holdout | excl_multi_quant | popularity | 138 | 0.014 | 0.017 | 0.196 | 0.914 |
| test | holdout | excl_multi_quant | reentry | 138 | 0.060 | 0.070 | 0.420 | 0.914 |
| test | holdout | ret_clean | cluster_popularity | 190 | 0.014 | 0.073 | 0.395 | 0.916 |
| test | holdout | ret_clean | popularity | 190 | 0.014 | 0.073 | 0.395 | 0.916 |
| test | holdout | ret_clean | reentry | 190 | 0.054 | 0.143 | 0.574 | 0.916 |
| test | seed | all | cluster_popularity | 893 | 0.045 | 0.094 | 0.469 | 0.863 |
| test | seed | all | popularity | 893 | 0.012 | 0.066 | 0.325 | 0.863 |
| test | seed | all | reentry | 893 | 0.090 | 0.164 | 0.664 | 0.863 |
| test | seed | excl_multi_quant | cluster_popularity | 737 | 0.052 | 0.046 | 0.357 | 0.863 |
| test | seed | excl_multi_quant | popularity | 737 | 0.012 | 0.018 | 0.186 | 0.863 |
| test | seed | excl_multi_quant | reentry | 737 | 0.105 | 0.106 | 0.593 | 0.863 |
| test | seed | ret_clean | cluster_popularity | 887 | 0.051 | 0.097 | 0.472 | 0.869 |
| test | seed | ret_clean | popularity | 887 | 0.013 | 0.067 | 0.327 | 0.869 |
| test | seed | ret_clean | reentry | 887 | 0.097 | 0.168 | 0.669 | 0.869 |
| val | seed | all | cluster_popularity | 274 | 0.050 | 0.096 | 0.471 | 0.914 |
| val | seed | all | popularity | 274 | 0.009 | 0.060 | 0.288 | 0.914 |
| val | seed | all | reentry | 274 | 0.109 | 0.175 | 0.679 | 0.914 |
| val | seed | excl_multi_quant | cluster_popularity | 226 | 0.058 | 0.049 | 0.358 | 0.917 |
| val | seed | excl_multi_quant | popularity | 226 | 0.009 | 0.017 | 0.146 | 0.917 |
| val | seed | excl_multi_quant | reentry | 226 | 0.128 | 0.121 | 0.611 | 0.917 |
| val | seed | ret_clean | cluster_popularity | 272 | 0.051 | 0.097 | 0.474 | 0.920 |
| val | seed | ret_clean | popularity | 272 | 0.009 | 0.061 | 0.290 | 0.920 |
| val | seed | ret_clean | reentry | 272 | 0.113 | 0.178 | 0.684 | 0.920 |
