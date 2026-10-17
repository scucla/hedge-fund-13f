# Task C leakage audit

Written by `src/models/leakage_audit.py` (also run by `tests/test_task_c.py`). The prediction date is the
fund's own quarter-q 13F-HR filing date; every check counts violations and must be 0.

| check | rows_checked | violations | note |
|---|---|---|---|
| known_at:own | 1309286 | 0 | 0 late, 0 with features but no own_known_at |
| known_at:stock | 1309286 | 0 | 0 late, 0 with features but no stock_known_at |
| known_at:prior | 1309286 | 0 | 0 late, 0 with features but no prior_known_at |
| known_at:crowd | 1309286 | 0 | 0 late, 0 with features but no crowd_known_at |
| known_at:peer | 1309286 | 0 | 0 late, 0 with features but no peer_known_at |
| known_at:cluster | 1309286 | 0 | 0 late, 0 with features but no cluster_known_at |
| label_after_pred | 1309286 | 0 |  |
| peer_recompute | 155982 | 0 | 300 sampled fund-quarters, independent count from raw positions |
| peer_filing_dates | 224099 | 0 | info: a naive quarter-q peer count would use a not-yet-public filing on 360,074 of 1,309,286 rows |
| cluster_vintage | 1154579 | 0 | 79% of rows use an earlier quarter's clustering (q's was not yet public) |
| embargo:train | 465781 | 0 | first val prediction 2022-05-04, latest train label 2022-05-04 |
| embargo:trainval | 627992 | 0 | first test prediction 2023-05-09, latest trainval label 2023-05-09 |
| fit_frames | 1093773 | 0 | fit rows outside their years or not seed |
| amendments | 312 | 0 | 312 panel fund-quarters have a later 13F-HR/A; all predicted from the original filing |
| forbidden_features | 52 | 0 |  |
| c2_inputs | 600 | 0 | recommender inputs per (method, quarter, prediction date) |
