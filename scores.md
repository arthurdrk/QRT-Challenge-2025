| model                                 | train IPCW | test IPCW | CV train mean ICPW | CV test mean ICPW | leaderbord IPCW |
|---------------------------------------|------------|-----------|--------------------|-------------------|-----------------|
| benchmark (CoxPH)                     | 0.68       | 0.68      | 0.6845             | 0.6842            | 0.6541          |
| CoxPH enhanced                        | 0.7105     | 0.7079    | 0.7108             | 0.7074            |                 |
| XGBSurvival                           | 0.7651     | 0.7094    | 0.7569             | 0.7049            |                 |
| XGBSurv Tuned                         |            |           | 0.7603             | 0.7101            |                 |
| CoxNet Tuned                          |            |           | 0.7109             | 0.7088            |                 |
| IPCRidge                              |            |           | 0.3005             | 0.3047            |                 |
| EnsembleSelection(CoxPH, CoxNet, XGB) | 0.7648     | 0.7101    | 0.7588             | 0.7101            | 0.7330          |