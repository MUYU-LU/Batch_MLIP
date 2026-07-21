# Direct ASE/matscipy/auto frontier

| Model | Task | Workload | Backend | B | ASE speedup | matscipy speedup | Peak GB | Neighbor % |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|
| AtomBit | eval | EVAL-H276-R256-v1 | auto (cuda_dense) | 32 | 6.134x | 2.930x | 13.686 | 15.7% |
| AtomBit | eval | EVAL-H276-R32-v1 | auto (cuda_dense) | 32 | 5.993x | 2.946x | 13.686 | 16.0% |
| AtomBit | eval | EVAL-H46-R256-v1 | auto (cuda_dense) | 128 | 18.091x | 3.243x | 10.075 | 18.9% |
| AtomBit | eval | EVAL-H46-R32-v1 | auto (cuda_dense) | 32 | 11.781x | 5.024x | 2.596 | 27.2% |
| AtomBit | eval | EVAL-MIX-R256-v1 | auto (cuda_dense) | 64 | 8.332x | 2.898x | 16.045 | 19.0% |
| AtomBit | eval | EVAL-MIX-R32-v1 | auto (cuda_dense) | 32 | 7.783x | 2.967x | 7.898 | 23.9% |
| MACE-OFF-Small | eval | EVAL-H276-R256-v1 | auto (cuda_dense) | 64 | 8.488x | 1.882x | 29.564 | 15.2% |
| MACE-OFF-Small | eval | EVAL-H276-R32-v1 | auto (cuda_dense) | 32 | 25.425x | 1.817x | 14.818 | 16.6% |
| MACE-OFF-Small | eval | EVAL-H46-R256-v1 | auto (cuda_dense) | 256 | 32.078x | 2.017x | 19.625 | 18.6% |
| MACE-OFF-Small | eval | EVAL-H46-R32-v1 | auto (cuda_dense) | 32 | 66.801x | 1.692x | 2.559 | 29.5% |
| MACE-OFF-Small | eval | EVAL-MIX-R256-v1 | auto (cuda_dense) | 128 | 12.604x | 1.865x | 34.041 | 17.4% |
| MACE-OFF-Small | eval | EVAL-MIX-R32-v1 | auto (cuda_dense) | 32 | 38.814x | 1.710x | 8.640 | 25.4% |
| AtomBit | nve | MD-NVE-H276-R32-v1 | auto (cuda_dense,matscipy) | 16 | 3.979x | 1.054x | 45.355 | 5.9% |
| AtomBit | nve | MD-NVE-H46-R32-v1 | auto (cuda_dense,matscipy) | 32 | 10.617x | 1.011x | 4.236 | 6.7% |
| AtomBit | nve | MD-NVE-MIX-R32-v1 | auto (cuda_dense,matscipy) | 32 | 6.255x | 1.071x | 7.982 | 7.6% |
| MACE-OFF-Small | nve | MD-NVE-H276-R32-v1 | auto (cuda_dense,matscipy) | 32 | 6.095x | 1.032x | 18.889 | 5.1% |
| MACE-OFF-Small | nve | MD-NVE-H46-R32-v1 | matscipy (matscipy) | 32 | 16.274x | 1.000x | 3.238 | 4.9% |
| MACE-OFF-Small | nve | MD-NVE-MIX-R32-v1 | auto (cuda_dense,matscipy) | 32 | 8.300x | 1.011x | 11.067 | 7.7% |
| AtomBit | variable_bfgs | BFGS-H276-R32 | auto (cuda_dense,matscipy) | 32 | 8.732x | 1.377x | 12.361 | 4.1% |
| AtomBit | variable_bfgs | BFGS-H46-R32 | auto (cuda_dense,matscipy) | 32 | 6.957x | 1.214x | 2.366 | 9.8% |
| MACE-OFF-Small | variable_bfgs | BFGS-H276-R32 | auto (cuda_dense,matscipy) | 32 | 8.925x | 1.106x | 13.635 | 4.3% |
| MACE-OFF-Small | variable_bfgs | BFGS-H46-R32 | auto (cuda_dense,matscipy) | 32 | 8.806x | 1.044x | 2.330 | 9.4% |
| AtomBit | variable_fire | FIRE-H276-R32 | auto (cuda_dense,matscipy) | 32 | 6.367x | 1.701x | 12.183 | 9.2% |
| AtomBit | variable_fire | FIRE-H46-R32 | auto (cuda_dense,matscipy) | 16 | 3.618x | 1.025x | 1.224 | 11.8% |
