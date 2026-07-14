## Acknowledgements

This repository adapts the **Knockoff-ML** framework developed by Wang, Li & Yang.

- **Original software:** [keiwong493/Knockoff-ML](https://github.com/keiwong493/Knockoff-ML)
- **Original publication:** Wang Q, Li L, Yang Y. Knockoff-ML: a knockoff machine learning 
  framework for controlled variable selection and risk stratification in electronic health 
  record data. *npj Digital Medicine* **8**, 723 (2025). 
  https://doi.org/10.1038/s41746-025-02102-2

### Code adaptation

The following modules are Python ports/adaptations of the original R implementation 
(`Knockoff-ML.R`) and workflow (`Knockoff-ML_FI.ipynb`):

| This repository | Source (Knockoff-ML) |
|-----------------|----------------------|
| `knockoff_tb/knockoffs.py` | `generate_knockoff()`, `create.MK()` |
| `knockoff_tb/selection.py` | `Get_select_info()`, knockoff filter functions |
| `knockoff_tb/feature_importance.py` | `calculate_fi()` workflow |

**Knockoff-TB** extends Knockoff-ML to *M. tuberculosis* genotypic resistance data 
(CRyPTIC cohorts); all TB-specific data handling, evaluation, and interaction testing 
are original to this repository.

Licensed under **GNU GPL v3.0** (inherited from Knockoff-ML).
