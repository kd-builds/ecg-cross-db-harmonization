# Cross-Database ECG Evaluation Reporting Checklist

- [ ] Database-level sample counts and unique patient counts reported
- [ ] Per-class prevalence and annotated coverage reported per database
- [ ] Excluded and masked record counts documented with reasons
- [ ] Primary metric (Macro-F1) reported on the preregistered shared class set
- [ ] Undefined metrics (e.g., AUROC when a class has 0 positives in a cohort) explicitly reported as `"not estimable"` rather than silently averaged
- [ ] Internal-to-external performance difference ($\Delta$ Macro-F1) quantified
- [ ] Patient-cluster bootstrap 95% confidence intervals reported (or grouping limitation stated where patient IDs are absent)
- [ ] Conservative vs. broad label-mapping sensitivity results reported separately from primary locked evaluation
