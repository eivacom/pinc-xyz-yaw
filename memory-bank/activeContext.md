# Active Context

**Project Name:** pinc-xyz-yaw

**Current Work Focus:** Evaluating model performance on different trajectory types.

**Recent Changes:**
- Modified `data/data_utility.py` to add functions for generating 'line', 'circle', and 'figure8' input trajectories.
- Modified `data/create_data_2.py` to generate ground truth datasets for these new trajectory types (`test_set_line`, `test_set_circle`, `test_set_figure8`), saving them to the project root.
- Created a new evaluation script `scripts/evaluate_multi_rollouts.py` to perform rollout predictions using multiple models on multiple datasets.
- Executed `scripts/evaluate_multi_rollouts.py` using models `_best_dev_l_0` and `_best_dev_l_993` on the `test_set_line`, `test_set_circle`, and `test_set_figure8` datasets. (Skipped `_best_dev_l_50` as file was not found).
- Saved evaluation results (plots, summary files) to `results/multi_rollout_results/`, organized by dataset and model.

**Next Steps:**
- Update `progress.md` to reflect the new evaluation capabilities and generated results.
- Update `systemPatterns.md` to describe the new evaluation script and workflow.
- Review the generated plots and results in `results/multi_rollout_results/`.
- Consider adding the missing `_best_dev_l_50` model file if desired for complete evaluation.
- Continue with original project goals (code reorganization, requirements, README, GitHub push).

**Active Decisions and Considerations:**
- The model `models/noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_50` was specified in the evaluation script but the file was not found. Evaluation proceeded with the other two models.
