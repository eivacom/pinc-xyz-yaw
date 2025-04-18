# Computational Time Comparison: Full Trajectory Duration

Comparison performed over the full duration of the first trajectory from dataset: `test_set_line`
Trajectory Duration: 5.20 seconds (65 steps)
Using device: `cuda`

## Description of Tests:
- **PINN Model Time:** Measures the total wall-clock time to perform sequential forward passes of the neural network for each step in the trajectory duration. Input for each step uses the ground truth state from the previous step.
- **Simulation Time:** Measures the total wall-clock time to perform a closed-loop simulation using RK4 integration for the entire trajectory duration.

## Model Descriptions (Based on Filenames):

All tested PINN models share the prefix `noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2` and end with `_best_dev_l_XXX`, suggesting the following characteristics:
*   **`noisy_input`**: Trained with noisy input data augmentation.
*   **`rotated`**: Possibly related to data augmentation involving rotation or the coordinate system used.
*   **`not_config`**: Trained *without* the ConFIG gradient combination technique (using manual gradient scaling/summation instead).
*   **`yaw_interval_increased`**: Suggests a specific setting related to the yaw state or its interval during training/data generation.
*   **`dev_data_fixed_2`**: Might refer to a specific version or configuration of the development dataset used.
*   **`_best_dev_l_XXX`**: Indicates the model state saved at epoch `XXX` which achieved the best loss on the development set up to that point.

*(Note: One file, `noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2`, was skipped due to an incompatible save format.)*

## Results:
| Method                                                                             | Total Time (seconds) |
|------------------------------------------------------------------------------------|----------------------|
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_15 | 0.0979             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_4 | 0.1055             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_13 | 0.0713             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_52 | 0.0855             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_125 | 0.1033             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_257 | 0.0918             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_1 | 0.0775             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_74 | 0.0873             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_77 | 0.0996             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_58 | 0.0642             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_75 | 0.0906             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_5 | 0.0728             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_59 | 0.0703             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_940 | 0.0939             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_189 | 0.0719             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_63 | 0.1162             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_11 | 0.1390             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_937 | 0.0666             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_3 | 0.0813             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_969 | 0.0784             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_206 | 0.0785             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_226 | 0.0936             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_57 | 0.0676             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_313 | 0.0958             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_817 | 0.1025             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_81 | 0.0869             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_72 | 0.0943             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_49 | 0.0828             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_68 | 0.0660             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_993 | 0.0827             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_2 | 0.1100             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_0 | 0.0603             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_329 | 0.1264             |
| PINN: noisy_input_rotated_not_config_yaw_interval_increased_dev_data_fixed_2_best_dev_l_62 | 0.0692             |
| Simulation (RK4 Rollout)                                                           | 0.0017             |
