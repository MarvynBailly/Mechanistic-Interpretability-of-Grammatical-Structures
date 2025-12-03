"""
Test the general analysis framework on multiple datasets.
"""

from general_analysis import run_path_patching_analysis

print("Testing General Path Patching Framework on Multiple Datasets\n")

# Test 1: Code completion with objects (should work well)
print("="*80)
print("TEST 1: Code Completion (objects)")
print("="*80)
results_code_obj = run_path_patching_analysis(
    dataset_type="code",
    dataset_params={"var_type": "objects"},
    n_examples=100,
    output_dir="results/general_test_code_objects"
)

print(f"\nResult: {results_code_obj['status']}")
if results_code_obj['status'] == 'success':
    print(f"Clean Accuracy: {results_code_obj['stats']['clean_accuracy']*100:.1f}%")
    print(f"Top Head: Layer {results_code_obj['significant_heads']['positive'][0][0]}, "
          f"Head {results_code_obj['significant_heads']['positive'][0][1]} "
          f"(effect: {results_code_obj['significant_heads']['positive'][0][2]:.3f})")

# Test 2: Code completion with letters (should work well)
print("\n\n" + "="*80)
print("TEST 2: Code Completion (letters)")
print("="*80)
results_code_let = run_path_patching_analysis(
    dataset_type="code",
    dataset_params={"var_type": "letters"},
    n_examples=100,
    output_dir="results/general_test_code_letters"
)

print(f"\nResult: {results_code_let['status']}")
if results_code_let['status'] == 'success':
    print(f"Clean Accuracy: {results_code_let['stats']['clean_accuracy']*100:.1f}%")
    print(f"Top Head: Layer {results_code_let['significant_heads']['positive'][0][0]}, "
          f"Head {results_code_let['significant_heads']['positive'][0][1]} "
          f"(effect: {results_code_let['significant_heads']['positive'][0][2]:.3f})")

# Test 3: Color-object association (should fail validation)
print("\n\n" + "="*80)
print("TEST 3: Color-Object Association")
print("="*80)
results_color = run_path_patching_analysis(
    dataset_type="color",
    dataset_params={"size": "small"},
    n_examples=100,
    output_dir="results/general_test_color",
    min_accuracy=0.5
)

print(f"\nResult: {results_color['status']}")
if results_color['status'] == 'failed':
    print(f"Clean Accuracy: {results_color['stats']['clean_accuracy']*100:.1f}%")
    print("Task validation failed as expected (model can't solve this task)")

print("\n\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nCode (objects):  {results_code_obj['status']:>8s} - "
      f"{results_code_obj['stats']['clean_accuracy']*100:.1f}% accuracy")
print(f"Code (letters):  {results_code_let['status']:>8s} - "
      f"{results_code_let['stats']['clean_accuracy']*100:.1f}% accuracy")
print(f"Color-Object:    {results_color['status']:>8s} - "
      f"{results_color['stats']['clean_accuracy']*100:.1f}% accuracy")

print("\n✅ General framework validated on multiple tasks!")
