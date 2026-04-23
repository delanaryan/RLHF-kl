"""
Test/Demo Script: Quick verification that experiments work
Run this to validate that all components are functional before running full experiments
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from fixed_beta_sweep import FixedBetaSweepExperiment
        print("✓ FixedBetaSweepExperiment imported")
    except Exception as e:
        print(f"✗ Failed to import FixedBetaSweepExperiment: {e}")
        return False
    
    try:
        from adaptive_beta_controller import AdaptiveKLController, AdaptiveOptimizationExperiment
        print("✓ AdaptiveKLController & AdaptiveOptimizationExperiment imported")
    except Exception as e:
        print(f"✗ Failed to import adaptive controller: {e}")
        return False
    
    try:
        from comparison_utils import ExperimentRunner, ResultsComparator
        print("✓ ExperimentRunner & ResultsComparator imported")
    except Exception as e:
        print(f"✗ Failed to import comparison utilities: {e}")
        return False
    try:
        from src import generate, score, utils
        print("✓ Source modules (generate, score, utils) imported")
    except Exception as e:
        print(f"✗ Failed to import source modules: {e}")
        return False
    
    return True


def test_data_loading():
    """Test that data files can be loaded"""
    print("\nTesting data loading...")
    try:
        import config
        from src import utils
        
        prompts = utils.csvToArr(config.PROMPT_PATH)
        print(f"✓ Loaded {len(prompts)-1} prompts from {config.PROMPT_PATH}")
        
        if len(prompts) > 1:
            print(f"  Sample prompt: {prompts[1][1][:50]}...")
        else:
            print("✗ No prompts found!")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False


def test_core_functions():
    """Test that core functions work"""
    print("\nTesting core functions...")
    
    try:
        from fixed_beta_sweep import FixedBetaSweepExperiment
        from src import score
        exp = FixedBetaSweepExperiment()
        
        # Test KL calculation
        test_response = "This is a test response with some variety"
        kl = score.calculate_kl_divergence(test_response, [])
        print(f"✓ KL divergence calculation works: {kl:.4f}")
        
        # Test reward calculation
        reward = exp.compute_rlhf_reward(0.8, 0.3, 0.5)
        print(f"✓ RLHF reward calculation works: {reward:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Core function test failed: {e}")
        return False


def test_adaptive_controller():
    """Test adaptive controller logic"""
    print("\nTesting adaptive controller...")
    
    try:
        from adaptive_beta_controller import AdaptiveKLController, AdaptiveControllerConfig
        
        config = AdaptiveControllerConfig()
        controller = AdaptiveKLController(config)
        
        # Test β adjustment with high KL
        initial_beta = controller.beta
        new_beta, action = controller.adjust_beta(0.8)  # Above upper threshold
        
        if action == 'increase' and new_beta > initial_beta:
            print(f"✓ β increase works: {initial_beta:.4f} → {new_beta:.4f} (action: {action})")
        else:
            print(f"✗ β increase failed: {initial_beta:.4f} → {new_beta:.4f} (action: {action})")
            return False
        
        # Reset and test β adjustment with low KL
        controller = AdaptiveKLController(config)
        initial_beta = controller.beta
        new_beta, action = controller.adjust_beta(0.1)  # Below lower threshold
        
        if action == 'decrease' and new_beta < initial_beta:
            print(f"✓ β decrease works: {initial_beta:.4f} → {new_beta:.4f} (action: {action})")
        else:
            print(f"✗ β decrease failed: {initial_beta:.4f} → {new_beta:.4f} (action: {action})")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Adaptive controller test failed: {e}")
        return False


def test_data_generation():
    """Test data generation and model calls"""
    print("\nTesting generation and scoring...")
    
    try:
        from src import generate, score
        
        test_prompt = "Say something positive."
        
        # Test generation
        response = generate.generateSingleResponse(test_prompt)
        if response and "Error" not in response:
            print(f"✓ Generation works")
            print(f"  Response preview: {response[:80]}...")
        else:
            print(f"✗ Generation failed: {response}")
            return False
        
        # Test sentiment scoring
        sentiment = score.getSentimentScore(response)
        if 0 <= sentiment <= 1:
            print(f"✓ Sentiment scoring works: {sentiment:.4f}")
        else:
            print(f"✗ Sentiment score invalid: {sentiment}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Generation/scoring test failed: {e}")
        return False


def run_mini_experiment():
    """Run a tiny experiment to verify end-to-end flow"""
    print("\nRunning mini fixed-β experiment (1 prompt, 2 generations, 1 β value)...")
    
    try:
        from fixed_beta_sweep import FixedBetaSweepExperiment
        
        exp = FixedBetaSweepExperiment(output_dir="experiments/results/test_mini")
        exp.beta_values = [0.1]  # Test with single β
        exp.run_sweep(num_prompts=1, generations_per_prompt=2)
        
        print("✓ Mini experiment completed!")
        return True
    except Exception as e:
        print(f"✗ Mini experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("KL DIVERGENCE EXPERIMENTS - VALIDATION SUITE")
    print("="*70)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Loading", test_data_loading),
        ("Core Functions", test_core_functions),
        ("Adaptive Controller", test_adaptive_controller),
        ("Generation & Scoring", test_data_generation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready to run experiments!")
        print("\nNext steps:")
        print("1. Run fixed-β sweep: python fixed_beta_sweep.py")
        print("2. Run adaptive controller: python adaptive_beta_controller.py")
        print("3. Compare results: python comparison_utils.py")
        print("4. Visualize results: python visualize_results.py")
    else:
        print("✗ SOME TESTS FAILED - Fix issues before running experiments")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
