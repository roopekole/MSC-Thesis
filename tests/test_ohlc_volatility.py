"""
OHLC Volatility Analysis Unit Tests

Focused unit tests for the OHLC heuristic computation methods.
Tests are designed to be imported and executed within the notebook environment.
"""

from typing import Dict, Tuple

# Global variable to hold OHLCAnalyzer class reference from notebook
_OHLCAnalyzer = None

def set_ohlc_analyzer(analyzer_class):
    """Set the OHLCAnalyzer class reference from the notebook"""
    global _OHLCAnalyzer
    _OHLCAnalyzer = analyzer_class

def _get_analyzer():
    """Get the OHLCAnalyzer with error checking"""
    if _OHLCAnalyzer is None:
        raise ImportError("OHLCAnalyzer not set. Call set_ohlc_analyzer() first.")
    return _OHLCAnalyzer

def test_ohlc_analyzer_structure():
    """Test that OHLCAnalyzer methods return expected structure"""
    analyzer = _get_analyzer()
    
    # Test data
    predicted_ohlc = {"O": 100.0, "H": 105.0, "L": 99.0, "C": 104.0}
    current_ohlc = {"O": 95.0, "H": 100.0, "L": 94.0, "C": 99.0}
    
    result = analyzer.analyze_ohlc_patterns(predicted_ohlc, current_ohlc)
    
    # Test structure
    expected_keys = [
        "range_significance", "bullish_alignment", "bearish_alignment",
        "long_protection", "short_protection", "volatility_quality", 
        "direction_clarity", "body_pct", "upper_wick_pct", "lower_wick_pct"
    ]
    
    assert isinstance(result, dict), "Result should be a dictionary"
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], (int, float)), f"Key {key} should be numeric"
    
    return "PASS: test_ohlc_analyzer_structure passed"


def test_ohlc_zero_volatility():
    """Test OHLC analysis with zero volatility (flat prices)"""
    analyzer = _get_analyzer()
    
    zero_vol_ohlc = {"O": 100.0, "H": 100.0, "L": 100.0, "C": 100.0}
    current_ohlc = {"O": 100.0, "H": 100.0, "L": 100.0, "C": 100.0}
    
    result = analyzer.analyze_ohlc_patterns(zero_vol_ohlc, current_ohlc)
    
    # Zero range should produce predictable results
    assert result["range_significance"] == 0.0, "Zero volatility should have zero range significance"
    assert result["body_pct"] == 0.33, "Zero range should default body_pct to 0.33"
    assert result["upper_wick_pct"] == 0.33, "Zero range should default upper_wick_pct to 0.33"
    assert result["lower_wick_pct"] == 0.33, "Zero range should default lower_wick_pct to 0.33"
    
    return "PASS: test_ohlc_zero_volatility passed"


def test_ohlc_high_volatility():
    """Test OHLC analysis with high volatility"""
    analyzer = _get_analyzer()
    
    high_vol_ohlc = {"O": 100.0, "H": 110.0, "L": 90.0, "C": 105.0}
    current_ohlc = {"O": 100.0, "H": 103.0, "L": 97.0, "C": 102.0}
    
    result = analyzer.analyze_ohlc_patterns(high_vol_ohlc, current_ohlc)
    
    # High volatility should produce high range significance
    assert result["range_significance"] > 0.5, "High volatility should have significant range"
    assert result["range_significance"] <= 1.0, "Range significance should not exceed 1.0"
    
    return "PASS: test_ohlc_high_volatility passed"


def test_ohlc_bounds_checking():
    """Test that all OHLC metrics stay within expected bounds"""
    analyzer = _get_analyzer()
    
    test_cases = [
        {"O": 100.0, "H": 105.0, "L": 99.0, "C": 104.0},  # Bullish
        {"O": 100.0, "H": 101.0, "L": 95.0, "C": 96.0},   # Bearish
        {"O": 100.0, "H": 102.0, "L": 98.0, "C": 100.0},  # Doji
        {"O": 100.0, "H": 110.0, "L": 90.0, "C": 105.0}   # High volatility
    ]
    
    current_ohlc = {"O": 100.0, "H": 102.0, "L": 98.0, "C": 101.0}
    
    for i, predicted_ohlc in enumerate(test_cases):
        result = analyzer.analyze_ohlc_patterns(predicted_ohlc, current_ohlc)
        
        # All metrics should be within [0, 1] bounds
        assert 0 <= result["range_significance"] <= 1, f"Range significance out of bounds in case {i+1}"
        assert 0 <= result["bullish_alignment"] <= 1, f"Bullish alignment out of bounds in case {i+1}"
        assert 0 <= result["bearish_alignment"] <= 1, f"Bearish alignment out of bounds in case {i+1}"
        assert 0 <= result["long_protection"] <= 1, f"Long protection out of bounds in case {i+1}"
        assert 0 <= result["short_protection"] <= 1, f"Short protection out of bounds in case {i+1}"
        assert 0 <= result["body_pct"] <= 1, f"Body percentage out of bounds in case {i+1}"
        assert 0 <= result["upper_wick_pct"] <= 1, f"Upper wick percentage out of bounds in case {i+1}"
        assert 0 <= result["lower_wick_pct"] <= 1, f"Lower wick percentage out of bounds in case {i+1}"
    
    return "PASS: test_ohlc_bounds_checking passed"


def test_ohlc_mathematical_consistency():
    """Test mathematical consistency of OHLC calculations"""
    analyzer = _get_analyzer()
    
    predicted_ohlc = {"O": 100.0, "H": 105.0, "L": 99.0, "C": 104.0}
    current_ohlc = {"O": 95.0, "H": 100.0, "L": 94.0, "C": 99.0}
    
    result = analyzer.analyze_ohlc_patterns(predicted_ohlc, current_ohlc)
    
    # Direction clarity = |bullish_alignment - bearish_alignment|
    expected_clarity = abs(result["bullish_alignment"] - result["bearish_alignment"])
    assert abs(result["direction_clarity"] - expected_clarity) < 1e-10, "Direction clarity calculation inconsistent"
    
    # Volatility quality = range_significance * direction_clarity
    expected_quality = result["range_significance"] * result["direction_clarity"]
    assert abs(result["volatility_quality"] - expected_quality) < 1e-10, "Volatility quality calculation inconsistent"
    
    return "PASS: test_ohlc_mathematical_consistency passed"


def test_signal_modifier_structure():
    """Test signal modifier method returns proper structure"""
    analyzer = _get_analyzer()
    
    metrics = {
        "range_significance": 0.5,
        "bullish_alignment": 0.7,
        "bearish_alignment": 0.3,
        "long_protection": 0.6,
        "short_protection": 0.4,
        "volatility_quality": 0.2,
        "direction_clarity": 0.4,
        "body_pct": 0.6,
        "upper_wick_pct": 0.2,
        "lower_wick_pct": 0.2
    }
    
    modifier, explanation = analyzer.calculate_ohlc_signal_modifier(metrics, "BUY")
    
    assert isinstance(modifier, (int, float)), "Modifier should be numeric"
    assert isinstance(explanation, str), "Explanation should be string"
    assert explanation.startswith("OHLC:"), "Explanation should start with 'OHLC:'"
    
    return "PASS: test_signal_modifier_structure passed"


def test_signal_modifier_buy_signals():
    """Test signal modification for BUY signals"""
    analyzer = _get_analyzer()
    
    strong_bullish_metrics = {
        "range_significance": 0.8,  # High volatility
        "bullish_alignment": 0.9,   # Strong bullish
        "bearish_alignment": 0.1,   # Weak bearish
        "long_protection": 0.8,     # Good downside protection
        "short_protection": 0.2,
        "volatility_quality": 0.64,
        "direction_clarity": 0.8,
        "body_pct": 0.7,
        "upper_wick_pct": 0.1,
        "lower_wick_pct": 0.2
    }
    
    for signal in ["BUY", "STRONG_BUY"]:
        modifier, explanation = analyzer.calculate_ohlc_signal_modifier(strong_bullish_metrics, signal)
        assert modifier > 0, f"Strong bullish metrics should boost {signal} signal"
        assert "high volatility" in explanation, "Should mention high volatility"
    
    return "PASS: test_signal_modifier_buy_signals passed"


def test_signal_modifier_bounds():
    """Test that signal modifiers stay within reasonable bounds"""
    analyzer = _get_analyzer()
    
    extreme_metrics = {
        "range_significance": 1.0,
        "bullish_alignment": 1.0,
        "bearish_alignment": 0.0,
        "long_protection": 1.0,
        "short_protection": 0.0,
        "volatility_quality": 1.0,
        "direction_clarity": 1.0,
        "body_pct": 1.0,
        "upper_wick_pct": 0.0,
        "lower_wick_pct": 0.0
    }
    
    modifier, _ = analyzer.calculate_ohlc_signal_modifier(extreme_metrics, "STRONG_BUY")
    
    # Maximum possible: 0.15 (range) + 0.10 (directional) + 0.025 (risk) = 0.275
    assert modifier <= 0.3, "Signal modifier should not exceed reasonable bounds"
    assert modifier >= 0, "Extreme positive metrics should give positive modifier"
    
    return "PASS: test_signal_modifier_bounds passed"


def run_all_ohlc_tests():
    """Run all OHLC volatility tests and return results"""
    if _OHLCAnalyzer is None:
        return {
            "passed": 0,
            "failed": 1,
            "total": 1, 
            "results": ["ERROR: OHLCAnalyzer not set. Call set_ohlc_analyzer(OHLCAnalyzer) first."],
            "success": False
        }
    
    test_functions = [
        test_ohlc_analyzer_structure,
        test_ohlc_zero_volatility, 
        test_ohlc_high_volatility,
        test_ohlc_bounds_checking,
        test_ohlc_mathematical_consistency,
        test_signal_modifier_structure,
        test_signal_modifier_buy_signals,
        test_signal_modifier_bounds
    ]
    
    results = []
    passed = 0
    failed = 0
    
    print("Running OHLC Volatility Analysis Tests")
    print("=" * 50)
    
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
            print(result)
            passed += 1
        except Exception as e:
            error_msg = f"ERROR: {test_func.__name__} failed: {str(e)}"
            results.append(error_msg)
            print(error_msg)
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return {
        "passed": passed,
        "failed": failed, 
        "total": len(test_functions),
        "results": results,
        "success": failed == 0
    }


if __name__ == "__main__":
    # This allows running the tests directly if needed
    run_all_ohlc_tests()
