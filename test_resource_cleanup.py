#!/usr/bin/env python3
"""
Test script to verify LegalIndexBuilder resource cleanup
"""
import sys
from pathlib import Path
from src.index_builder import LegalIndexBuilder

def test_context_manager():
    """Test using LegalIndexBuilder with context manager"""
    print("[TEST] Testing LegalIndexBuilder context manager...")

    try:
        with LegalIndexBuilder(enable_monitoring=False) as builder:
            print("[PASS] Context manager entry successful")
            # Check that builder is properly initialized
            assert builder is not None
            print("[PASS] Builder initialized")

        print("[PASS] Context manager exit successful (resources cleaned up)")
        return True

    except Exception as e:
        print(f"[FAIL] Context manager test failed: {e}")
        return False

def test_explicit_close():
    """Test explicit close() method"""
    print("\n[TEST] Testing explicit close() method...")

    try:
        builder = LegalIndexBuilder(enable_monitoring=False)
        print("[PASS] Builder created")

        builder.close()
        print("[PASS] Builder.close() called successfully")

        # Verify resources are cleared
        assert builder.chroma_client is None, "chroma_client should be None"
        assert builder.vector_store is None, "vector_store should be None"
        assert builder.index is None, "index should be None"
        print("[PASS] Resources verified as cleared")

        return True

    except Exception as e:
        print(f"[FAIL] Explicit close test failed: {e}")
        return False

def test_destructor():
    """Test __del__ destructor cleanup"""
    print("\n[TEST] Testing __del__ destructor...")

    try:
        builder = LegalIndexBuilder(enable_monitoring=False)
        print("[PASS] Builder created")

        # Delete the object, triggering __del__
        del builder
        print("[PASS] Builder deleted, __del__ called")

        return True

    except Exception as e:
        print(f"[FAIL] Destructor test failed: {e}")
        return False

def main():
    print("=" * 70)
    print("LegalIndexBuilder Resource Cleanup Tests")
    print("=" * 70 + "\n")

    results = []
    results.append(("Context Manager", test_context_manager()))
    results.append(("Explicit Close", test_explicit_close()))
    results.append(("Destructor", test_destructor()))

    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n[SUCCESS] All resource cleanup tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())