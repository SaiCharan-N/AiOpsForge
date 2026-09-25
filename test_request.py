#!/usr/bin/env python3
"""Test the AIOpsForge pipeline with a simple request."""
import requests
import json
import time

BACKEND_URL = "http://localhost:8000"

def test_pipeline():
    request_text = "write a function that checks if a number is prime, with tests"
    
    print(f"[TEST] Submitting request: '{request_text}'")
    print("[TEST] This may take 2-5 minutes on CPU...")
    print()
    
    start = time.time()
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/request",
            json={"request": request_text},
            timeout=600  # 10 minutes
        )
        response.raise_for_status()
        result = response.json()
        
        elapsed = time.time() - start
        
        print("=" * 70)
        print(f"✓ PIPELINE COMPLETED in {elapsed:.1f} seconds")
        print("=" * 70)
        print(f"\nProject ID: {result['project_id']}")
        print(f"Status: {'PASSED ✓' if result['done'] else 'NEEDS REVIEW'}")
        print(f"Tasks completed: {len(result['task_history'])}")
        
        print("\nTask History:")
        for i, task in enumerate(result['task_history'], 1):
            status = "✓ PASSED" if task['passed'] else "✗ FAILED"
            print(f"  {i}. {task['task'][:60]}...")
            print(f"     Status: {status} | Attempts: {task['attempts']}")
            if task.get('filename'):
                print(f"     File: {task['filename']}")
        
        print("\n" + "=" * 70)
        return True
        
    except requests.exceptions.Timeout:
        print("✗ TIMEOUT: Pipeline took longer than 10 minutes")
        return False
    except requests.exceptions.RequestException as e:
        print(f"✗ REQUEST ERROR: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)
