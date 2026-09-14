import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 50)
print("WEATHER CLUSTERING ENVIRONMENT TEST")
print("=" * 50)

print(f"Python version : {sys.version}")
print(f"Python path    : {sys.executable}")

print("\nTesting packages...")

import requests
print("✓ requests")

import pandas
print("✓ pandas")

import pyarrow
print("✓ pyarrow")

import duckdb
print("✓ duckdb")

print("\nALL BASE COMPONENTS PASSED")
