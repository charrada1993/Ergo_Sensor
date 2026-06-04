"""
test_report_full.py
===================
Comprehensive test suite for report_generator.py.
Run with: python test_report_full.py
"""
import os
import sys
import glob
import traceback
import pandas as pd
from config import Config
from report_generator import ReportGenerator

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

results = []

def run_test(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    try:
        pdf = fn()
        size_kb = os.path.getsize(pdf) / 1024
        # Validate PDF structure with PyPDF2 if available
        page_info = ""
        try:
            import PyPDF2
            with open(pdf, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                page_info = f" | {len(reader.pages)} pages"
        except ImportError:
            pass
        print(f"  -> {PASS} | {os.path.basename(pdf)} | {size_kb:.1f} KB{page_info}")
        results.append((name, True, None))
        return pdf
    except Exception as e:
        print(f"  -> {FAIL} | {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))
        return None


# ─── Shared config & generator ──────────────────────────────────────────────
config = Config()

CSV_FULL    = r"csv_data\session_20260506_020211.csv"
CSV_SMALL   = r"csv_data\session_20260423_144217.csv"   # 0 rows (header only)
CSV_MEDIUM  = r"csv_data\session_20260506_002736.csv"   # ~1400 rows

TMP_FAST    = r"csv_data\tmp_test_200.csv"
TMP_NOSCORES = r"csv_data\tmp_noscores.csv"
TMP_NOTS    = r"csv_data\tmp_nots.csv"
TMP_ANGLEONLY = r"csv_data\tmp_angleonly.csv"


def make_generator():
    """Fresh generator per test (reload AI models each time)."""
    return ReportGenerator(config)


# ─── Test 1: Fast run — 200 rows with full AI pipeline ──────────────────────
def test_fast_200():
    df = pd.read_csv(CSV_FULL).head(200)
    df.to_csv(TMP_FAST, index=False)
    return make_generator().generate(TMP_FAST)

# ─── Test 2: Full CSV (all rows) ────────────────────────────────────────────
def test_full_csv():
    return make_generator().generate(CSV_FULL)

# ─── Test 3: Tiny / empty CSV (<60 rows → AI disabled gracefully) ────────────
def test_tiny_csv():
    df = pd.read_csv(CSV_FULL).head(30)
    tmp = r"csv_data\tmp_tiny30.csv"
    df.to_csv(tmp, index=False)
    return make_generator().generate(tmp)

# ─── Test 4: Header-only CSV (0 data rows) ──────────────────────────────────
def test_empty_csv():
    return make_generator().generate(CSV_SMALL)

# ─── Test 5: Missing RULA/REBA columns ─────────────────────────────────────
def test_missing_scores():
    df = pd.read_csv(CSV_FULL).head(200)
    drop_cols = [c for c in ['RULA_R_Final','RULA_L_Final','REBA_R_Final','REBA_L_Final']
                 if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    df.to_csv(TMP_NOSCORES, index=False)
    return make_generator().generate(TMP_NOSCORES)

# ─── Test 6: No Timestamp column ────────────────────────────────────────────
def test_no_timestamp():
    df = pd.read_csv(CSV_FULL).head(200)
    df = df.drop(columns=['Timestamp'], errors='ignore')
    df.to_csv(TMP_NOTS, index=False)
    return make_generator().generate(TMP_NOTS)

# ─── Test 7: Angle columns only (no scores, no timestamp) ───────────────────
def test_angle_only():
    df = pd.read_csv(CSV_FULL).head(200)
    keep = [c for c in df.columns if 'deg' in c or 'Pitch' in c or 'Roll' in c or 'Yaw' in c]
    df = df[keep]
    df.to_csv(TMP_ANGLEONLY, index=False)
    return make_generator().generate(TMP_ANGLEONLY)

# ─── Test 8: Medium session (~1400 rows) ────────────────────────────────────
def test_medium_csv():
    if not os.path.exists(CSV_MEDIUM):
        print(f"  -> {SKIP} — {CSV_MEDIUM} not found")
        results.append(("Medium CSV", None, "skipped"))
        return None
    return make_generator().generate(CSV_MEDIUM)


# ─── Run all tests ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ERGO SENSOR — Report Generator Test Suite")
    print("="*60)

    run_test("Test 1: Fast 200-row with AI", test_fast_200)
    run_test("Test 2: Full CSV with AI",     test_full_csv)
    run_test("Test 3: Tiny session (30 rows)", test_tiny_csv)
    run_test("Test 4: Empty CSV (0 rows)",   test_empty_csv)
    run_test("Test 5: Missing RULA/REBA",    test_missing_scores)
    run_test("Test 6: No Timestamp column",  test_no_timestamp)
    run_test("Test 7: Angle columns only",   test_angle_only)
    run_test("Test 8: Medium session",       test_medium_csv)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    passed = sum(1 for _, ok, _ in results if ok is True)
    failed = sum(1 for _, ok, _ in results if ok is False)
    skipped = sum(1 for _, ok, _ in results if ok is None)
    for name, ok, err in results:
        status = PASS if ok else (SKIP if ok is None else FAIL)
        extra = f" — {err}" if err and ok is False else ""
        print(f"  [{status}] {name}{extra}")
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60)

    # Cleanup temp files
    for tmp in [TMP_FAST, TMP_NOSCORES, TMP_NOTS, TMP_ANGLEONLY,
                r"csv_data\tmp_tiny30.csv"]:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    sys.exit(0 if failed == 0 else 1)
