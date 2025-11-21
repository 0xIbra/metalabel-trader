import time
import numpy as np
import pandas as pd
from src.quant_engine.main import QuantEngine
from src.oracle.main import Oracle
from src.shared.datatypes import TickData

def benchmark_quant_engine(iterations=1000):
    """Benchmark QuantEngine processing time"""
    print("=" * 60)
    print("BENCHMARKING QUANT ENGINE")
    print("=" * 60)

    # Load real data
    df = pd.read_csv("data/raw/eurusd_m1.csv")
    df.columns = [c.lower() for c in df.columns]

    engine = QuantEngine(window_size=20)

    # Warm up (first 20 ticks to fill window)
    for idx in range(20):
        row = df.iloc[idx]
        tick = TickData(
            symbol="EURUSD",
            bid=row['close'],
            ask=row['close'] + 0.0001,
            timestamp=float(row['timestamp']),
            volume=float(row['volume'])
        )
        engine.on_tick(tick)

    # Actual benchmark
    latencies = []
    for idx in range(20, min(20 + iterations, len(df))):
        row = df.iloc[idx]
        tick = TickData(
            symbol="EURUSD",
            bid=row['close'],
            ask=row['close'] + 0.0001,
            timestamp=float(row['timestamp']),
            volume=float(row['volume'])
        )

        start = time.perf_counter()
        features = engine.on_tick(tick)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # Stats
    latencies = np.array(latencies)
    print(f"\nIterations: {len(latencies)}")
    print(f"Mean Latency: {latencies.mean():.3f} ms")
    print(f"Median Latency: {np.median(latencies):.3f} ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.3f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.3f} ms")
    print(f"Max Latency: {latencies.max():.3f} ms")
    print(f"Min Latency: {latencies.min():.3f} ms")

    # Check requirement
    requirement = 50.0  # ms
    if latencies.mean() < requirement:
        print(f"\n✅ PASS: Mean latency ({latencies.mean():.3f}ms) < {requirement}ms")
    else:
        print(f"\n❌ FAIL: Mean latency ({latencies.mean():.3f}ms) >= {requirement}ms")

    return latencies

def benchmark_oracle(iterations=1000):
    """Benchmark Oracle inference time"""
    print("\n" + "=" * 60)
    print("BENCHMARKING ORACLE (AI INFERENCE)")
    print("=" * 60)

    oracle = Oracle()

    # Sample feature vector
    features = {
        "z_score": 1.5,
        "rsi": 65.0,
        "volatility": 0.002,
        "adx": 25.0,
        "time_sin": 0.5,
        "volume_delta": 0.1
    }

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        decision = oracle.predict(features)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # Stats
    latencies = np.array(latencies)
    print(f"\nIterations: {len(latencies)}")
    print(f"Mean Latency: {latencies.mean():.3f} ms")
    print(f"Median Latency: {np.median(latencies):.3f} ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.3f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.3f} ms")
    print(f"Max Latency: {latencies.max():.3f} ms")

    return latencies

def benchmark_end_to_end(iterations=500):
    """Benchmark full pipeline: Tick -> Features -> Inference -> Decision"""
    print("\n" + "=" * 60)
    print("BENCHMARKING END-TO-END PIPELINE")
    print("=" * 60)

    # Load real data
    df = pd.read_csv("data/raw/eurusd_m1.csv")
    df.columns = [c.lower() for c in df.columns]

    engine = QuantEngine(window_size=20)
    oracle = Oracle()

    # Warm up
    for idx in range(20):
        row = df.iloc[idx]
        tick = TickData(
            symbol="EURUSD",
            bid=row['close'],
            ask=row['close'] + 0.0001,
            timestamp=float(row['timestamp']),
            volume=float(row['volume'])
        )
        engine.on_tick(tick)

    # Benchmark
    latencies = []
    for idx in range(20, min(20 + iterations, len(df))):
        row = df.iloc[idx]
        tick = TickData(
            symbol="EURUSD",
            bid=row['close'],
            ask=row['close'] + 0.0001,
            timestamp=float(row['timestamp']),
            volume=float(row['volume'])
        )

        start = time.perf_counter()
        features = engine.on_tick(tick)
        decision = oracle.predict(features) if features else None
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # Stats
    latencies = np.array(latencies)
    print(f"\nIterations: {len(latencies)}")
    print(f"Mean Latency: {latencies.mean():.3f} ms")
    print(f"Median Latency: {np.median(latencies):.3f} ms")
    print(f"P95 Latency: {np.percentile(latencies, 95):.3f} ms")
    print(f"P99 Latency: {np.percentile(latencies, 99):.3f} ms")
    print(f"Max Latency: {latencies.max():.3f} ms")

    # Check total requirement (Quant < 50ms is the explicit one)
    print(f"\n✅ Full pipeline latency: {latencies.mean():.3f} ms")

    return latencies

if __name__ == "__main__":
    print("\n🚀 METALABEL TRADER PERFORMANCE BENCHMARK")
    print("Requirements: Quant Engine < 50ms (Implementation.md)\n")

    # Run benchmarks
    quant_latencies = benchmark_quant_engine(iterations=1000)
    oracle_latencies = benchmark_oracle(iterations=1000)
    e2e_latencies = benchmark_end_to_end(iterations=500)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Quant Engine (mean): {quant_latencies.mean():.3f} ms")
    print(f"Oracle Inference (mean): {oracle_latencies.mean():.3f} ms")
    print(f"End-to-End (mean): {e2e_latencies.mean():.3f} ms")
    print("\nREQUIREMENT: Quant Engine < 50ms")
    if quant_latencies.mean() < 50:
        print("✅ PERFORMANCE REQUIREMENT MET")
    else:
        print("❌ PERFORMANCE REQUIREMENT NOT MET")
