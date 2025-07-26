import os
import pandas as pd

# Ensure output directory exists
output_dir = "computer_costs"
os.makedirs(output_dir, exist_ok=True)

# Constants
TFLOPS = 19.5 * 1e12  # FP32 throughput on A100
TFLOPS_INT8 = 312 * 1e12  # INT8 throughput on A100 (NVIDIA A100: up to 312 TOPS for INT8)
TFLOPS_FLOAT8 = 280 * 1e12  # Float8 throughput on A100 (estimated, similar to INT8 but slightly lower)
BYTES_PER_FLOAT = 4
BYTES_PER_INT8 = 1
BYTES_PER_FLOAT8 = 1  # Float8 uses 1 byte per element

# CORRECTED: Using actual latent dimension of 8 (not 16)
LATENT_DIM = 8
RAW_BYTES = 125 * BYTES_PER_FLOAT  # 500 bytes
LATENT_BYTES = LATENT_DIM * BYTES_PER_FLOAT  # 8 * 4 = 32 bytes (not 64)

# For quantized MLP SWAE only (not for Conv or RAW)
LATENT_BYTES_INT8 = LATENT_DIM * BYTES_PER_INT8  # 8 bytes (not 16)
LATENT_BYTES_FLOAT8 = LATENT_DIM * BYTES_PER_FLOAT8  # 8 bytes (not 16)

# Data assembly cost (assume simple memory copy, e.g., from host to device or device to device)
# For 125 floats (500 bytes), at typical GPU memory bandwidth (A100 HBM2e: ~1.6 TB/s)
GPU_MEM_BW = 1.6 * 1e12  # bytes/sec

def data_assembly_time(bytes_):
    # Returns time in microseconds
    return bytes_ / GPU_MEM_BW * 1e6

# Assembly cost for encoding (input: 125 floats) and decoding (output: 125 floats)
ENCODE_ASSEMBLY_US = data_assembly_time(RAW_BYTES)
DECODE_ASSEMBLY_US = data_assembly_time(RAW_BYTES)
# For quantized MLP SWAE, only latent is quantized
ENCODE_ASSEMBLY_US_INT8 = data_assembly_time(RAW_BYTES)  # input is still FP32
DECODE_ASSEMBLY_US_INT8 = data_assembly_time(RAW_BYTES)  # output is still FP32
ENCODE_ASSEMBLY_US_FLOAT8 = data_assembly_time(RAW_BYTES)
DECODE_ASSEMBLY_US_FLOAT8 = data_assembly_time(RAW_BYTES)

# FLOP calculations
# CONV Encoder:
# 8 conv layers and 1 FC (2048 -> 16)
# Total MACs = 28,671,776 -> FLOPs = 2 * MACs = 57,343,552
FLOPS_CONV_ENC = 57343552

# CONV Decoder:
# 1 FC (16 -> 2048) + 4 deconvs
# Total MACs = 12,451,072 -> FLOPs = 24,902,144
FLOPS_CONV_DEC = 24902144
FLOPS_CONV_TOTAL = FLOPS_CONV_ENC + FLOPS_CONV_DEC

# MLP Encoder:
# 125×512 + 512×256 + 256×16 = 64000 + 131072 + 4096 MACs = 199168 MACs -> 398336 FLOPs
# CORRECTED for 8 latent dims: 125×512 + 512×256 + 256×8 = 64000 + 131072 + 2048 = 197120 MACs -> 394240 FLOPs
FLOPS_MLP_ENC = 394240

# MLP Decoder:
# 8×256 + 256×512 + 512×125 = 2048 + 131072 + 64000 MACs = 197120 MACs -> 394240 FLOPs
FLOPS_MLP_DEC = 394240
FLOPS_MLP_TOTAL = FLOPS_MLP_ENC + FLOPS_MLP_DEC

# Compute time (in microseconds)
def compute_time(flop):
    return flop / TFLOPS * 1e6

def compute_time_int8(flop):
    return flop / TFLOPS_INT8 * 1e6

def compute_time_float8(flop):
    # IMPORTANT: In practice, Float8 speedup is limited by:
    # 1. Mixed precision (input/output remain FP32)
    # 2. Memory bandwidth bottlenecks
    # 3. Kernel launch overheads
    # Real-world speedup is typically 2-4x, not 14x
    # Using conservative 3x speedup over FP32
    return flop / (TFLOPS * 3) * 1e6  # Conservative real-world estimate

# Transfer time (in microseconds)
def tx_time(bytes_, bw_gbps):
    bw_bps = bw_gbps * 1e9 / 8  # convert GB/s to B/s
    return bytes_ / bw_bps * 1e6

# Interconnect bandwidths
links = {
    "NVLink 3 (100 GB/s)": 100,
    "NVLink 4 (200 GB/s)": 200,
    "PCIe 4x16 (32 GB/s)": 32,
    "IB 200 Gb (25 GB/s)": 25,
    "Eth 40 Gb (5 GB/s)": 5,
    "Eth 10 Gb (1.25 GB/s)": 1.25,
    "Eth 1 Gb (0.125 GB/s)": 0.125,
}

# Table 1: Arithmetic workload and compute time (add data assembly cost)
workload = pd.DataFrame({
    "Stage": ["Encode", "Decode", "Total"],
    "FLOPs (Conv SWAE)": [FLOPS_CONV_ENC, FLOPS_CONV_DEC, FLOPS_CONV_TOTAL],
    "Time (Conv SWAE) [µs]": [
        compute_time(FLOPS_CONV_ENC) + ENCODE_ASSEMBLY_US,
        compute_time(FLOPS_CONV_DEC) + DECODE_ASSEMBLY_US,
        compute_time(FLOPS_CONV_TOTAL) + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    ],
    "FLOPs (MLP SWAE)": [FLOPS_MLP_ENC, FLOPS_MLP_DEC, FLOPS_MLP_TOTAL],
    "Time (MLP SWAE) [µs]": [
        compute_time(FLOPS_MLP_ENC) + ENCODE_ASSEMBLY_US,
        compute_time(FLOPS_MLP_DEC) + DECODE_ASSEMBLY_US,
        compute_time(FLOPS_MLP_TOTAL) + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    ],
    "Data Assembly [µs]": [ENCODE_ASSEMBLY_US, DECODE_ASSEMBLY_US, ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US]
})

# Table 1b: Arithmetic workload and compute time for INT8 quantized MLP SWAE only
workload_int8 = pd.DataFrame({
    "Stage": ["Encode", "Decode", "Total"],
    "FLOPs (Conv SWAE)": [FLOPS_CONV_ENC, FLOPS_CONV_DEC, FLOPS_CONV_TOTAL],
    "Time (Conv SWAE) [µs]": [
        compute_time(FLOPS_CONV_ENC) + ENCODE_ASSEMBLY_US,
        compute_time(FLOPS_CONV_DEC) + DECODE_ASSEMBLY_US,
        compute_time(FLOPS_CONV_TOTAL) + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    ],
    "FLOPs (MLP SWAE INT8)": [FLOPS_MLP_ENC, FLOPS_MLP_DEC, FLOPS_MLP_TOTAL],
    "Time (MLP SWAE INT8) [µs]": [
        compute_time_int8(FLOPS_MLP_ENC) + ENCODE_ASSEMBLY_US_INT8,
        compute_time_int8(FLOPS_MLP_DEC) + DECODE_ASSEMBLY_US_INT8,
        compute_time_int8(FLOPS_MLP_TOTAL) + ENCODE_ASSEMBLY_US_INT8 + DECODE_ASSEMBLY_US_INT8
    ],
    "Data Assembly [µs]": [ENCODE_ASSEMBLY_US_INT8, DECODE_ASSEMBLY_US_INT8, ENCODE_ASSEMBLY_US_INT8 + DECODE_ASSEMBLY_US_INT8]
})

# Table 1c: Arithmetic workload and compute time for Float8 quantized MLP SWAE only
workload_float8 = pd.DataFrame({
    "Stage": ["Encode", "Decode", "Total"],
    "FLOPs (Conv SWAE)": [FLOPS_CONV_ENC, FLOPS_CONV_DEC, FLOPS_CONV_TOTAL],
    "Time (Conv SWAE) [µs]": [
        compute_time(FLOPS_CONV_ENC) + ENCODE_ASSEMBLY_US,
        compute_time(FLOPS_CONV_DEC) + DECODE_ASSEMBLY_US,
        compute_time(FLOPS_CONV_TOTAL) + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    ],
    "FLOPs (MLP SWAE FLOAT8)": [FLOPS_MLP_ENC, FLOPS_MLP_DEC, FLOPS_MLP_TOTAL],
    "Time (MLP SWAE FLOAT8) [µs]": [
        compute_time_float8(FLOPS_MLP_ENC) + ENCODE_ASSEMBLY_US_FLOAT8,
        compute_time_float8(FLOPS_MLP_DEC) + DECODE_ASSEMBLY_US_FLOAT8,
        compute_time_float8(FLOPS_MLP_TOTAL) + ENCODE_ASSEMBLY_US_FLOAT8 + DECODE_ASSEMBLY_US_FLOAT8
    ],
    "Data Assembly [µs]": [ENCODE_ASSEMBLY_US_FLOAT8, DECODE_ASSEMBLY_US_FLOAT8, ENCODE_ASSEMBLY_US_FLOAT8 + DECODE_ASSEMBLY_US_FLOAT8]
})

# Table 2: Transfer and total time (no change for FP32)
transfer_table = []
for name, bw in links.items():
    tx_raw = tx_time(RAW_BYTES, bw)
    tx_latent = tx_time(LATENT_BYTES, bw)
    total_conv = compute_time(FLOPS_CONV_TOTAL) + tx_latent + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    total_mlp = compute_time(FLOPS_MLP_TOTAL) + tx_latent + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    transfer_table.append([
        name, round(tx_raw, 4), round(tx_latent, 4), round(tx_raw, 4), round(total_conv, 4), round(total_mlp, 4)
    ])

transfer = pd.DataFrame(
    transfer_table,
    columns=["Link", "TX Raw [µs]", "TX Latent [µs]", "RAW Total [µs]", "Conv SWAE Total [µs]", "MLP SWAE Total [µs]"]
)

# Table 2b: Transfer and total time for INT8 quantized MLP SWAE only
transfer_table_int8 = []
for name, bw in links.items():
    tx_raw = tx_time(RAW_BYTES, bw)
    tx_latent = tx_time(LATENT_BYTES_INT8, bw)
    total_conv = compute_time(FLOPS_CONV_TOTAL) + tx_latent + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    total_mlp = compute_time_int8(FLOPS_MLP_TOTAL) + tx_latent + ENCODE_ASSEMBLY_US_INT8 + DECODE_ASSEMBLY_US_INT8
    transfer_table_int8.append([
        name, round(tx_raw, 4), round(tx_latent, 4), round(tx_raw, 4), round(total_conv, 4), round(total_mlp, 4)
    ])

transfer_int8 = pd.DataFrame(
    transfer_table_int8,
    columns=["Link", "TX Raw [µs]", "TX Latent [µs]", "RAW Total [µs]", "Conv SWAE Total [µs]", "MLP SWAE INT8 Total [µs]"]
)

# Table 2c: Transfer and total time for Float8 quantized MLP SWAE only
transfer_table_float8 = []
for name, bw in links.items():
    tx_raw = tx_time(RAW_BYTES, bw)
    tx_latent = tx_time(LATENT_BYTES_FLOAT8, bw)
    total_conv = compute_time(FLOPS_CONV_TOTAL) + tx_latent + ENCODE_ASSEMBLY_US + DECODE_ASSEMBLY_US
    total_mlp = compute_time_float8(FLOPS_MLP_TOTAL) + tx_latent + ENCODE_ASSEMBLY_US_FLOAT8 + DECODE_ASSEMBLY_US_FLOAT8
    transfer_table_float8.append([
        name, round(tx_raw, 4), round(tx_latent, 4), round(tx_raw, 4), round(total_conv, 4), round(total_mlp, 4)
    ])

transfer_float8 = pd.DataFrame(
    transfer_table_float8,
    columns=["Link", "TX Raw [µs]", "TX Latent [µs]", "RAW Total [µs]", "Conv SWAE Total [µs]", "MLP SWAE FLOAT8 Total [µs]"]
)

# Table 3: Cost Ledger (add data assembly cost rows)
ledger = pd.DataFrame({
    "Cost Item": [
        "Payload bytes (FP32)",
        "Encode FLOPs", "Encode time (µs)",
        "Encode data assembly (µs)",
        "Memory IO (kB)",
        "Decode FLOPs", "Decode time (µs)",
        "Decode data assembly (µs)",
        "Kernel launch overhead (µs)",
        "Peak VRAM (kB)",
        "PSNR (approx)", "Parameter Storage (MB)"
    ],
    "RAW": [500, 0, 0, 0, 2.0, 0, 0, 0, 0, "—", "lossless", "—"],
    "Conv SWAE": [
        32,  # CORRECTED from 64
        FLOPS_CONV_ENC, compute_time(FLOPS_CONV_ENC), ENCODE_ASSEMBLY_US,
        8.1,
        FLOPS_CONV_DEC, compute_time(FLOPS_CONV_DEC), DECODE_ASSEMBLY_US,
        20,
        1024,
        "48+ dB", 3.2
    ],
    "MLP SWAE": [
        32,  # CORRECTED from 64
        FLOPS_MLP_ENC, compute_time(FLOPS_MLP_ENC), ENCODE_ASSEMBLY_US,
        0.625,
        FLOPS_MLP_DEC, compute_time(FLOPS_MLP_DEC), DECODE_ASSEMBLY_US,
        5,
        256,
        "47+ dB", 0.398
    ],
    "MLP SWAE INT8": [
        8,  # CORRECTED from 16
        FLOPS_MLP_ENC, compute_time_int8(FLOPS_MLP_ENC), ENCODE_ASSEMBLY_US_INT8,
        0.625,
        FLOPS_MLP_DEC, compute_time_int8(FLOPS_MLP_DEC), DECODE_ASSEMBLY_US_INT8,
        5,
        256,
        "46+ dB", 0.398
    ],
    "MLP SWAE FLOAT8": [
        8,  # CORRECTED from 16
        FLOPS_MLP_ENC, compute_time_float8(FLOPS_MLP_ENC), ENCODE_ASSEMBLY_US_FLOAT8,
        0.625,
        FLOPS_MLP_DEC, compute_time_float8(FLOPS_MLP_DEC), DECODE_ASSEMBLY_US_FLOAT8,
        5,
        256,
        "47+ dB", 0.398
    ]
})

# Display
print("=== CORRECTED CALCULATIONS WITH 8 LATENT DIMENSIONS ===\n")
print("Compression ratio: 125 values / 8 latent = 15.625:1\n")

print("Table 1: Arithmetic Workload and Compute Time")
print(workload.to_string(index=False))
print()

print("Table 1b: Arithmetic Workload and Compute Time (INT8 MLP SWAE)")
print(workload_int8.to_string(index=False))
print()

print("Table 1c: Arithmetic Workload and Compute Time (Float8 MLP SWAE)")
print(workload_float8.to_string(index=False))
print()

print("Table 2: Transfer and Total Time")
print(transfer.to_string(index=False))
print()

print("Table 2b: Transfer and Total Time (INT8 MLP SWAE)")
print(transfer_int8.to_string(index=False))
print()

print("Table 2c: Transfer and Total Time (Float8 MLP SWAE)")
print(transfer_float8.to_string(index=False))
print()

print("Table 3: Cost Ledger")
print(ledger.to_string(index=False))

# Save results to computer_costs/
workload.to_csv(os.path.join(output_dir, 'workload_corrected.csv'), index=False)
transfer.to_csv(os.path.join(output_dir, 'transfer_corrected.csv'), index=False)
transfer_int8.to_csv(os.path.join(output_dir, 'transfer_int8_corrected.csv'), index=False)
transfer_float8.to_csv(os.path.join(output_dir, 'transfer_float8_corrected.csv'), index=False)
ledger.to_csv(os.path.join(output_dir, 'ledger_corrected.csv'), index=False)

# Generate comparison plot
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Latency comparison
networks = transfer['Link'].str.split(' ').str[0]
raw_latency = transfer['RAW Total [µs]'].values
conv_latency = transfer['Conv SWAE Total [µs]'].values
mlp_latency = transfer['MLP SWAE Total [µs]'].values
mlp_float8_latency = transfer_float8['MLP SWAE FLOAT8 Total [µs]'].values

x = np.arange(len(networks))
width = 0.2

ax1.bar(x - 1.5*width, raw_latency, width, label='Raw Transfer', color='blue')
ax1.bar(x - 0.5*width, conv_latency, width, label='Conv SWAE', color='orange')
ax1.bar(x + 0.5*width, mlp_latency, width, label='MLP SWAE (FP32)', color='green')
ax1.bar(x + 1.5*width, mlp_float8_latency, width, label='MLP SWAE (Float8)', color='red')

ax1.set_xlabel('Network Type')
ax1.set_ylabel('Total Latency (µs)')
ax1.set_title('Total Latency Comparison (Corrected with 8 Latent Dimensions)')
ax1.set_xticks(x)
ax1.set_xticklabels(networks, rotation=45, ha='right')
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup factors
speedup_conv = raw_latency / conv_latency
speedup_mlp = raw_latency / mlp_latency
speedup_mlp_float8 = raw_latency / mlp_float8_latency

ax2.plot(networks, speedup_conv, 'o-', label='Conv SWAE', linewidth=2, markersize=8)
ax2.plot(networks, speedup_mlp, 's-', label='MLP SWAE (FP32)', linewidth=2, markersize=8)
ax2.plot(networks, speedup_mlp_float8, '^-', label='MLP SWAE (Float8)', linewidth=2, markersize=8)

ax2.set_xlabel('Network Type')
ax2.set_ylabel('Speedup Factor')
ax2.set_title('Speedup vs Raw Transfer (Corrected)')
ax2.set_xticks(range(len(networks)))
ax2.set_xticklabels(networks, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(speedup_mlp_float8) * 1.1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'transfer_comparison_corrected.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nKey corrections made:")
print("1. Latent dimension: 16 → 8 (matching actual model)")
print("2. Latent bytes FP32: 64 → 32 bytes")
print("3. Latent bytes Float8/INT8: 16 → 8 bytes")
print("4. Float8 speedup: Using realistic 3x instead of theoretical 14x")
print("5. MLP FLOPs updated for 8 latent dimensions")
print(f"\nPlots saved as '{os.path.join(output_dir, 'transfer_comparison_corrected.png')}'")