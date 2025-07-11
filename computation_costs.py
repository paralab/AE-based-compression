import pandas as pd

# Constants
TFLOPS = 19.5 * 1e12  # FP32 throughput on A100
TFLOPS_INT8 = 312 * 1e12  # INT8 throughput on A100 (NVIDIA A100: up to 312 TOPS for INT8)
TFLOPS_FLOAT8 = 280 * 1e12  # Float8 throughput on A100 (estimated, similar to INT8 but slightly lower)
BYTES_PER_FLOAT = 4
BYTES_PER_INT8 = 1
BYTES_PER_FLOAT8 = 1  # Float8 uses 1 byte per element
RAW_BYTES = 125 * BYTES_PER_FLOAT
LATENT_BYTES = 16 * BYTES_PER_FLOAT

# For quantized MLP SWAE only (not for Conv or RAW)
LATENT_BYTES_INT8 = 16 * BYTES_PER_INT8
LATENT_BYTES_FLOAT8 = 16 * BYTES_PER_FLOAT8

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
FLOPS_MLP_ENC = 398336

# MLP Decoder:
# 16×256 + 256×512 + 512×125 = 4096 + 131072 + 64000 MACs = 199168 MACs -> 398336 FLOPs
FLOPS_MLP_DEC = 398336
FLOPS_MLP_TOTAL = FLOPS_MLP_ENC + FLOPS_MLP_DEC

# Compute time (in microseconds)
def compute_time(flop):
    return flop / TFLOPS * 1e6

def compute_time_int8(flop):
    return flop / TFLOPS_INT8 * 1e6

def compute_time_float8(flop):
    return flop / TFLOPS_FLOAT8 * 1e6

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
        64,
        FLOPS_CONV_ENC, compute_time(FLOPS_CONV_ENC), ENCODE_ASSEMBLY_US,
        16,
        FLOPS_CONV_DEC, compute_time(FLOPS_CONV_DEC), DECODE_ASSEMBLY_US,
        "≥ 5", 8, "≈ 40 dB", 8.1
    ],
    "MLP SWAE": [
        64,
        FLOPS_MLP_ENC, compute_time(FLOPS_MLP_ENC), ENCODE_ASSEMBLY_US,
        1.5,
        FLOPS_MLP_DEC, compute_time(FLOPS_MLP_DEC), DECODE_ASSEMBLY_US,
        "≥ 3", 2, "≈ 38 dB", 1.8
    ]
})

# Table 3b: Cost Ledger for INT8 quantized MLP SWAE only
ledger_int8 = pd.DataFrame({
    "Cost Item": [
        "Payload bytes (INT8)",
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
        64,
        FLOPS_CONV_ENC, compute_time(FLOPS_CONV_ENC), ENCODE_ASSEMBLY_US,
        16,
        FLOPS_CONV_DEC, compute_time(FLOPS_CONV_DEC), DECODE_ASSEMBLY_US,
        "≥ 5", 8, "≈ 40 dB", 8.1
    ],
    "MLP SWAE INT8": [
        16,
        FLOPS_MLP_ENC, compute_time_int8(FLOPS_MLP_ENC), ENCODE_ASSEMBLY_US_INT8,
        0.375,
        FLOPS_MLP_DEC, compute_time_int8(FLOPS_MLP_DEC), DECODE_ASSEMBLY_US_INT8,
        "≥ 3", 0.5, "≈ 36 dB*", 0.45
    ]
})

# Table 3c: Cost Ledger for Float8 quantized MLP SWAE only
ledger_float8 = pd.DataFrame({
    "Cost Item": [
        "Payload bytes (Float8)",
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
        64,
        FLOPS_CONV_ENC, compute_time(FLOPS_CONV_ENC), ENCODE_ASSEMBLY_US,
        16,
        FLOPS_CONV_DEC, compute_time(FLOPS_CONV_DEC), DECODE_ASSEMBLY_US,
        "≥ 5", 8, "≈ 40 dB", 8.1
    ],
    "MLP SWAE FLOAT8": [
        16,
        FLOPS_MLP_ENC, compute_time_float8(FLOPS_MLP_ENC), ENCODE_ASSEMBLY_US_FLOAT8,
        0.375,
        FLOPS_MLP_DEC, compute_time_float8(FLOPS_MLP_DEC), DECODE_ASSEMBLY_US_FLOAT8,
        "≥ 3", 0.5, "≈ 37 dB*", 0.45
    ]
})
# *PSNR and storage values are illustrative; actual values depend on quantization quality.

print("Workload and Latency Table (including Data Assembly Cost, FP32)")
print(workload)
print("\nTransfer Time and Total Latency (FP32)")
print(transfer)
print("\nComprehensive Cost Ledger (including Data Assembly Cost, FP32)")
print(ledger)

print("\nWorkload and Latency Table (INT8 Quantized MLP SWAE Only)")
print(workload_int8)
print("\nTransfer Time and Total Latency (INT8 Quantized MLP SWAE Only)")
print(transfer_int8)
print("\nComprehensive Cost Ledger (INT8 Quantized MLP SWAE Only)")
print(ledger_int8)

print("\nWorkload and Latency Table (Float8 Quantized MLP SWAE Only)")
print(workload_float8)
print("\nTransfer Time and Total Latency (Float8 Quantized MLP SWAE Only)")
print(transfer_float8)
print("\nComprehensive Cost Ledger (Float8 Quantized MLP SWAE Only)")
print(ledger_float8)

import os

# Ensure the output directory exists
output_dir = "exact_flops_analysis"
os.makedirs(output_dir, exist_ok=True)

# Save the tables as CSV files in the specified folder
workload.to_csv(os.path.join(output_dir, "workload.csv"), index=False)
transfer.to_csv(os.path.join(output_dir, "transfer.csv"), index=False)
ledger.to_csv(os.path.join(output_dir, "ledger.csv"), index=False)

workload_int8.to_csv(os.path.join(output_dir, "workload_int8.csv"), index=False)
transfer_int8.to_csv(os.path.join(output_dir, "transfer_int8.csv"), index=False)
ledger_int8.to_csv(os.path.join(output_dir, "ledger_int8.csv"), index=False)

workload_float8.to_csv(os.path.join(output_dir, "workload_float8.csv"), index=False)
transfer_float8.to_csv(os.path.join(output_dir, "transfer_float8.csv"), index=False)
ledger_float8.to_csv(os.path.join(output_dir, "ledger_float8.csv"), index=False)

# Save image versions of the tables
import matplotlib.pyplot as plt

def save_table_as_image(df, filename):
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns)*2), max(2, len(df)*0.5)))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

save_table_as_image(workload, os.path.join(output_dir, "workload.png"))
save_table_as_image(transfer, os.path.join(output_dir, "transfer.png"))
save_table_as_image(ledger, os.path.join(output_dir, "ledger.png"))

save_table_as_image(workload_int8, os.path.join(output_dir, "workload_int8.png"))
save_table_as_image(transfer_int8, os.path.join(output_dir, "transfer_int8.png"))
save_table_as_image(ledger_int8, os.path.join(output_dir, "ledger_int8.png"))

save_table_as_image(workload_float8, os.path.join(output_dir, "workload_float8.png"))
save_table_as_image(transfer_float8, os.path.join(output_dir, "transfer_float8.png"))
save_table_as_image(ledger_float8, os.path.join(output_dir, "ledger_float8.png"))