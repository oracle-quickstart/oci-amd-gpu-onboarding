> [!WARNING]
> **INTERNAL USE ONLY - NOT FOR PUBLIC DISTRIBUTION**
> This document has not been approved for public release. It must not be committed to a public repository or shared externally until this notice is explicitly removed by the document owner prior to merge.

# OCI GPU Quick Start: NVIDIA GB300 Multiplanar

This guide covers OCI deployments using the 4-plane Multiplanar variant of `BM.GPU.GB300.4`. GB300 is a rack-scale, multi-host NVLink shape: one public shape name can represent different RDMA topology variants, so confirm the fabric metadata before you scale a workload.

## At a Glance

- Shape: `BM.GPU.GB300.4`
- Variant: Multiplanar
- Metadata signal: `rdmaFabricData.planes = 4`
- GPU configuration: `4 x NVIDIA B300`
- Recommended OS baseline: `Oracle Linux 9+` or `Ubuntu Linux 24.04+`
- Recommended software baseline: `DOCA OFED 3.1.0+, NVIDIA Driver 580+ (Open), CUDA 13.0, NCCL 2.28.7+`
- Primary verification command: `nvidia-smi`
- Operational profile: `rack-scale NVLink with 4-plane Multiplanar RDMA topology`

## When To Use This Shape

Use this variant when you need GB300 GPU performance with rack-aware placement and multi-host collective behavior. The Multiplanar variant is intended for scale-out workloads that need multiple RDMA planes available to the same host.

It is best suited for topology-sensitive training, benchmark validation, and deployments where host placement, GPU memory fabric boundaries, and RDMA plane count can materially affect performance.

## First 15 Minutes

1. Launch `BM.GPU.GB300.4` with an approved arm64 image from the [Provided Images](#provided-images) table.
2. Run `nvidia-smi` and confirm all four GB300 GPUs are visible.
3. Confirm the host is on the Multiplanar variant by checking that instance metadata reports `rdmaFabricData.planes = 4`.
4. Verify RDMA links before running NCCL or OKE workloads.
5. Use the GB300 OKE manifest or your own topology-aware scheduler configuration when scaling beyond a single host.

# Table of Contents

* [Hardware Specifications](#hardware-specifications)
* [Topology Variant](#topology-variant)
* [When To Use This Shape](#when-to-use-this-shape)
* [First 15 Minutes](#first-15-minutes)
* [Recommended Operating Systems](#recommended-operating-systems)
    * [Recommended Software Version](#recommended-software-version)
    * [Custom OS Image Creation with Packer](#custom-os-image-creation-with-packer)
    * [Provided Images](#provided-images)
    * [Hello World Verification](#hello-world-verification)
* [Performance Benchmarks](#performance-benchmarks)
* [OKE GPU Getting Started](#oke-gpu-getting-started)
* [Troubleshooting](#troubleshooting)
* [Further Reading & Support](#further-reading--support)

# Hardware Specifications

| Shape Name | GPU Model | GPUs/Node | GPU Memory (GB/GPU) | GPU Memory Total | CPU | # of CPUs | System Memory | Local Storage | Host NIC | RDMA (ROCe) NICs |
|---|---|---|---|---|---|---|---|---|---|---|
| BM.GPU.GB300.4 | B300 | 4 | 278 GB | 1112 GB | NVIDIA Grace / Arm Neoverse V2 | 144 cores | 960 GB | 4 x 7.68 TB NVMe | 100 Gb/s | 8 x 400 Gb/s RDMA |

See the [OCI Compute Shapes Docs](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) for up-to-date details.

# Topology Variant

`BM.GPU.GB300.4` currently has at least two RDMA topology variants that use the same public OCI shape name. This README is for the Multiplanar variant only.

| Variant | Instance metadata signal | Operational meaning |
|---|---|---|
| Multiplanar | `rdmaFabricData.planes = 4` | Four RDMA planes are exposed to the host. Use this for Multiplanar GB300 deployments and topology-aware scale-out validation. |

Do not rely on the shape name alone to identify this variant. Confirm the topology from instance metadata and keep hosts with different plane counts out of the same performance baseline unless the workload is intentionally testing mixed topology behavior.

# Recommended Operating Systems

- Oracle Linux 9+
- Ubuntu Linux 24.04+

## Recommended Software Version

- DOCA OFED 3.1.0+
- NVIDIA Driver 580+ (Open)
- CUDA 13.0
- NCCL 2.28.7+
- HPC-X 2.24.1+
- Oracle Cloud Agent 1.57.0+

## Custom OS Image Creation with Packer

To build your images using packer clone the OCI HPC Images repo and run the commands found there [OCI HPC Images GitHub Repo](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/README.md).

## Provided Images

| OS Version | Image Packer Build Details | OCI Platform Image Link | Driver Versions | Build & Dependency Status |
|---|---|---|---|---|
| OCI GPU AI Image with Ubuntu Linux 22.04 | [`Canonical-Ubuntu-22.04-aarch64-64k-page-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-22/Canonical-Ubuntu-22.04-aarch64-64k-page-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-22.04-aarch64-2026.02.28-0-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0-2026.05.05-0) | NVIDIA OPEN 580, DOCA OFED 3.2.1, CUDA 13.0, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 24.04 | [`Canonical-Ubuntu-24.04-aarch64-64k-page-6.8-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-24/Canonical-Ubuntu-24.04-aarch64-64k-page-6.8-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-24.04-aarch64-2026.02.28-0-6.8-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0-2026.05.05-0) | NVIDIA OPEN 580, DOCA OFED 3.2.1, CUDA 13.0, Kernel 6.8, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 24.04 | [`Canonical-Ubuntu-24.04-aarch64-64k-page-6.17-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-24/Canonical-Ubuntu-24.04-aarch64-64k-page-6.17-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-24.04-aarch64-2026.02.28-0-6.17-DOCA-OFED-3.2.1-GPU-580-OPEN-CUDA-13.0-2026.05.05-0) | NVIDIA OPEN 580, DOCA OFED 3.2.1, CUDA 13.0, Kernel 6.17, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |

## Hello World Verification

Run `nvidia-smi` to verify that all four GPUs are visible and the driver stack is loaded:

```bash
nvidia-smi
```

Expected result:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.x             Driver Version: 580.x        CUDA Version: 13.0             |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|=========================================+========================+======================|
|   0  NVIDIA GB300                   On  | ...                    |                    0 |
|   1  NVIDIA GB300                   On  | ...                    |                    0 |
|   2  NVIDIA GB300                   On  | ...                    |                    0 |
|   3  NVIDIA GB300                   On  | ...                    |                    0 |
+-----------------------------------------------------------------------------------------+
```

You should see four GB300 GPUs, no unexpected ECC errors, and the expected CUDA and driver versions for the image you selected.

Confirm Multiplanar topology from instance metadata:

```bash
curl -sH "Authorization: Bearer Oracle" -L \
  http://169.254.169.254/opc/v2/instance/ | jq '.shape, .rdmaFabricData'
```

Expected Multiplanar signal:

```json
"BM.GPU.GB300.4"
{
  "ipv6": true,
  "planes": 4
}
```

# Performance Benchmarks

NVIDIA publishes [NCCL](https://developer.nvidia.com/nccl) as the primary collective communication library for multi-GPU AI and HPC workloads. For guidance on running NCCL tests, see the [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html).

The benchmark data collected for this shape is most useful when interpreted by topology:

| Benchmark | Scope | Healthy signal |
|---|---|---|
| NCCL AllReduce | Single host | Large-message bandwidth should confirm that all four local GPUs are participating. |
| NCCL AllReduce | Multiple hosts in one Multiplanar rack | Large-message bandwidth should remain strong when placement stays within the same rack/fabric boundary. |
| NCCL All-to-All | Multiple hosts across racks | Expect lower bandwidth when traffic crosses rack or fabric boundaries. Treat this as a topology signal, not necessarily a host fault. |

When comparing runs, record the number of hosts, the host placement boundary, and whether each host reports `rdmaFabricData.planes = 4`. Mixing Single Plane and Multiplanar hosts can make benchmark results misleading.

## NCCL Result Format

Keep large-message rows from `all_reduce_perf` or `alltoall_perf` output and omit smaller rows, launch scripts, hostnames, usernames, and internal paths before sharing results.

```text
# Collective test starting: all_reduce_perf
# nThread 1 nGpus 1 minBytes 524288 maxBytes 17179869184 step: 2(factor)
NCCL version 2.28.7+cuda13.0
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
# <smaller message sizes truncated>
  8589934592    2147483648     float     sum      -1    ...    ...     ...        0    ...    ...     ...        0
 17179869184    4294967296     float     sum      -1    ...    ...     ...        0    ...    ...     ...        0
# Out of bounds values : 0 OK
# Collective test concluded: all_reduce_perf
```

# OKE GPU Getting Started

Information on getting up and running on OKE can be found [here](https://github.com/oracle-quickstart/oci-hpc-oke).

Useful GB300-specific OKE starting points in `oci-hpc-oke`:

- [GB300 NCCL test manifest](https://github.com/oracle-quickstart/oci-hpc-oke/blob/main/manifests/nccl-tests/kueue/BM.GPU.GB300.4.yaml)
- [Running active health checks on OKE](https://github.com/oracle-quickstart/oci-hpc-oke/blob/main/docs/running-active-health-checks.md)
- [Running ib_write_bw on OKE](https://github.com/oracle-quickstart/oci-hpc-oke/blob/main/docs/running-ib-write-bw-test.md)

For Multiplanar validation, make sure the scheduler places pods on hosts that report the expected plane count and fabric boundary before treating NCCL results as a baseline.

# Troubleshooting

Here you can find suggested troubleshooting methods.

* [GPU visibility](#gpu-visibility)
* [Topology metadata](#topology-metadata)
* [RDMA links](#rdma-links)
* [NUMA placement](#numa-placement)
* [DCGMI diagnostics](#dcgmi-diagnostics)

## GPU visibility

Use `nvidia-smi` first. It confirms that the driver stack is loaded and all four GPUs are visible to the host.

```bash
nvidia-smi
```

Healthy output should show four NVIDIA GB300 GPUs and no unexpected ECC errors.

## Topology metadata

Use instance metadata to confirm that the host is the intended topology variant:

```bash
curl -sH "Authorization: Bearer Oracle" -L \
  http://169.254.169.254/opc/v2/instance/ | jq '.shape, .rdmaFabricData'
```

For Multiplanar GB300, `planes` should be `4`.

## RDMA links

Use `rdma link` to confirm that the expected RDMA interfaces are active:

```bash
rdma link
```

Healthy output should show the expected RDMA interfaces in `ACTIVE` state with `LINK_UP` physical state. If a link is down, check placement, cabling state, and the image/network initialization path before running NCCL.

## NUMA placement

Use `numactl` to understand CPU and memory locality before running RDMA or NCCL tests:

```bash
numactl --hardware
numactl --show
```

Bind validation tools to the NUMA node associated with the interface under test when comparing per-plane RDMA behavior.

## DCGMI diagnostics

Use DCGMI diagnostics when GPU health is uncertain:

```bash
dcgmi diag -r 1
```

Increase the diagnostic level only when basic checks indicate a GPU or driver issue. See the [NVIDIA DCGM diagnostics documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html) for test levels and expected behavior.

# Further Reading & Support

- [OCI Compute Shapes Docs](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm)
- [OCI HPC Images GitHub Repo](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/README.md)
- [OCI HPC OKE GitHub Repo](https://github.com/oracle-quickstart/oci-hpc-oke)
- [NVIDIA NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NVIDIA DCGM diagnostics documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [NVIDIA IMEX guide](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html)
