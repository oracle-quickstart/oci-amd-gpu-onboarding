# OCI GPU Quick Start: AMD MI300X Instinct GPU

This document provides hardware specifications, supported OS images, onboarding verification, sample benchmarks, and best-practices for OCI deployments using the AMD MI300X GPU bare-metal shape.

BM.GPU.MI300X.8 is a scale-up and scale-out OCI GPU shape built around eight AMD Instinct MI300X GPUs with XGMI connectivity, dual Intel Xeon Platinum 8480+ processors, and RoCE-capable networking for AI/ML and HPC workloads.

## At a Glance

- Shape: `BM.GPU.MI300X.8`
- GPU configuration: `8 x AMD Instinct MI300X 192 GB`
- Recommended OS baseline: `Oracle Linux 9+` or `Ubuntu Linux 22.04+`
- Recommended software baseline: `ROCm 7.2+, RCCL 2.26.6, OFED 28.40.1202+`
- Primary verification command: `amd-smi`
- Operational profile: `single-node and multi-node AI/ML and HPC with XGMI topology`

# Table of Contents

* [Hardware Specifications](#hardware-specifications)
* [Recommended Operating Systems](#recommended-operating-systems)
    * [Recommended Software Version](#recommended-software-version)
    * [Custom OS Image Creation with Packer](#custom-os-image-creation-with-packer)
    * [Provided Images](#provided-images)
    * [Hello World Verification](#hello-world-verification)
* [Performance Benchmarks](#performance-benchmarks)
    * [RCCL & Model Inference Performance](#rccl--model-inference-performance)
* [OKE GPU Getting Started](#oke-gpu-getting-started)
* [Troubleshooting](#troubleshooting)
* [Further Reading & Support](#further-reading--support)

# Hardware Specifications

| Shape Name | GPU Model | GPUs/Node | GPU Memory (GB/GPU) | GPU Memory Total | CPU | # of CPUs | System Memory | Local Storage | Host NIC | RDMA (RoCE) NICs |
|---|---|---|---|---|---|---|---|---|---|---|
| BM.GPU.MI300X.8 | AMD Instinct MI300X | 8 | 192 | 1.5 TB | 2 x Intel Xeon Platinum 8480+ | 112 Cores | 2 TB DDR5 | 8 x 3.5 TB NVMe = 28 TB | 100 Gb/s | 8 x 1 x 400 Gb/s = 3.2 Tb/s |

See the [OCI Compute Shapes Docs](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) for up-to-date details.

# Recommended Operating Systems

- Oracle Linux 9+
- Ubuntu Linux 22.04+

## Recommended Software Version

- ROCm 7.2+
- RCCL 2.26.6
- OFED 28.40.1202+
- Oracle Cloud Agent 1.52.0+
- HPC-X 2.23+

## Custom OS image Creation with Packer

To build your images using packer clone the OCI HPC Images repo and run the commands found there [OCI HPC Images GitHub Repo](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/README.md).

## Provided Images

It is recommended to use Oracle-provided or organizationally-approved images for security and supportability. See the OS images table for region-specific OCIDs and version information.

| OS Version | Image Packer Build Details | OCI Platform Image Link | Driver Versions | Build & Dependency Status |
|---|---|---|---|---|
| OCI GPU AI Image with Oracle Linux 9 | [`Oracle-Linux-9.6-RHCK-DOCA-OFED-3.2.1-AMD-ROCM-643`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/OracleLinux-9/Oracle-Linux-9.6-RHCK-DOCA-OFED-3.2.1-AMD-ROCM-643.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Oracle-Linux-9.6-2025.11.20-0-RHCK-DOCA-OFED-3.2.1-AMD-ROCM-643-2026.03.13-0) | ROCm 6.4.3, DOCA OFED 3.2.1, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Oracle Linux 9 | [`Oracle-Linux-9.7-RHCK-DOCA-OFED-3.2.1-AMD-ROCM-72`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/OracleLinux-9/Oracle-Linux-9.7-RHCK-DOCA-OFED-3.2.1-AMD-ROCM-72.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Oracle-Linux-9.7-2026.02.28-0-RHCK-DOCA-OFED-3.2.1-AMD-ROCM-72-2026.03.13-0) | ROCm 7.2, DOCA OFED 3.2.1, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 22.04 | [`Canonical-Ubuntu-22.04-DOCA-OFED-3.2.1-AMD-ROCM-643`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-22/Canonical-Ubuntu-22.04-DOCA-OFED-3.2.1-AMD-ROCM-643.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-22.04-2026.02.28-0-DOCA-OFED-3.2.1-AMD-ROCM-643-2026.05.05-0) | ROCm 6.4.3, DOCA OFED 3.2.1, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 22.04 | [`Canonical-Ubuntu-22.04-DOCA-OFED-3.2.1-AMD-ROCM-72`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-22/Canonical-Ubuntu-22.04-DOCA-OFED-3.2.1-AMD-ROCM-72.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-22.04-2026.02.28-0-DOCA-OFED-3.2.1-AMD-ROCM-72-2026.05.05-0) | ROCm 7.2, DOCA OFED 3.2.1, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 24.04 | [`Canonical-Ubuntu-24.04-6.8-DOCA-OFED-3.2.1-AMD-ROCM-643`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-24/Canonical-Ubuntu-24.04-6.8-DOCA-OFED-3.2.1-AMD-ROCM-643.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-24.04-2026.02.28-0-6.8-DOCA-OFED-3.2.1-AMD-ROCM-643-2026.05.05-0) | ROCm 6.4.3, DOCA OFED 3.2.1, Kernel 6.8, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 24.04 | [`Canonical-Ubuntu-24.04-6.8-DOCA-OFED-3.2.1-AMD-ROCM-72`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-24/Canonical-Ubuntu-24.04-6.8-DOCA-OFED-3.2.1-AMD-ROCM-72.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-24.04-2026.02.28-0-6.8-DOCA-OFED-3.2.1-AMD-ROCM-72-2026.05.05-0) | ROCm 7.2, DOCA OFED 3.2.1, Kernel 6.8, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |
| OCI GPU AI Image with Ubuntu Linux 24.04 | [`Canonical-Ubuntu-24.04-6.14-DOCA-OFED-3.2.1-AMD-ROCM-72`](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/images/Ubuntu-24/Canonical-Ubuntu-24.04-6.14-DOCA-OFED-3.2.1-AMD-ROCM-72.pkr.hcl) | [PAR Link](https://objectstorage.ca-montreal-1.oraclecloud.com/p/AIo4CP0P_DlUelDlsWgGPWmY6FcBQzJWmmFyGKdY0epkh87a9Q3ndvFYycjIxTQ9/n/idxzjcdglx2s/b/images/o/Canonical-Ubuntu-24.04-2026.02.28-0-6.14-DOCA-OFED-3.2.1-AMD-ROCM-72-2026.05.05-0) | ROCm 7.2, DOCA OFED 3.2.1, Kernel 6.14, OCA 1.57.0 | ![Build](/media/icons/build-passing.svg) ![Build](/media/icons/dependencies.svg) |

# Hello World Verification

Use `amd-smi` or `rocm-smi` to confirm that all eight GPUs are visible and idle before launching a workload:

```bash
amd-smi
rocm-smi
```

`amd-smi topology` should show a fully connected XGMI topology across the eight devices.

# Performance Benchmarks

For benchmark setup guidance, see the [AMD RCCL documentation](https://rocm.docs.amd.com/projects/rccl) and the [AMD system validation guidance](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/common/system-validation.html).

## RCCL & Model Inference Performance

The current MI300X source material emphasizes RCCL, RVS, and system-validation workflows more than a normalized model-inference table. Collective communication results below are sourced from the current handover document.

### All Reduce - Single Node

- Large-message reference point: `188.65 GB/s` algorithm bandwidth and `330.13 GB/s` bus bandwidth at `17179869184` bytes

### All Reduce - 2 Nodes

- Large-message reference point: `200.11 GB/s` algorithm bandwidth and `375.21 GB/s` bus bandwidth at `17179869184` bytes

### Model Inference Performance

No normalized MI300X model-inference table is currently published in the source material used for this workflow.

# OKE GPU Getting Started

1. Create an OKE cluster through the OCI Console or CLI.
2. Add the GPU node pool using one of the approved MI300X images listed above.
3. Install the AMD device plugin:

```bash
kubectl apply -f https://github.com/AMD/k8s-device-plugin/raw/main/AMD-device-plugin.yml
```

4. Verify that the GPUs are visible from a test pod before deploying production workloads.

# Troubleshooting

Here you can find suggested troubleshooting methods.

* [Confirm Hardware & Drivers](#confirm-hardware--drivers)
* [RDMA Link](#rdma-link)
* [RVS](#rvs)

## Confirm Hardware & Drivers

```bash
amd-smi
amd-smi topology
numactl --hardware
```

The MI300X source material expects all eight GPUs to be visible and `amd-smi topology` to report a symmetric XGMI topology.

## RDMA Link

```bash
rdma link
```

The front-end network is expected on `mlx5_1 (eth0)` with RoCE RDMA interfaces exposed as `rdma*` devices and links reporting `ACTIVE` and `LINK_UP`.

## RVS

```bash
/opt/rocm/bin/rvs
```

RVS is included with the ROCm stack and is the preferred system-validation workflow called out by the source material for MI300X.

## Further Reading & Support

### AMD Technical Documents

1. [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
2. [AMD ROCm examples](https://github.com/ROCm/rocm-examples)
3. [AMD ROCm Validation Suite](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/latest/)
4. [AMD system validation guidance](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/common/system-validation.html)

### OCI AMD Technical Docs & Supported Solutions

1. [OCI GPU Scanner - Cluster Health Management Solution](https://github.com/oracle-quickstart/oci-gpu-scanner)
2. [OCI HPC SLURM Deployed Terraform Stack](https://github.com/oracle-quickstart/oci-hpc)
3. [OCI HPC OKE Deployment Stack](https://github.com/oracle-quickstart/oci-hpc-oke)
