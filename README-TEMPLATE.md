> [!WARNING]
> **INTERNAL USE ONLY — NOT FOR PUBLIC DISTRIBUTION**
> This document has not been approved for public release. It must not be committed to a public repository or shared externally until this notice is explicitly removed by the document owner prior to merge.

# OCI GPU Quick Start: <Vendor> <GPU_FAMILY>

This document provides hardware specifications, supported OS images, onboarding verification, sample benchmarks, and best-practices for OCI deployments using the <Vendor> <GPU_FAMILY> GPU shape.

## At a Glance

- Shape: `<shape name>`
- GPU configuration: `<gpu model and count>`
- Recommended OS baseline: `<minimum supported OS>`
- Recommended software baseline: `<primary driver/CUDA or ROCm/NCCL or RCCL>`
- Primary verification command: `<hello world or device visibility command>`
- Operational profile: `<single-node, scale-out, topology-sensitive, etc.>`

## When To Use This Shape

Use this shape when you need `<brief workload fit>`.

It is best suited for `<single-node vs scale-out guidance>` and is a strong fit when `<topology, memory, or networking differentiator>`.

## First 15 Minutes

1. Launch the shape with one of the approved images from the [Provided Images](#provided-images) table.
2. Run the primary verification command and confirm all GPUs are visible.
3. Run the hello-world or container smoke test from [Hello World Verification](#hello-world-verification).
4. Confirm the expected RDMA or interconnect interfaces are up before moving on to benchmarks.

# Table of Contents
* [Hardware Specifications](#hardware-specifications)
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
| <!-- Shape Name --> | <!-- GPU Model --> | <!-- GPUs/Node --> | <!-- GPU Memory --> | <!-- GPU Memory Total --> | <!-- CPU --> | <!-- # of CPUs --> | <!-- System Memory --> | <!-- Local Storage --> | <!-- Host NIC --> | <!-- RDMA (ROCe) NICs --> |

See the [OCI Compute Shapes Docs](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) for up-to-date details.

# Recommended Operating Systems

## Recommended Software Version

## Custom OS Image Creation with Packer

To build your images using packer clone the OCI HPC Images repo and run the commands found there [OCI HPC Images GitHub Repo](https://github.com/oracle-quickstart/oci-hpc-images/blob/main/README.md).

## Provided Images

| OS Version | Image Packer Build Details | OCI Platform Image Link | Driver Versions | Build & Dependency Status |
|---|---|---|---|---|

## Hello World Verification

# Performance Benchmarks

# OKE GPU Getting Started

# Troubleshooting

Here you can find suggested troubleshooting methods.

# Further Reading & Support
