---
title: "i.MX93 bringup using executorch"
date: 2026-03-19T23:52:16+01:00
draft: true
---

Due to my recent journey into the intestines of Pytorch I wanted to try out an adjacent edge workflow.
What better to use than [Executorch](https://pytorch.org/blog/deploying-pytorch-models-to-the-micro-edge-with-executorch-and-arm/)?
Specifically, I'm evaluating some ARM boards containing NPUs for an inference pipeline. Extending a public toolchain with sufficient
organized buy-in seems like a good route to me so I spent a few days understanding how a model maps to through the Executorch path
to an edge device. I'm lining out the steps I've taken and what numbers I could gather here, specifically quantifying what kinds of
models we can run.


## Porting

On the i.MX93, the Ethos-U65 NPU sits behind a Cortex-M33 coprocessor. The A55 Linux side can't talk to the NPU directly, it rather sends
models and data to the M33 via `remoteproc/rpmsg`, the M33 firmware runs TFLite Micro with an Ethos-U delegate,
and the NPU executes the compiled command stream. So for an inference call you'll need to through two CPUs, a bus and of course the NPU.

Executorch had no support for this when I touched this a few days ago. Arm's existing Ethos-U backend targets bare-metal
Cortex-M. The i.MX93 Linux path needs a completely different runtime. Open `/dev/ethosu0`, hand the NXP kernel driver a compiled
tflite model and an arena buffer, and let the driver stack handle the rest.

### Slow paths

As I'm not a very experienced embedded developer I took the hard route and thought I'd definitely needed to write my own bringup.
In doing so I've learned a lot about hardware and how Linux kernels see it, it's a painful process that in the end, it turns it,
doesn't really help getting to better performance.

The M33 has 128KB ITCM (code), 128KB DTCM (data), and I reserved 128MB of DDR for the NPU's tensor arena.
GNU `ld`'s error messages for linker script bugs sometimes don't really mean anything to me, such as "section has no contents" can mean you accidentally
discarded `.tm_clone_table`, or you put PROGBITS sections in a NOLOAD segment, or half a dozen other things.
The final layout routes code to ITCM, data to DTCM, and the tensor arena & model weights to DDR via a separate `PT_LOAD` segment.

The M33's Address Translation Table maps ITCM and DTCM to A55-visible physical addresses, but DDR is _not_ in the ATT.
You can't ioremap 128MB. The `.ddr` section must use PHDRS `:NONE` to prevent remoteproc from trying to load it.
And if you ever manually poke `ethosu_firmware` into the remoteproc sysfs node while the kernel's ethosu driver is loaded, the kernel crashes.
If you're not there to powercycle, well, you're out of luck and have to wait until you can touch the device.

The NPU itself needs `secure=0, privilege=0` on i.MX93, different from what I've seen from other Arm platforms.
And it cannot DMA from the M33's DTCM, so the PTE model blob must be copied to DDR before inference.

All of this produced a (barely) working bare-metal inference path: SmallConvnet, 23/26 ops on NPU, 0.08ms per inference.
More importantly, it gave us a ground-truth understanding of what the NPU actually does, how memory flows, and where the abstraction boundaries are.

### Fast path on A55 Linux

With the hardware model established, the Linux backend was a lot less worrying.
The core is `EthosUBackend_iMX.cpp` which is 350 lines that open the NXP device, stage the compiled tflite into an arena buffer,
copy input tensors to offsets implied by the model, invoke the driver, and copy outputs back.

The NXP driver stack works smoothly even if you sometimes have to go through some trial and error to get the hang of it.
`getIfmDims()` returns buffer *sizes*, not tensor dimensions. The arena is a single flat buffer for all IO and scratch.
There is no separate IFM/OFM buffers like Arm's reference driver. `Inference::invoke()` blocks until the M33 firmware completes and signals back via rpmsg.

The export side needed some changes. Arm's `arm_vela.py` compiles models to a raw NPZ format.
The NXP Linux stack needs the full Vela-compiled tflite. We run Vela twice: once for the raw command stream
(which Executorch's delegate payload format expects) and once for the tflite. The latter  gets embedded as a `vela_model` block in the delegate payload.
The iMX backend extracts this block and hands it to the NXP driver.

The device runs glibc. Compiling on MacOS has its challenges if you try to run homebrew and not rely on a good package manager (mamba or pixi?). Ie.
running your cross-compilation you need ot figure out where some flags are injected through your env you do not necessarily want especially if you do not
compile for your aarch64 or whatever it is on my machine.

### A few hiccups

The first models that ran produced nothing valuable. MobileNetV2 executed successfully but the argmax was completely wrong.
Cosine similarity between device and host output was 0.19. So I had to look into why the pipeline was broken.

After some digging: `--channels_last_4d`, the machine expects some standard format for image-digesting models.
Executorch's export flag converts model inputs to NHWC memory layout.
The host-side quantization and reference computation work correctly in NHWC.
But the NXP driver stack expects NCHW input at the delegate boundary; it handles the layout conversion internally via the tflite metadata.
Feeding NHWC bytes into a model expecting NCHW scrambles the spatial dimensions. As alwas, garbage in, garbage out.
In hindisght of course obvious but you'll have to be completely aware of the importance of this format.

Fixing `channels_last` worked, great, then the reduction ops crashed. Delegated subgraphs return tensors with `channels_last` `dim_order` metadata.
The portable CPU fallback kernels for `mean`, `sum`, and `var` had `tensor_is_default_dim_order` checks that reject non-contiguous dim orders.
In a fully-delegated model this doesn't matter. Models like MobileNetV2, though, have a global average pooling layer that falls back
to CPU after the final delegated conv block. To fix this I relaxed the strict input-output-order relation in the Arm infra.

Then there's MobileNetV3. HardSwish and HardSigmoid both work perfectly as standalone ops on the NPU. But MobileNetV3's architecture
composes them into multi-input delegate segments, ie. `HardSwish = x * HardSigmoid(x)` which creates a segment with two input tensors.
The NXP M33 firmware's IOCTL handler fails on these segments. Not sure what's going on here, maybe some difference in runtimes?

Vela has its own memory model.`Shared_Sram` mode places the NPU's scratch arena on AXI0/SRAM. The i.MX93 has only 96KB
of SRAM accessible to the NPU. Models that need more scratch than 96KB silently produce wrong results or fail to run.
You can work around this with `Dedicated_Sram` mode and `--arena-cache-size 98304`. 

## Setup

- **Board**: NXP i.MX93 EVK (MCIMX93-EVK)
- **NPU**: Ethos-U65-256 (256 MACs/cycle, 1 GHz clock)
- **Architecture**: Cortex-A55 (Linux) → `/dev/ethosu0` → kernel driver → rpmsg → Cortex-M33 firmware → NPU
- **Software**: Executorch with Arm TOSA backend, Vela compiler, NXP ethos-u-driver-stack-imx
- **Quantization**: int8 symmetric per-channel weights, per-tensor activations (PT2E)
- **Vela config**: `Ethos_U65_High_End`, `Dedicated_Sram`, arena cache 96KB


## A few models

Four ImageNet classification models, with several executions each. Timing from the NPU hardware cycle counter at 1 GHz.
Results are compared to the ground truth from host (also PTQ'd).

<img src="/plots/blog_models.png">

Every working model achieves top-1 argmax agreement with the host, nice.
MobileNetV3 fails due to an NXP firmware limitation on multi-input delegate segments. I think it's this fact: HardSwish = x * HardSigmoid(x).


### Are we memory-bound?

The Ethos-U65-256 has 256 MACs/cycle. At 1 GHz, that's 256 GMACs/s (or [512 GOPs in Arm's convention](https://documentation-service.arm.com/static/6821edae8f79851ff2c3e485) where 1 MAC = 2 ops). To keep the MAC array fully utilized, you need to feed it 256 int8 weights + 256 int8 activations per cycle = 512 bytes/cycle = 512 GB/s of data. Even with Vela's weight compression the decompressed data demand is 128-256 GB/s.

The i.MX93's LPDDR4 delivers about 2 GB/s to the NPU through [two 128-bit AXI interfaces](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/arm-ethos-u65-powering-innovation-in-a-new-world-of-ai-devices). That's a 250:1 mismatch if I think right between what the engine can consume and what the memory system can deliver.

We measured this directly using the i.MX93's DDR performance counters (`imx9_ddr0` perf events via `perf stat`). During 100 MobileNetV2 inferences,
the DDR PMU reports 181M read beats. On the 64-bit LPDDR4 bus, that's 1.45 GB transferred in 1.18 seconds: _1.23 GB/s sustained DRAM bandwidth_ during NPU inference, or 29% of the ~4.3 GB/s theoretical DRAM peak.

Each MobileNetV2 inference transfers 14.5 MB over the DRAM bus. The model weights are 6.7 MB and the input is 0.6 MB — the remaining ~7 MB is intermediate activations being written and read back between layers. This is the fundamental cost of the Ethos-U65 architecture: no hardware operator chaining, so every layer's output round-trips through DRAM before the next layer can consume it.

This is a known limitation. Arm redesigned the architecture in the [Ethos-U85](https://newsroom.arm.com/blog/ethos-u85), which adds up to six 128-bit AXI ports (vs two on U65) and hardware operator chaining that avoids intermediate memory round-trips — claiming "up to 85% utilization on popular networks." The fact that a new silicon generation was needed to reach 85% tells you the U55/U65 were architecturally constrained well below that.

Vela's own [PERFORMANCE.md](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/raw/5.0.0/PERFORMANCE.md) warns: *"The cycle and bandwidth numbers should not be taken as accurate representations of real performance numbers."* Our measured 14.5% utilization is the first published number from real hardware that quantifies what that warning actually means in practice.

No software power measurement is available — the PCA9451A PMIC on the EVK has no current sensing, and the USB-C PD monitor reads zero (board uses barrel jack). An external shunt resistor or Joulescope would be needed for energy-per-inference numbers.

Numerical accuracy for YOLOv8n-seg at 640x640: detection outputs achieve 96.4% int8 exact match with cosine 0.998. Prototype masks (spatial output) show 4.7% exact match but cosine 0.872 — expected for high-resolution spatial predictions where single-step quantization errors compound across the decoder.

InceptionV3 is 3x faster than ResNet-50 despite having more parameters. The architecture has more parallel branches and fewer deep sequential bottlenecks, which maps better to the NPU's dataflow engine.

MobileNetV3 does NOT work. Its HardSwish activation creates a multi-input delegate segment that triggers an IOCTL failure in the NXP M33 firmware. HardSwish and HardSigmoid work fine as standalone ops — the failure is specific to how the NXP firmware handles multi-tensor-input command stream segments. This is an NXP firmware limitation, not an Executorch or Vela issue.

## Probing operators

We tested 32 operators individually at tensor sizes 16 and 32 (shape [1, 8, S, S] for 4D ops). Each is exported, quantized, compiled through Vela, executed on the NPU, and compared against the host-quantized reference.

<img src="/plots/blog_ops.png">

28 out of 32 ops pass at size 32. The heatmap shows two clear patterns:

**The size boundary.** Many ops that fail at size 16 (orange cells) pass at size 32 (green). Tensors below ~2048 elements don't have enough distinct values for reliable int8 PTQ calibration. This is a quantization limitation, not a hardware bug. At size 32 (8192 elements), everything except `minimum` works.

**TABLE ops are less precise.** Activations implemented via 256-entry lookup tables (sigmoid, tanh, exp, log, rsqrt, ceil, floor) show cosine 0.91–0.97 — lower than arithmetic ops (add, sub, mul) which hit 0.97–1.00. The lookup table quantizes the nonlinearity itself, adding approximation error on top of the int8 tensor quantization.

Three ops don't delegate at all: `eq`, `gt`, `le` produce bool tensors that can't be int8-quantized. And `minimum` is broken at all sizes — likely a Vela or firmware bug.


## Methodology

All measurements follow the same protocol:

1. Export model on host with `aot_arm_compiler` (int8 PTQ, TOSA target, Vela compile)
2. Generate host-quantized reference by running the quantized model on the same input
3. SCP PTE + input binary to device
4. Run `executor_runner` with `--num_executions=N`
5. SCP output binary back to host
6. Compare device output against host reference: cosine similarity, RMSE, argmax agreement

The input for each model is deterministic (from Executorch's `EagerModelFactory`). No ImageNet validation set was used — this is a numerical fidelity test, not an accuracy benchmark.

Tools: `device_numerics_test.py` for model benchmarks, `op_sweep.py` for operator microbenchmarks, `run_benchmarks.py` for multi-execution timing. All in `examples/arm/imx93/` in the Executorch fastpath branch.

## Resume

The Ethos-U65-256 on i.MX93 is a real, usable inference accelerator. MobileNetV2 at 114 FPS, ResNet-18 at 44 FPS, InceptionV3 at 29 FPS — all with cosine > 0.98 against the quantized reference. 28 out of 32 tested operators work correctly at practical tensor sizes. The hardware is deterministic (CV < 0.4%), the quantization is faithful, and the Executorch integration is functional.

But you will hit undocumented walls. channels_last breaks silently. MobileNetV3 won't work. `minimum` is broken. Small tensors produce unreliable quantization. None of this is in Arm's or NXP's documentation. If you're deploying on this hardware, you need to validate, not assume.
