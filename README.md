# Outline

Outline is a command-line background remover with a flexible mask-processing pipeline and a built-in SVG tracing workflow.  

It is written in Rust, powered by ONNX Runtime (ort) and VTracer, and works with U2-Net, BiRefNet, and similar ONNX models (place your model at the project root as `model.onnx`).

## Quick Start

```bash
# Remove the background and export a foreground PNG
outline cut input.jpg -o subject.png

# Export a processed binary mask
outline mask input.jpg --binary -o subject-mask.png

# Generate an SVG outline (photo preset + stacked hierarchy)
outline trace input.jpg -o subject.svg \
	--preset photo --color-mode color --hierarchy stacked
```

Run the CLI either directly or after building the release binary:

```bash
cargo run --release -- <command> <input> [options]
# or build once and reuse
cargo build --release
./target/release/outline <command> <input> [options]
```

## CLI Overview

Use `outline <COMMAND> --help` to inspect the options of each subcommand.

### Subcommands

- `cut`: Primary background-removal workflow. Produces a foreground PNG, optionally saves the raw matte and the processed mask, and lets you choose the alpha source.
- `mask`: Exports only the mask. It saves the raw matte by default and switches to the processed binary mask when `--binary` is provided.
- `trace`: Generates an SVG outline using the same mask-processing pipeline. Exposes VTracer presets, color modes, hierarchy selection, path precision, and more.

### Global Options (shared by all subcommands)

- `-m, --model <path>`: Path to the ONNX model (defaults to `model.onnx`).
- `--intra-threads <n>`: ORT intra-op thread count. Omit to let ORT decide.
- `--model-filter {nearest,triangle,catmull-rom,gaussian,lanczos3}`: Resampling filter used when scaling the input down to the model resolution.
- `--matte-filter {nearest,triangle,catmull-rom,gaussian,lanczos3}`: Filter used to resize the matte back to the original image size.

### Shared Mask-Processing Options

The following switches can be used in `mask`, `cut`, and `trace`:

- `--blur`: Apply Gaussian blur before thresholding.
- `--blur-sigma <float>`: Blur sigma (default `6.0`).
- `--mask-threshold <0-255>`: Threshold applied to the matte (default `120`).
- `--dilate`: Enable dilation after thresholding.
- `--dilation-radius <float>`: Dilation radius in pixels (default `5.0`).
- `--fill-holes`: Fill enclosed holes before exporting/tracing.

The raw matte (soft mask) preserves the grayscale alpha predicted by the model. The processed mask (binary mask) goes through the blur/threshold/dilate/fill pipeline, producing a clean silhouette for tracing or foreground compositing.

### `cut` Command

- `-o, --output <path>`: Foreground PNG output path (default `<name>-foreground.png`).
- `--save-mask [path]`: Additionally save the raw matte (default `<name>-matte.png`).
- `--save-processed-mask [path]`: Save the processed binary mask (default `<name>-mask.png`).
- `--alpha-from {raw|processed}`: Choose which mask becomes the PNG alpha (default `raw`).

### `mask` Command

- `-o, --output <path>`: Output path (default `<name>-matte.png` or `<name>-mask.png`).
- `--binary`: Export the processed binary mask; omit to save the raw matte.

### `trace` Command

- `-o, --output <path>`: SVG output path (default is the input name with `.svg`).
- `--mask-source {raw|processed|auto}`: Select the mask used for tracing (`auto` prefers the processed mask).
- `--preset {bw,poster,photo}`: VTracer preset (default `bw`).
- `--color-mode {color,binary}`: Color mode (default `binary`).
- `--hierarchy {stacked,cutout}`: Hierarchy strategy (default `cutout`).
- `--mode {none,polygon,spline}`: Path simplification mode (default `spline`).
- `--filter-speckle <usize>`: Speckle filter size (default `4`).
- `--path-precision <u32>` / `--no-path-precision`: Decimal precision for path coordinates (default `8`).
- `--invert-svg`: Invert foreground/background in the SVG output.

## Typical Workflows

```bash
# 1. Quick background removal
outline cut portrait.jpg -o portrait-foreground.png

# 2. Save both the raw matte and the processed mask
outline cut product.png -o product-foreground.png \
	--save-mask --save-processed-mask

# 3. Strengthen edges (blur + threshold + dilation) and use the processed mask as alpha
outline cut scene.jpg -o scene-foreground.png \
	--blur --blur-sigma 6 --mask-threshold 120 \
	--dilate --dilation-radius 5 --fill-holes \
	--alpha-from processed

# 4. Export only a binary mask
outline mask item.png --binary -o item-mask.png

# 5. Produce an SVG outline with color
outline trace logo.png -o logo.svg \
	--preset poster --color-mode color --hierarchy stacked
```

## Next Steps
- Expose a library API for integration into other projects.
- Provide a lightweight GUI for easier use.

## License
Distributed under the MIT License. See `LICENSE` for details.