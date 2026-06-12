# Outline

Outline is an image background removal tool with flexible mask processing options.

It is written in Rust, powered by [ONNX Runtime](https://onnxruntime.ai/) ([ort](https://ort.pyke.io/)) and VTracer, and works with U2-Net, BiRefNet, and other ONNX models with a compatible input/output shape.

> This project is still in early development. Breaking changes may occur in future releases.

## Installation

### CLI
```bash
cargo install outline-core --features cli
```

For the CLI, you can also install with `--features fetch-model` to enable one-command model downloading:

```bash
cargo install outline-core --features "cli fetch-model"
outline fetch-model
```

### Library
```bash
cargo add outline-core
# or use this if VtracerSvgVectorizer is needed:
cargo add outline-core --features vectorizer-vtracer
```

<details>
<summary><strong>Advanced: Custom ONNX Runtime Setup</strong></summary>

Most users do not need this section.
By default, `outline-core` enables `backend-ort` and `ort-download-binaries`, so `ort`
downloads a prebuilt ONNX Runtime package for supported targets.
In some environments, the prebuilt runtime may run into
compatibility issues.

If your environment needs a different runtime strategy, `outline-core` exposes the supported
non-default paths directly. These features are additive. If `ort-load-dynamic` is enabled,
`ort` skips build-time linking and loads ONNX Runtime at runtime. Otherwise, build-time
linking uses the enabled inputs in order: `ort-pkg-config`, `ORT_LIB_LOCATION`, then
`ort-download-binaries`.

```bash
# Discover a system installation via pkg-config
cargo add outline-core --no-default-features --features ort-pkg-config

# Dynamically load a specific .so/.dylib/.dll at runtime
cargo add outline-core --no-default-features --features ort-load-dynamic

# Link against a custom ONNX Runtime build from a known directory
cargo add outline-core --no-default-features --features backend-ort
ORT_LIB_LOCATION=/opt/onnxruntime/lib cargo build

# Prefer shared-library linking for a custom build
ORT_LIB_LOCATION=/opt/onnxruntime/lib ORT_PREFER_DYNAMIC_LINK=1 cargo build
```

`--no-default-features` also disables the default backend selection. Use it together with
`backend-ort` plus one ONNX Runtime strategy.

For `ort-load-dynamic`, initialize ONNX Runtime before using `Outline`, or set
`ORT_DYLIB_PATH` before the first ORT API use:

```rust
let filename = format!(
    "{}onnxruntime{}",
    std::env::consts::DLL_PREFIX,
    std::env::consts::DLL_SUFFIX
);
let committed = outline::runtime::init_onnx_runtime_from(format!("/opt/onnxruntime/lib/{filename}"))?;
assert!(committed);
```

`outline-core` exposes the relevant environment variable names as
`outline::runtime::ENV_ORT_DYLIB_PATH`,
`outline::runtime::ENV_ORT_LIB_LOCATION`, and
`outline::runtime::ENV_ORT_PREFER_DYNAMIC_LINK`.
See the upstream `ort` linking guide for platform-specific details:
https://ort.pyke.io/setup/linking

If you want to avoid ONNX Runtime entirely, there is also an experimental pure-Rust backend option based on RTen. See the next section for details.

</details>

<details>
<summary><strong>Experimental: Pure-Rust Alternative to ONNX Runtime</strong></summary>

The pure-Rust [RTen](https://github.com/robertknight/rten) backend is another option if you want to avoid ONNX Runtime compatibility issues.

It is still experimental: model/operator compatibility is
narrower than ORT, and inference can be slower. It has been tested with several models, but validate it with your own before use.

Enable the `backend-rten` feature to use the RTen backend:

```bash
# Library
cargo add outline-core --no-default-features --features backend-rten

# CLI
cargo install outline-core --no-default-features --features "cli backend-rten"
```

Notice that there is no implicit fallback between backends; if `backend-ort` and `backend-rten` are both enabled, ORT is still selected by default. You can select the RTen backend explicitly with `Outline::with_backend(InferenceBackend::Rten)`.

</details>

## Usage

Outline works as both a library and a CLI.

Before using Outline, specify your ONNX model path:

- CLI flag: `-m, --model <path>`
- Library API: `Outline::new(<path>)`
- Environment variable: set `OUTLINE_MODEL_PATH`

If you don't have a model file yet, download one first: [silueta.onnx](https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx).

Resolution order: user value > environment variable > cached model (CLI only with `fetch-model` enabled) > default (`model.onnx`).

### CLI Usage

Use `outline <COMMAND> --help` to inspect the options of each subcommand.

#### Examples

```bash
# Remove the background and export a foreground PNG
outline cut input.jpg -o subject.png

# Export a foreground PNG with feathered edges
outline cut input.jpg --blur -o subject-soft-alpha.png

# Export the raw matte PNG
outline mask input.jpg -o subject-matte.png

# Export a processed binary mask png
outline mask input.jpg --threshold -o subject-mask.png

# Generate an SVG outline from the raw matte
outline trace input.jpg -o subject-raw-mask.svg

# Generate an SVG outline with sticker-style processing
outline trace input.jpg -o subject.svg \
	--dilate 50.0 --fill-holes --blur 20.0
```

<details>
<summary><strong>Detailed CLI Reference</strong></summary>

#### Subcommands

- `cut`: Primary background-removal workflow. Produces a foreground PNG, optionally saves the raw matte and the processed mask, and lets you choose the alpha source.
- `mask`: Exports only the mask. It saves the raw matte by default and switches to the processed mask when mask-processing options are provided.
- `trace`: Generates an SVG outline using the same mask-processing pipeline. Exposes VTracer color modes, hierarchy selection, path precision, and other options.

#### Global Options (shared by all subcommands)

- `-m, --model <path>`: Path to the ONNX model (defaults to `model.onnx`).
- `--model-input-size <HEIGHTxWIDTH>`: Override the model input size when it cannot be inferred from the ONNX graph.
- `--intra-threads <n>`: ORT intra-op thread count. Omit to let ORT decide; ignored by RTen.
- `--input-resample-filter {nearest,triangle,catmull-rom,gaussian,lanczos3}`: Resampling filter used when scaling the input down to the model resolution.
- `--output-resample-filter {nearest,triangle,catmull-rom,gaussian,lanczos3}`: Filter used to resize the matte back to the original image size.

#### Shared Mask-Processing Options

The following switches can be used in `mask`, `cut`, and `trace`:

- `--blur [sigma]`: Apply Gaussian blur (defaults to `6.0` when no value is provided).
- `--threshold [0-255 | 0.0-1.0]`: Threshold the matte at this point (defaults to `120` when no value is provided).
- `--no-implicit-threshold`: Disable implicit `--threshold`; require one before hard-mask operations.
- `--dilate [radius]`: Enable dilation (defaults to `5.0` when no value is provided).
- `--erode [radius]`: Enable erosion (defaults to `5.0` when no value is provided).
- `--erode-border {outside-is-background,outside-is-unknown}`: Choose how erosion treats pixels outside the image bounds. The default `outside-is-background` lets edge-touching foreground shrink; `outside-is-unknown` preserves the visible image boundary.
- `--fill-holes [0-255 | 0.0-1.0]`: Fill enclosed holes (defaults to `120` when no value is provided).

Mask-processing options run in command-line order. `--dilate`, `--erode`, and `--fill-holes` need a hard mask; by default, `outline` inserts `--threshold` before them when needed. Currently, repeated mask-processing options are not supported; each option is ordered by its first occurrence.

#### `cut` Command

- `-o, --output <path>`: Foreground PNG output path (default `<name>-foreground.png`).
- `--export-matte [path]`: Additionally save the raw matte (default `<name>-matte.png`).
- `--export-mask [path]`: Save the processed mask (default `<name>-mask.png`).
- `--alpha-source {raw|processed|auto}`: Choose which mask becomes the PNG alpha (default `auto`). `auto` keeps the raw matte unless any mask-processing options are provided, in which case it uses the processed mask.

#### `mask` Command

- `-o, --output <path>`: Output path (default `<name>-matte.png` or `<name>-mask.png` depending on processing flags).
- `--mask-source {raw|processed|auto}`: Choose which mask to export. `auto` (default) exports the raw matte unless any mask-processing options are provided, in which case it exports the processed mask.

#### `trace` Command

- `-o, --output <path>`: SVG output path (default is the input name with `.svg`).
- `--mask-source {raw|processed|auto}`: Choose the mask used for tracing. `auto` (default) uses the raw matte unless any mask-processing options are enabled, in which case it uses the processed mask.
- `--color-mode {color,binary}`: Color mode (default `binary`).
- `--hierarchy {stacked,cutout}`: Hierarchy strategy (default `stacked`).
- `--mode {none,polygon,spline}`: Path simplification mode (default `spline`).
- `--invert-svg`: Invert foreground/background in the SVG output.

<details>
<summary>Other VTracer related options</summary>
<br>

- `--filter-speckle <usize>`: Speckle filter size (default `4`).
- `--color-precision <i32>` / `--layer-difference <i32>` / `--corner-threshold <i32>` / `--length-threshold <float>` / `--max-iterations <usize>` / `--splice-threshold <i32>`: Fine-tune the remaining VTracer parameters (defaults: `6`, `16`, `60`, `4.0`, `10`, `45`).
- `--path-precision <u32>`: Decimal precision for path coordinates (default `2`).
- `--no-path-precision`: Clear the explicit path precision override and defer to VTracer's internal behaviour.

</details>

</details>

### Library Usage

```rust
use outline::Outline;

fn generate_assets() -> outline::OutlineResult<()> {
	let outline = Outline::new("model.onnx"); // or: Outline::try_from_env()
	let session = outline.for_image("input.png")?; // or: outline.for_image_bytes(&bytes)?
	let matte = session.matte();

	// Compose the foreground directly from the raw matte (soft edges)
	let foreground = matte.foreground()?;
	foreground.save("input-foreground.png")?;

	// Declare the desired mask steps, then apply them with `processed`.
	let mask = matte
		.clone()
		.blur_with(6.0)
		.threshold_with(120)
		.processed()?;
	mask.save("input-mask.png")?;

	// Compose the foreground with processed mask (hard edges)
	let foreground_processed = mask.foreground()?;
	foreground_processed.save("input-foreground-processed.png")?;

	// Save a flat-color silhouette preview
	mask.colorize([255, 64, 160])
		.save("input-mask-silhouette.png")?;

	Ok(())
}
```

Use `Outline::with_model_input_size(height, width)` when you need to override the model input size. By default, `outline` tries to infer it from the ONNX graph. This can be useful when a model does not clearly declare its input shape.

#### Optional SVG Tracing

Enable `vectorizer-vtracer` if you want to trace masks into SVG:

```bash
cargo add outline-core --features vectorizer-vtracer
```

```rust
use outline::{Outline, TraceOptions, VtracerSvgVectorizer};

fn trace_mask() -> outline::OutlineResult<()> {
	let outline = Outline::new("model.onnx");
	let session = outline.for_image("input.png")?;
	let mask = session.matte().blur().threshold().processed()?;

	let vectorizer = VtracerSvgVectorizer;
	let svg = mask.trace(&vectorizer, &TraceOptions::default())?;
	std::fs::write("input.svg", svg)?;
	Ok(())
}
```

You can also avoid depending on VTracer directly by implementing the `MaskVectorizer` trait with your own vectorizer.

## Next Steps

- Add detailed documentation for library API
- Expose a WASM version of library API
- Improve the CLI syntax for better usability and expressiveness
- Provide a lightweight GUI for easier use.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
