# Outline

Outline is a background remover with a flexible mask-processing pipeline and a built-in SVG tracing workflow.

It is written in Rust, powered by ONNX Runtime (ort) and VTracer, and works with U2-Net, BiRefNet, and other ONNX models that share a similar input/output shape.

> This project is still in early development. Breaking changes may occur in future releases.

## Installation

```bash
cargo install outline-core --version 0.1.0-alpha.1
```

## Usage

Outline supports being used as a library or via a command-line interface (CLI).

Before using, specify your ONNX model path:

- CLI flag: `-m, --model <path>`
- Library API: `Outline::new(<path>)`
- Environment variable: set `OUTLINE_MODEL_PATH`

Resolution order: user value > environment variable > default (`model.onnx`).

### CLI Usage

Use `outline <COMMAND> --help` to inspect the options of each subcommand.

#### Examples

```bash
# Remove the background and export a foreground PNG
outline cut input.jpg -o subject.png

# Export a foreground PNG with feathered edges
outline cut input.jpg --blur -o subject-soft-alpha.png

# Export the raw matte png
outline mask input.jpg -o subject-matte.png

# Export a processed binary mask png
outline mask input.jpg --binary -o subject-mask.png

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
- `mask`: Exports only the mask. It saves the raw matte by default and switches to the processed binary mask when `--binary` is provided.
- `trace`: Generates an SVG outline using the same mask-processing pipeline. Exposes VTracer color modes, hierarchy selection, path precision, and other options.

#### Global Options (shared by all subcommands)

- `-m, --model <path>`: Path to the ONNX model (defaults to `model.onnx`).
- `--intra-threads <n>`: ORT intra-op thread count. Omit to let ORT decide.
- `--input-resample-filter {nearest,triangle,catmull-rom,gaussian,lanczos3}`: Resampling filter used when scaling the input down to the model resolution.
- `--output-resample-filter {nearest,triangle,catmull-rom,gaussian,lanczos3}`: Filter used to resize the matte back to the original image size.

#### Shared Mask-Processing Options

The following switches can be used in `mask`, `cut`, and `trace`:

- `--blur [sigma]`: Apply Gaussian blur before thresholding (defaults to `6.0` when no value is provided).
- `--mask-threshold <0-255 | 0.0-1.0>`: Threshold applied to the matte (default `120`) to produce a binary mask.
- `--binary [enabled|disabled|auto]`: Control thresholding in the processed pipeline (use the flag without a value to force a binary mask, or pass `--binary disabled` to keep soft edges; the default `auto` preserves the raw matte unless other processing steps require a hard mask).
- `--dilate [radius]`: Enable dilation after thresholding (defaults to `5.0` when no value is provided).
- `--fill-holes`: Fill enclosed holes before exporting/tracing.

The raw matte (soft mask) preserves the grayscale alpha predicted by the model. The processed mask (binary mask) goes through the blur/threshold/dilate/fill pipeline, producing a clean silhouette for tracing or foreground compositing.

#### `cut` Command

- `-o, --output <path>`: Foreground PNG output path (default `<name>-foreground.png`).
- `--export-matte [path]`: Additionally save the raw matte (default `<name>-matte.png`).
- `--export-mask [path]`: Save the processed mask (default `<name>-mask.png`); add `--binary` for hard edges.
- `--alpha-source {raw|processed|auto}`: Choose which mask becomes the PNG alpha (default `auto`). `auto` keeps the raw matte unless any mask-processing options are provided, in which case it uses the processed mask.

#### `mask` Command

- `-o, --output <path>`: Output path (default `<name>-matte.png` or `<name>-mask.png` depending on processing flags).
- `--mask-source {raw|processed|auto}`: Choose which mask to export. `auto` (default) exports the raw matte unless any mask-processing options (binary/blur/dilate/fill-holes or a custom threshold) are provided, in which case it exports the processed mask.

#### `trace` Command

- `-o, --output <path>`: SVG output path (default is the input name with `.svg`).
- `--mask-source {raw|processed|auto}`: Choose the mask used for tracing. `auto` (default) exports the raw matte unless any mask-processing options are enabled, in which case it exports the processed mask.
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

```bash
cargo add outline-core --version 0.1.0-alpha.1
# or use this if VtracerSvgVectorizer is needed:
cargo add outline-core --version 0.1.0-alpha.1 --features vectorizer-vtracer
```

```rust
use outline::{MaskProcessingOptions, Outline, TraceOptions, VtracerSvgVectorizer};

fn generate_assets() -> outline::OutlineResult<()> {
	let outline = Outline::new("model.onnx") // or: Outline::try_from_env()
		.with_default_mask_processing(MaskProcessingOptions::default());
	let session = outline.for_image("input.png")?;
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

	let vectorizer = VtracerSvgVectorizer;
	let svg = mask.trace(&vectorizer, &TraceOptions::default())?;
	std::fs::write("input.svg", svg)?;

	Ok(())
}
```

> VtracerSvgVectorizer is only available when the `vectorizer-vtracer` feature is enabled. You can use your own vectorizer by implementing the `SvgVectorizer` trait to avoid depending on VTracer directly.

## Next Steps

- Add detailed documentation for library API
- Expose a WASM version of library API
- Improve the CLI syntax for better usability and expressiveness
- Provide a lightweight GUI for easier use.

## License

Distributed under the MIT License. See `LICENSE` for details.
