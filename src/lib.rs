use cfg_if::cfg_if;
use thiserror::Error;

pub mod core {
    use super::{Error, cfg_if};
    use realfft::num_complex::Complex;
    use realfft::{RealFftPlanner, RealToComplex};

    #[derive(Debug, Clone, Copy)]
    pub struct SpectrogramConfig {
        pub sample_rate: u32,
        pub fft_size: usize,
        pub hop_length: usize,
        pub img_width: usize,
        pub img_height: usize,
        pub gain: f32,
    }

    #[derive(Error, Debug, Clone, PartialEq, Eq)]
    pub enum SpectrogramError {
        #[error("Audio data is too short to produce a spectrogram.")]
        AudioTooShort,
        #[error("The number of frequency bins is zero, cannot process.")]
        NoFrequencyBins,
    }

    pub fn calculate_spectrogram(
        audio_data: &[f32],
        fft: &dyn RealToComplex<f32>,
        fft_size: usize,
        hop_length: usize,
        num_freq_bins_to_render: usize,
    ) -> (Vec<f32>, usize) {
        let num_time_bins = if audio_data.len() >= fft_size {
            (audio_data.len() - fft_size) / hop_length + 1
        } else {
            0
        };
        if num_time_bins == 0 {
            return (Vec::new(), 0);
        }

        struct SpectrogramProcessor<'a> {
            fft: &'a dyn RealToComplex<f32>,
            audio_data: &'a [f32],
            fft_size: usize,
            hop_length: usize,
            num_freq_bins_to_render: usize,
        }

        impl SpectrogramProcessor<'_> {
            fn process_time_bin(
                &self,
                time_index: usize,
                time_bin_slice: &mut [f32],
                real_input: &mut [f32],
                complex_output: &mut [Complex<f32>],
            ) {
                let current_pos = time_index * self.hop_length;
                let audio_chunk = &self.audio_data[current_pos..current_pos + self.fft_size];

                real_input.copy_from_slice(audio_chunk);
                self.fft.process(real_input, complex_output).unwrap();

                time_bin_slice
                    .iter_mut()
                    .zip(complex_output[0..self.num_freq_bins_to_render].iter())
                    .for_each(|(mag_slot, complex_val)| {
                        *mag_slot = complex_val.norm();
                    });
            }
        }

        let processor = SpectrogramProcessor {
            fft,
            audio_data,
            fft_size,
            hop_length,
            num_freq_bins_to_render,
        };

        let mut flat_spectrogram = vec![0.0f32; num_time_bins * num_freq_bins_to_render];

        cfg_if! {
            if #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-parallel"))] {
                use rayon::prelude::*;

                flat_spectrogram
                    .par_chunks_mut(num_freq_bins_to_render)
                    .enumerate()
                    .for_each_init(
                        || (fft.make_input_vec(), fft.make_output_vec()),
                        |buffers, (i, time_bin_slice)| {
                            processor.process_time_bin(i, time_bin_slice, &mut buffers.0, &mut buffers.1);
                        },
                    );
            } else {
                let mut real_input = fft.make_input_vec();
                let mut complex_output = fft.make_output_vec();

                flat_spectrogram
                    .chunks_mut(num_freq_bins_to_render)
                    .enumerate()
                    .for_each(|(i, time_bin_slice)| {
                        processor.process_time_bin(i, time_bin_slice, &mut real_input, &mut complex_output);
                    });
            }
        }

        (flat_spectrogram, num_time_bins)
    }

    pub fn apply_gain_mapping(spectrogram: &mut [f32], fft_size: usize, gain: f32) {
        let n = fft_size as f32;
        let scale_factor = 9.0 / (2.0 * n).sqrt();

        let mapping_op = |v: &mut f32| {
            let scaled_mag = *v * scale_factor;
            let log_val = scaled_mag.mul_add(1.0, 1.0).log10();
            *v = log_val * gain;
        };

        cfg_if! {
            if #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-parallel"))] {
                use rayon::prelude::*;
                spectrogram.par_iter_mut().for_each(mapping_op);
            } else {
                spectrogram.iter_mut().for_each(mapping_op);
            }
        }
    }

    pub struct RenderableSpectrogram {
        pub values: Vec<f32>,
        pub num_time_bins: usize,
        pub num_freq_bins: usize,
    }

    pub fn process_audio_to_spectrogram(
        audio_data: &[f32],
        sample_rate: u32,
        fft_size: usize,
        hop_length: usize,
        gain: f32,
    ) -> Result<RenderableSpectrogram, SpectrogramError> {
        let mut planner = RealFftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let max_freq = 20000.0;
        let freq_resolution = sample_rate as f32 / fft_size as f32;
        let num_freq_bins = (max_freq / freq_resolution).round() as usize;
        let num_freq_bins = num_freq_bins.min(fft_size / 2);

        if num_freq_bins == 0 {
            return Err(SpectrogramError::NoFrequencyBins);
        }

        let (mut values, num_time_bins) =
            calculate_spectrogram(audio_data, &*fft, fft_size, hop_length, num_freq_bins);

        if num_time_bins == 0 {
            return Err(SpectrogramError::AudioTooShort);
        }

        apply_gain_mapping(&mut values, fft_size, gain);

        Ok(RenderableSpectrogram {
            values,
            num_time_bins,
            num_freq_bins,
        })
    }

    pub fn render_pixel_row(
        y_logical: usize,
        img_width: usize,
        img_height: usize,
        spectrogram: &RenderableSpectrogram,
        row_buffer: &mut [u8],
        log_ratio: f32,
        palette: &[u8],
    ) {
        let min_bin = 1;
        let max_bin = spectrogram.num_freq_bins;

        if max_bin <= min_bin {
            let color = &palette[0..4];
            for x in 0..img_width {
                let pixel_start_index = x * 4;
                row_buffer[pixel_start_index..pixel_start_index + 4].copy_from_slice(color);
            }
            return;
        }

        let min_bin_f = min_bin as f32;
        let max_bin_f = max_bin as f32;
        let scale_log = (max_bin_f / min_bin_f).ln();

        let map_y_to_bin = |y: usize| -> f32 {
            let pos_rel = y as f32 / img_height as f32;
            let b_lin = pos_rel.mul_add(max_bin_f - min_bin_f, min_bin_f);
            let b_log = min_bin_f * (pos_rel * scale_log).exp();
            log_ratio.mul_add(b_log - b_lin, b_lin)
        };

        let bin_start_f = map_y_to_bin(y_logical);
        let bin_end_f = map_y_to_bin(y_logical + 1);

        let bin_start = bin_start_f.floor() as usize;
        let bin_end = bin_end_f.floor() as usize;

        let bin_start = bin_start.min(spectrogram.num_freq_bins - 1);
        let bin_end = bin_end.min(spectrogram.num_freq_bins);

        for x in 0..img_width {
            let time_index = (x as f32 / img_width as f32 * (spectrogram.num_time_bins - 1) as f32)
                .round() as usize;
            let time_bin_start_index = time_index * spectrogram.num_freq_bins;
            let time_bin = &spectrogram.values
                [time_bin_start_index..time_bin_start_index + spectrogram.num_freq_bins];

            let final_value = if bin_start >= bin_end {
                time_bin[bin_start.min(time_bin.len().saturating_sub(1))]
            } else {
                let freq_slice = &time_bin[bin_start..bin_end];
                freq_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            };

            let color_index = (final_value.clamp(0.0, 1.0) * 255.0).round() as usize;

            let color_start_index = color_index * 4;
            let pixel_start_index = x * 4;

            if color_start_index + 4 <= palette.len() {
                let color_slice = &palette[color_start_index..color_start_index + 4];
                row_buffer[pixel_start_index..pixel_start_index + 4].copy_from_slice(color_slice);
            } else {
                let color_slice = &palette[0..4];
                row_buffer[pixel_start_index..pixel_start_index + 4].copy_from_slice(color_slice);
            }
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-parallel"))]
    #[must_use]
    pub fn render_pixels_parallel(
        spectrogram: &RenderableSpectrogram,
        img_width: usize,
        img_height: usize,
        log_ratio: f32,
        palette: &[u8],
    ) -> Vec<u8> {
        use rayon::prelude::*;
        let mut pixels = vec![0u8; img_width * img_height * 4];
        pixels
            .par_chunks_mut(img_width * 4)
            .enumerate()
            .for_each(|(y_pixel, row_slice)| {
                let y_logical = img_height - 1 - y_pixel;
                render_pixel_row(
                    y_logical,
                    img_width,
                    img_height,
                    spectrogram,
                    row_slice,
                    log_ratio,
                    palette,
                );
            });
        pixels
    }

    #[must_use]
    pub fn render_pixels_serial(
        spectrogram: &RenderableSpectrogram,
        img_width: usize,
        img_height: usize,
        log_ratio: f32,
        palette: &[u8],
    ) -> Vec<u8> {
        let mut pixels = vec![0u8; img_width * img_height * 4];

        for y_pixel in 0..img_height {
            let y_logical = img_height - 1 - y_pixel;

            let row_start_index = y_pixel * img_width * 4;
            let row_end_index = row_start_index + img_width * 4;
            let row_buffer = &mut pixels[row_start_index..row_end_index];

            render_pixel_row(
                y_logical,
                img_width,
                img_height,
                spectrogram,
                row_buffer,
                log_ratio,
                palette,
            );
        }
        pixels
    }

    const LOG_RATIO_POS_FREF: f32 = 0.001;
    const LOG_RATIO_FREQ_REF: f32 = 1000.0;

    #[must_use]
    pub fn calculate_log_ratio(
        sample_rate: u32,
        fft_size: usize,
        img_height: usize,
        num_freq_bins: usize,
    ) -> f32 {
        let min_bin = 1;
        let max_bin = num_freq_bins;

        if max_bin <= min_bin || img_height == 0 {
            return 0.0;
        }

        let freq_resolution = sample_rate as f32 / fft_size as f32;
        let b_fref =
            (LOG_RATIO_FREQ_REF / freq_resolution).clamp(min_bin as f32, (max_bin - 1) as f32);

        let min_bin_f = min_bin as f32;
        let max_bin_f = max_bin as f32;

        let clin = (max_bin_f - min_bin_f).mul_add(LOG_RATIO_POS_FREF, min_bin_f);
        let scale_log = (max_bin_f / min_bin_f).ln();
        let clog = min_bin_f * (LOG_RATIO_POS_FREF * scale_log).exp();

        let denominator = clog - clin;
        if denominator.abs() < 1e-6 {
            0.5
        } else {
            ((b_fref - clin) / denominator).clamp(0.0, 1.0)
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub mod wasm_api {
    use super::cfg_if;
    use super::core::{self, SpectrogramError};
    use wasm_bindgen::prelude::*;

    impl From<SpectrogramError> for JsValue {
        fn from(err: SpectrogramError) -> Self {
            js_sys::Error::new(&err.to_string()).into()
        }
    }

    #[cfg(feature = "wasm-parallel")]
    #[wasm_bindgen]
    pub fn init_thread_pool(num_threads: usize) -> js_sys::Promise {
        wasm_bindgen_rayon::init_thread_pool(num_threads)
    }

    #[wasm_bindgen]
    pub struct SpectrogramConfig {
        pub sample_rate: u32,
        pub fft_size: usize,
        pub hop_length: usize,
        pub img_width: usize,
        pub img_height: usize,
        pub gain: f32,
    }

    #[wasm_bindgen]
    impl SpectrogramConfig {
        #[wasm_bindgen(constructor)]
        #[must_use]
        #[allow(clippy::missing_const_for_fn)]
        pub fn new(
            sample_rate: u32,
            fft_size: usize,
            hop_length: usize,
            img_width: usize,
            img_height: usize,
            gain: f32,
        ) -> Self {
            Self {
                sample_rate,
                fft_size,
                hop_length,
                img_width,
                img_height,
                gain,
            }
        }
    }

    #[wasm_bindgen]
    pub fn generate_spectrogram_image(
        audio_data: &[f32],
        palette: &[u8],
        config: &SpectrogramConfig,
    ) -> Result<Vec<u8>, JsValue> {
        let spectrogram_data = core::process_audio_to_spectrogram(
            audio_data,
            config.sample_rate,
            config.fft_size,
            config.hop_length,
            config.gain,
        )?;

        let log_ratio = core::calculate_log_ratio(
            config.sample_rate,
            config.fft_size,
            config.img_height,
            spectrogram_data.num_freq_bins,
        );

        cfg_if! {
            if #[cfg(feature = "wasm-parallel")] {
                Ok(core::render_pixels_parallel(
                    &spectrogram_data,
                    config.img_width,
                    config.img_height,
                    log_ratio,
                    palette
                ))
            } else {
                Ok(core::render_pixels_serial(
                    &spectrogram_data,
                    config.img_width,
                    config.img_height,
                    log_ratio,
                    palette
                ))
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod native_api {
    use super::core::{self, SpectrogramConfig, SpectrogramError};

    pub fn generate_spectrogram_image_native(
        audio_data: &[f32],
        palette: &[u8],
        config: SpectrogramConfig,
    ) -> Result<Vec<u8>, SpectrogramError> {
        let spectrogram_data = core::process_audio_to_spectrogram(
            audio_data,
            config.sample_rate,
            config.fft_size,
            config.hop_length,
            config.gain,
        )?;

        let log_ratio = core::calculate_log_ratio(
            config.sample_rate,
            config.fft_size,
            config.img_height,
            spectrogram_data.num_freq_bins,
        );

        Ok(core::render_pixels_parallel(
            &spectrogram_data,
            config.img_width,
            config.img_height,
            log_ratio,
            palette,
        ))
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::core::SpectrogramConfig;
    use super::native_api;
    use std::time::Instant;

    const NUM_RUNS: u32 = 100;
    const SAMPLE_RATE: u32 = 48000;
    const DURATION_SECONDS: u32 = 5;
    const FFT_SIZE: usize = 1024;
    const HOP_LENGTH: usize = 128;
    const IMG_WIDTH: usize = 1024;
    const IMG_HEIGHT: usize = 256;
    const GAIN: f32 = 9.0;

    fn generate_test_audio() -> Vec<f32> {
        let num_samples = (SAMPLE_RATE * DURATION_SECONDS) as usize;
        let freq = 440.0;
        let audio_data: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (t * freq * 2.0 * std::f32::consts::PI).sin() * 0.5
            })
            .collect();
        audio_data
    }

    #[test]
    fn benchmark_generation() {
        let audio_data = generate_test_audio();

        let mut dummy_palette = vec![0u8; 256 * 4];
        for i in 0..256 {
            let val = i as u8;
            dummy_palette[i * 4] = val;
            dummy_palette[i * 4 + 1] = val;
            dummy_palette[i * 4 + 2] = val;
            dummy_palette[i * 4 + 3] = 255;
        }

        let config = SpectrogramConfig {
            sample_rate: SAMPLE_RATE,
            fft_size: FFT_SIZE,
            hop_length: HOP_LENGTH,
            img_width: IMG_WIDTH,
            img_height: IMG_HEIGHT,
            gain: GAIN,
        };

        let start_time = Instant::now();

        for _ in 0..NUM_RUNS {
            let _pixels =
                native_api::generate_spectrogram_image_native(&audio_data, &dummy_palette, config)
                    .unwrap();
        }

        let total_duration = start_time.elapsed();
        let avg_duration = total_duration / NUM_RUNS;

        println!("Total duration: {total_duration:?}");
        println!("Average time per run: {avg_duration:?}");
    }
}
