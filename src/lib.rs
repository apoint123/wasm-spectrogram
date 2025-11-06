#[cfg(target_arch = "wasm32")]
use cfg_if::cfg_if;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::LazyLock;
use thiserror::Error;

pub mod core {
    use super::{Complex, Error, Fft, FftPlanner, LazyLock};
    use cfg_if::cfg_if;

    #[derive(Error, Debug, Clone, PartialEq, Eq)]
    pub enum SpectrogramError {
        #[error("Audio data is too short to produce a spectrogram.")]
        AudioTooShort,
        #[error("The number of frequency bins is zero, cannot process.")]
        NoFrequencyBins,
    }

    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [u8; 3] {
        if s == 0.0 {
            let gray = (l * 255.0) as u8;
            return [gray, gray, gray];
        }

        let chroma = (1.0 - 2.0f32.mul_add(l, -1.0).abs()) * s;
        let h_prime = h / 60.0;
        let second_component = chroma * (1.0 - (h_prime % 2.0 - 1.0).abs());
        let lightness_modifier = l - chroma / 2.0;

        let (r_prime, g_prime, b_prime) = match h_prime as u32 {
            0 => (chroma, second_component, 0.0),
            1 => (second_component, chroma, 0.0),
            2 => (0.0, chroma, second_component),
            3 => (0.0, second_component, chroma),
            4 => (second_component, 0.0, chroma),
            5 => (chroma, 0.0, second_component),
            _ => (0.0, 0.0, 0.0),
        };

        let r = ((r_prime + lightness_modifier) * 255.0) as u8;
        let g = ((g_prime + lightness_modifier) * 255.0) as u8;
        let b = ((b_prime + lightness_modifier) * 255.0) as u8;

        [r, g, b]
    }

    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn get_icy_blue_color(value: f32) -> [u8; 4] {
        let v = value.clamp(0.0, 1.0);
        let h = v.mul_add(-128.0, 191.0).rem_euclid(256.0) * (360.0 / 255.0);
        let s = v.mul_add(128.0, 127.0).clamp(0.0, 255.0) / 255.0;
        let l = v.mul_add(255.0, 0.0).clamp(0.0, 255.0) / 255.0;
        let [r, g, b] = hsl_to_rgb(h, s, l);
        [r, g, b, 255]
    }

    pub static COLOR_LUT: LazyLock<Vec<[u8; 4]>> = LazyLock::new(|| {
        (0..256)
            .map(|i| get_icy_blue_color(i as f32 / 255.0))
            .collect()
    });

    pub fn calculate_spectrogram(
        audio_data: &[f32],
        fft: &dyn Fft<f32>,
        fft_size: usize,
        hop_length: usize,
        num_freq_bins_to_render: usize,
    ) -> (Vec<f32>, f32, usize) {
        let num_time_bins = if audio_data.len() >= fft_size {
            (audio_data.len() - fft_size) / hop_length + 1
        } else {
            0
        };
        if num_time_bins == 0 {
            return (Vec::new(), 0.0, 0);
        }

        let mut flat_spectrogram = vec![0.0f32; num_time_bins * num_freq_bins_to_render];

        let process_time_bin =
            |buffer: &mut Vec<Complex<f32>>, (i, time_bin_slice): (usize, &mut [f32])| -> f32 {
                let current_pos = i * hop_length;
                let audio_chunk = &audio_data[current_pos..current_pos + fft_size];

                buffer
                    .iter_mut()
                    .zip(audio_chunk.iter())
                    .for_each(|(c, &s)| {
                        *c = Complex { re: s, im: 0.0 };
                    });
                fft.process(buffer);

                let mut max_local = 0.0f32;
                time_bin_slice
                    .iter_mut()
                    .zip(buffer[0..num_freq_bins_to_render].iter())
                    .for_each(|(mag_slot, complex_val)| {
                        let mag = complex_val.norm();
                        *mag_slot = mag;
                        if mag > max_local {
                            max_local = mag;
                        }
                    });
                max_local
            };

        cfg_if! {
            if #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-parallel"))] {
                use rayon::prelude::*;

                let max_magnitude = flat_spectrogram
                    .par_chunks_mut(num_freq_bins_to_render)
                    .enumerate()
                    .map_init(
                        || vec![Complex::new(0.0, 0.0); fft_size],
                        process_time_bin,
                    )
                    .reduce(|| 0.0f32, f32::max);

                (flat_spectrogram, max_magnitude, num_time_bins)
            } else {
                let mut max_magnitude = 0.0f32;
                let mut buffer = vec![Complex::new(0.0, 0.0); fft_size];

                flat_spectrogram
                    .chunks_mut(num_freq_bins_to_render)
                    .enumerate()
                    .for_each(|(i, time_bin_slice)| {
                        let max_local = process_time_bin(&mut buffer, (i, time_bin_slice));
                        if max_local > max_magnitude {
                            max_magnitude = max_local;
                        }
                    });

                (flat_spectrogram, max_magnitude, num_time_bins)
            }
        }
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
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let max_freq = 20000.0;
        let freq_resolution = sample_rate as f32 / fft_size as f32;
        let num_freq_bins = (max_freq / freq_resolution).round() as usize;
        let num_freq_bins = num_freq_bins.min(fft_size / 2);

        if num_freq_bins == 0 {
            return Err(SpectrogramError::NoFrequencyBins);
        }

        let (mut values, _max_magnitude, num_time_bins) =
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
    ) {
        let min_bin = 1;
        let max_bin = spectrogram.num_freq_bins;

        if max_bin <= min_bin {
            let color = COLOR_LUT[0];
            for x in 0..img_width {
                let pixel_start_index = x * 4;
                row_buffer[pixel_start_index..pixel_start_index + 4].copy_from_slice(&color);
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
            let color = COLOR_LUT[color_index];
            let pixel_start_index = x * 4;
            row_buffer[pixel_start_index..pixel_start_index + 4].copy_from_slice(&color);
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-parallel"))]
    #[must_use]
    pub fn render_pixels_parallel(
        spectrogram: &RenderableSpectrogram,
        img_width: usize,
        img_height: usize,
        log_ratio: f32,
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
            );
        }
        pixels
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
    pub fn generate_spectrogram_image(
        audio_data: &[f32],
        sample_rate: u32,
        fft_size: usize,
        hop_length: usize,
        img_width: usize,
        img_height: usize,
        gain: f32,
    ) -> Result<Vec<u8>, JsValue> {
        let spectrogram_data = core::process_audio_to_spectrogram(
            audio_data,
            sample_rate,
            fft_size,
            hop_length,
            gain,
        )?;

        let pos_fref = 0.001;
        let freq_ref = 1000.0;

        let min_bin = 1;
        let max_bin = spectrogram_data.num_freq_bins;

        let log_ratio = if max_bin <= min_bin || img_height == 0 {
            0.0
        } else {
            let freq_resolution = sample_rate as f32 / fft_size as f32;
            let b_fref = (freq_ref / freq_resolution).clamp(min_bin as f32, (max_bin - 1) as f32);

            let min_bin_f = min_bin as f32;
            let max_bin_f = max_bin as f32;

            let clin = min_bin_f + (max_bin_f - min_bin_f) * pos_fref;
            let scale_log = (max_bin_f / min_bin_f).ln();
            let clog = min_bin_f * (pos_fref * scale_log).exp();

            let denominator = clog - clin;
            if denominator.abs() < 1e-6 {
                0.5
            } else {
                ((b_fref - clin) / denominator).clamp(0.0, 1.0)
            }
        };

        cfg_if! {
            if #[cfg(feature = "wasm-parallel")] {
                Ok(core::render_pixels_parallel(&spectrogram_data, img_width, img_height, log_ratio))
            } else {
                Ok(core::render_pixels_serial(&spectrogram_data, img_width, img_height, log_ratio))
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod native_api {
    use super::core::{self, SpectrogramError};

    pub fn generate_spectrogram_image_native(
        audio_data: &[f32],
        sample_rate: u32,
        fft_size: usize,
        hop_length: usize,
        img_width: usize,
        img_height: usize,
        gain: f32,
    ) -> Result<Vec<u8>, SpectrogramError> {
        let spectrogram_data = core::process_audio_to_spectrogram(
            audio_data,
            sample_rate,
            fft_size,
            hop_length,
            gain,
        )?;

        let pos_fref = 0.001;
        let freq_ref = 1000.0;

        let min_bin = 1;
        let max_bin = spectrogram_data.num_freq_bins;

        let log_ratio = if max_bin <= min_bin || img_height == 0 {
            0.0
        } else {
            let freq_resolution = sample_rate as f32 / fft_size as f32;
            let b_fref = (freq_ref / freq_resolution).clamp(min_bin as f32, (max_bin - 1) as f32);

            let min_bin_f = min_bin as f32;
            let max_bin_f = max_bin as f32;

            let clin = (max_bin_f - min_bin_f).mul_add(pos_fref, min_bin_f);
            let scale_log = (max_bin_f / min_bin_f).ln();
            let clog = min_bin_f * (pos_fref * scale_log).exp();

            let denominator = clog - clin;
            if denominator.abs() < 1e-6 {
                0.5
            } else {
                ((b_fref - clin) / denominator).clamp(0.0, 1.0)
            }
        };

        Ok(core::render_pixels_parallel(
            &spectrogram_data,
            img_width,
            img_height,
            log_ratio,
        ))
    }
}
