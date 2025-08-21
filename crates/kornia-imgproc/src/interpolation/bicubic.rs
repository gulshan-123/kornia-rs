use kornia_image::{allocator::ImageAllocator, Image};

/// Cubic interpolation kernel function
///
/// Uses cubic convolution with parameter a = -0.5
///
/// # Arguments
///
/// * `t` - The distance from the pixel center
///
/// # Returns
///
/// The interpolation weight
fn cubic_kernel(t: f32) -> f32 {
    let a = -0.5_f32;
    let abs_t = t.abs();
    
    if abs_t <= 1.0 {
        (a + 2.0) * abs_t.powi(3) - (a + 3.0) * abs_t.powi(2) + 1.0
    } else if abs_t <= 2.0 {
        a * abs_t.powi(3) - 5.0 * a * abs_t.powi(2) + 8.0 * a * abs_t - 4.0 * a
    } else {
        0.0
    }
}

/// Kernel for bicubic interpolation
///
/// # Arguments
///
/// * `image` - The input image container.
/// * `u` - The x coordinate of the pixel to interpolate.
/// * `v` - The y coordinate of the pixel to interpolate.
/// * `c` - The channel of the pixel to interpolate.
///
/// # Returns
///
/// The interpolated pixel value.
pub(crate) fn bicubic_interpolation<const C: usize, A: ImageAllocator>(
    image: &Image<f32, C, A>,
    u: f32,
    v: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows(), image.cols());
    
    // Get the integer and fractional parts
    let iu = u.floor() as i32;
    let iv = v.floor() as i32;
    let frac_u = u - iu as f32;
    let frac_v = v - iv as f32;
    
    let mut result = 0.0;
    
    // Iterate over the 4x4 neighborhood
    for j in -1..=2 {
        for i in -1..=2 {
            // Calculate the actual pixel coordinates
            let x = iu + i;
            let y = iv + j;
            
            // Handle boundary conditions by clamping to image bounds
            let x_clamped = x.clamp(0, cols as i32 - 1) as usize;
            let y_clamped = y.clamp(0, rows as i32 - 1) as usize;
            
            // Get the pixel value
            let pixel_value = *image.get_unchecked([y_clamped, x_clamped, c]);
            
            // Calculate the interpolation weights
            let weight_u = cubic_kernel(frac_u - i as f32);
            let weight_v = cubic_kernel(frac_v - j as f32);
            
            // Accumulate the weighted pixel value
            result += pixel_value * weight_u * weight_v;
        }
    }
    
    result
}