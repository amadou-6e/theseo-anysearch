#[derive(Clone, Copy)]
pub(crate) struct CameraAngles {
    pub(crate) cos_yaw: f32,
    pub(crate) sin_yaw: f32,
    pub(crate) cos_pitch: f32,
    pub(crate) sin_pitch: f32,
}

impl CameraAngles {
    pub(crate) fn new(yaw_deg: f32, pitch_deg: f32) -> Self {
        let yaw = yaw_deg.to_radians();
        let pitch = pitch_deg.to_radians();
        Self {
            cos_yaw: yaw.cos(),
            sin_yaw: yaw.sin(),
            cos_pitch: pitch.cos(),
            sin_pitch: pitch.sin(),
        }
    }
}
pub(crate) fn map_coord(
    coord: (u16, u16, u16),
    min_px: f32,
    max_px: f32,
    min_py: f32,
    max_py: f32,
    width: i32,
    height: i32,
    margin: i32,
    cam: CameraAngles,
) -> (i32, i32) {
    let (px, py) = project(coord, cam);
    let span_x = (max_px - min_px).abs().max(1.0);
    let span_y = (max_py - min_py).abs().max(1.0);
    let x = margin as f32 + ((px - min_px) / span_x) * ((width - 2 * margin) as f32);
    let y = margin as f32 + ((py - min_py) / span_y) * ((height - 2 * margin) as f32);
    (x.round() as i32, y.round() as i32)
}

pub(crate) fn project(coord: (u16, u16, u16), cam: CameraAngles) -> (f32, f32) {
    let (sx, sy, _) = project_with_depth((coord.0 as f32, coord.1 as f32, coord.2 as f32), cam);
    (sx, sy)
}

pub(crate) fn project_with_depth(coord: (f32, f32, f32), cam: CameraAngles) -> (f32, f32, f32) {
    let x = coord.0;
    let y = coord.1;
    let z = coord.2;

    // Yaw rotation around Y axis, then pitch rotation around X axis.
    let xr = x * cam.cos_yaw - z * cam.sin_yaw;
    let zr = x * cam.sin_yaw + z * cam.cos_yaw;
    let yr = y * cam.cos_pitch - zr * cam.sin_pitch;
    let zr2 = y * cam.sin_pitch + zr * cam.cos_pitch;

    // zr2 is depth: larger = closer to camera (DEPTH_NEAR_IS_GREATER = true).
    (xr, yr - zr2 * 0.15, zr2)
}

/// Back-face culling: returns true if the face normal points toward the camera.
/// Camera sits at world offset (cos_pitch*cos_yaw, sin_pitch, cos_pitch*sin_yaw).
/// A face is front-facing when dot(normal, camera_offset_dir) > 0.
pub(crate) fn is_face_visible(normal: (f32, f32, f32), cam: CameraAngles) -> bool {
    let (nx, ny, nz) = normal;
    nx * cam.cos_pitch * cam.cos_yaw + ny * cam.sin_pitch + nz * cam.cos_pitch * cam.sin_yaw > 0.0
}
