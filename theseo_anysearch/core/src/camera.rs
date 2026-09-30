use glam::{Mat4, Vec2, Vec3};

use crate::voxel::world::WORLD_CENTER;

#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Camera {
    pub fn new_default() -> Self {
        let target = Vec3::new(
            WORLD_CENTER.0 as f32,
            WORLD_CENTER.1 as f32,
            WORLD_CENTER.2 as f32,
        );
        let position = Vec3::new(500.0, 500.0, -200.0);
        let mut camera = Self {
            position,
            target,
            up: Vec3::Y,
            fov_y_deg: 60.0,
            near: 1.0,
            far: 2000.0,
            yaw: 0.0,
            pitch: 0.0,
            distance: 0.0,
        };
        camera.sync_angles_from_position();
        camera
    }

    pub fn reset(&mut self) {
        *self = Self::new_default();
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    pub fn projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        Mat4::perspective_rh(
            self.fov_y_deg.to_radians(),
            aspect_ratio.max(0.01),
            self.near,
            self.far,
        )
    }

    pub fn orbit(&mut self, delta: Vec2) {
        let sensitivity = 0.01;
        self.yaw -= delta.x * sensitivity;
        self.pitch += delta.y * sensitivity;
        self.pitch = self.pitch.clamp(-1.45, 1.45);
        self.sync_position_from_angles();
    }

    pub fn pan(&mut self, delta: Vec2) {
        let pan_scale = (self.distance / 450.0).max(0.05);
        let right = self.right();
        let up = self.up;
        let translation = (-right * delta.x + up * delta.y) * pan_scale;
        self.target += translation;
        self.position += translation;
    }

    pub fn zoom(&mut self, wheel_delta_y: f32) {
        let factor: f32 = if wheel_delta_y > 0.0 { 0.9 } else { 1.1 };
        let scaled = self.distance * factor.powf(wheel_delta_y.abs() * 0.05);
        self.distance = scaled.clamp(3.0, 5000.0);
        self.sync_position_from_angles();
    }

    pub fn move_local(&mut self, forward: f32, strafe: f32, delta_time: f32) {
        let speed = 180.0;
        let translate = (self.forward() * forward + self.right() * strafe) * speed * delta_time;
        self.position += translate;
        self.target += translate;
    }

    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize_or_zero()
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(self.up).normalize_or_zero()
    }

    fn sync_angles_from_position(&mut self) {
        let offset = self.position - self.target;
        self.distance = offset.length().max(0.001);
        self.yaw = offset.z.atan2(offset.x);
        self.pitch = (offset.y / self.distance).asin();
    }

    fn sync_position_from_angles(&mut self) {
        let cos_pitch = self.pitch.cos();
        let orbit = Vec3::new(
            self.distance * cos_pitch * self.yaw.cos(),
            self.distance * self.pitch.sin(),
            self.distance * cos_pitch * self.yaw.sin(),
        );
        self.position = self.target + orbit;
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_matrix_not_identity() {
        let camera = Camera::new_default();
        assert_ne!(camera.view_matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn projection_matrix_fov() {
        let camera = Camera::new_default();
        let mut camera2 = camera.clone();
        camera2.fov_y_deg = 90.0;
        assert_ne!(
            camera.projection_matrix(1.0),
            camera2.projection_matrix(1.0)
        );
    }

    #[test]
    fn orbit_changes_position() {
        let mut camera = Camera::new_default();
        let before = camera.position;
        camera.orbit(Vec2::new(1.0, 0.0));
        assert_ne!(camera.position, before);
    }

    #[test]
    fn zoom_clamped() {
        let mut camera = Camera::new_default();
        // Zoom in far beyond the min clamp
        for _ in 0..200 {
            camera.zoom(10.0);
        }
        assert!(camera.distance >= 3.0, "distance below minimum clamp");
        // Zoom out far beyond the max clamp
        for _ in 0..200 {
            camera.zoom(-10.0);
        }
        assert!(camera.distance <= 5000.0, "distance above maximum clamp");
    }
}
