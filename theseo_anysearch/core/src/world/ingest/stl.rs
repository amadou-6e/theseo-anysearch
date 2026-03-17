#[derive(Clone, Debug, Default)]
pub struct StlMesh {
    pub triangles: Vec<[[f32; 3]; 3]>,
}

#[derive(Debug, Clone)]
pub enum ImportError {
    InvalidFormat(String),
}

pub fn parse_ascii_stl(source: &str) -> Result<StlMesh, ImportError> {
    let mut vertices: Vec<[f32; 3]> = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("vertex ") {
            continue;
        }

        let mut parts = trimmed.split_whitespace();
        let _ = parts.next();
        let x = parts
            .next()
            .ok_or_else(|| ImportError::InvalidFormat("missing x".to_string()))?;
        let y = parts
            .next()
            .ok_or_else(|| ImportError::InvalidFormat("missing y".to_string()))?;
        let z = parts
            .next()
            .ok_or_else(|| ImportError::InvalidFormat("missing z".to_string()))?;

        let parse = |v: &str| -> Result<f32, ImportError> {
            v.parse::<f32>().map_err(|_| {
                ImportError::InvalidFormat(format!("could not parse float value '{v}'"))
            })
        };

        vertices.push([parse(x)?, parse(y)?, parse(z)?]);
    }

    if vertices.len() < 3 || vertices.len() % 3 != 0 {
        return Err(ImportError::InvalidFormat(
            "vertex list must be a multiple of 3".to_string(),
        ));
    }

    let mut triangles = Vec::new();
    for chunk in vertices.chunks_exact(3) {
        triangles.push([chunk[0], chunk[1], chunk[2]]);
    }

    Ok(StlMesh { triangles })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SINGLE_TRIANGLE: &str = r#"solid test
  facet normal 0 0 1
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
endsolid test"#;

    #[test]
    fn parse_valid_single_triangle() {
        let mesh = parse_ascii_stl(SINGLE_TRIANGLE).unwrap();
        assert_eq!(mesh.triangles.len(), 1);
        assert_eq!(mesh.triangles[0][0], [0.0, 0.0, 0.0]);
        assert_eq!(mesh.triangles[0][1], [1.0, 0.0, 0.0]);
        assert_eq!(mesh.triangles[0][2], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn parse_valid_two_triangles() {
        let src = "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n\
                   vertex 1 1 0\nvertex 0 0 1\nvertex 1 0 1\n";
        let mesh = parse_ascii_stl(src).unwrap();
        assert_eq!(mesh.triangles.len(), 2);
    }

    #[test]
    fn parse_ignores_non_vertex_lines() {
        let src = "solid foo\nfacet normal 0 0 1\nouter loop\n\
                   vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n\
                   endloop\nendfacet\nendsolid foo";
        let mesh = parse_ascii_stl(src).unwrap();
        assert_eq!(mesh.triangles.len(), 1);
    }

    #[test]
    fn parse_empty_input() {
        let err = parse_ascii_stl("").unwrap_err();
        assert!(matches!(err, ImportError::InvalidFormat(_)));
    }

    #[test]
    fn parse_vertex_count_not_multiple_of_3() {
        let src = "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nvertex 0 0 1\n";
        let err = parse_ascii_stl(src).unwrap_err();
        assert!(matches!(err, ImportError::InvalidFormat(_)));
    }

    #[test]
    fn parse_bad_float() {
        let src = "vertex 1.0 abc 3.0\nvertex 0 0 0\nvertex 1 0 0\n";
        let err = parse_ascii_stl(src).unwrap_err();
        assert!(matches!(err, ImportError::InvalidFormat(_)));
    }

    #[test]
    fn parse_missing_coord() {
        let src = "vertex 1.0 2.0\nvertex 0 0 0\nvertex 1 0 0\n";
        let err = parse_ascii_stl(src).unwrap_err();
        assert!(matches!(err, ImportError::InvalidFormat(_)));
    }
}
