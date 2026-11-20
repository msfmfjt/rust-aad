use rust_aad_core::{AadError, GraphBuilder};

#[test]
fn weighted_multi_output_vjp_and_workspace_reuse() {
    let mut b = GraphBuilder::new();
    let x = b.input();let y = b.input();
    let xy = b.mul(x,y).unwrap();
    let sum = b.add(x,y).unwrap();
    let g = b.compile(&[xy,sum]).unwrap();let mut w=g.workspace();
    assert_eq!(g.evaluate(&[2.0,3.0],&mut w).unwrap(), &[6.0,5.0]);
    assert_eq!(g.vjp(&[2.0,3.0],&[1.0,2.0],&mut w).unwrap(), &[5.0,4.0]);
    assert_eq!(g.vjp(&[5.0,7.0],&[1.0,0.0],&mut w).unwrap(), &[7.0,5.0]);
}
#[test]
fn select_is_lazy_for_invalid_inactive_branch_and_reverse_tracks_branch() {
    let mut b=GraphBuilder::new();let x=b.input();let z=b.constant(0.0).unwrap();
    let cond=b.greater(x,z).unwrap();
    let positive=b.ln(x).unwrap();let negative=b.neg(x).unwrap();
    let out=b.select(cond,positive,negative).unwrap();
    let g=b.compile(&[out]).unwrap();let mut w=g.workspace();
    assert_eq!(g.evaluate(&[-2.0],&mut w).unwrap(), &[2.0]);
    assert_eq!(g.vjp(&[-2.0],&[1.0],&mut w).unwrap(), &[-1.0]);
    assert_eq!(g.vjp(&[2.0],&[1.0],&mut w).unwrap(), &[0.5]);
}
#[test]
fn min_tie_goes_left() {
    let mut b=GraphBuilder::new();let x=b.input();let y=b.input();
    let o=b.min(x,y).unwrap();let g=b.compile(&[o]).unwrap();let mut w=g.workspace();
    assert_eq!(g.vjp(&[2.0,2.0],&[1.0],&mut w).unwrap(), &[1.0,0.0]);
}
#[test]
fn invalid_shapes_are_rejected() {
    let mut b=GraphBuilder::new();let x=b.input();let g=b.compile(&[x]).unwrap();let mut w=g.workspace();
    assert_eq!(g.evaluate(&[],&mut w).unwrap_err(), AadError::InputShape);
    assert_eq!(g.vjp(&[2.0],&[],&mut w).unwrap_err(), AadError::SeedShape);
}
