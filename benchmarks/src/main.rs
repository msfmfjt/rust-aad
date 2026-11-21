use rust_aad_bench::{FNV_OFFSET, Scalar, Simd, fixture::Case, fnv, run_with_output};
use serde_json::json;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

fn size(arg: &str) -> Result<usize, Box<dyn Error>> {
    let n = arg.parse()?;
    if !(1..=1_000_000).contains(&n) {
        return Err("path count must be in 1..=1000000".into());
    }
    Ok(n)
}

fn case(id: &str) -> Result<Case, Box<dyn Error>> {
    Case::parse(id).ok_or_else(|| format!("unknown workload {id}").into())
}

fn fixture(id: &str, paths: usize, directory: Option<&Path>) -> Result<(), Box<dyn Error>> {
    let case = case(id)?;
    let program = case.program();
    let bytes = serde_json::to_vec(&program)?;
    let mut hash = FNV_OFFSET;
    let mut x = vec![0.; case.inputs()];
    let mut y = vec![0.; program.outputs.len()];
    let mut grad = x.clone();
    // Create a fresh directory to avoid overwriting other replay bundles.
    let mut streams = if let Some(dir) = directory {
        fs::create_dir(dir)?;
        fs::write(dir.join("program.json"), &bytes)?;
        Some((
            BufWriter::new(File::create(dir.join("inputs.f64le"))?),
            BufWriter::new(File::create(dir.join("expected.f64le"))?),
        ))
    } else {
        None
    };
    for path in 0..paths {
        for (j, value) in x.iter_mut().enumerate() {
            *value = case.input(path, j);
            hash = fnv(hash, &value.to_le_bytes());
        }
        if let Some((inputs, expected)) = &mut streams {
            case.oracle(&x, &mut y, &mut grad);
            for v in &x {
                inputs.write_all(&v.to_le_bytes())?;
            }
            for v in y.iter().chain(&grad) {
                expected.write_all(&v.to_le_bytes())?;
            }
        }
    }
    if let Some((mut inputs, mut expected)) = streams {
        inputs.flush()?;
        expected.flush()?;
    }
    let info = json!({"schema":1,"workload":id,"paths":paths,"inputs":x.len(),
        "outputs":y.len(),"nodes":program.nodes.len(),"seeds":program.seeds,
        "program_fnv1a64":format!("{:016x}",fnv(FNV_OFFSET,&bytes)),
        "input_fnv1a64":format!("{hash:016x}")});
    if let Some(dir) = directory {
        fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&info)?)?;
    }
    println!("{info}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, id] if flag == "--describe" => {
            println!("{}", serde_json::to_string(&case(id)?.program())?);
        }
        [flag, id, paths] if flag == "--fingerprint" => return fixture(id, size(paths)?, None),
        [flag, id, paths, dir] if flag == "--export" => {
            return fixture(id, size(paths)?, Some(Path::new(dir)));
        }
        [backend, id, paths] | [backend, id, paths, _] => {
            case(id)?;
            let paths = size(paths)?;
            let evaluate = match backend.as_str() {
                "rust-scalar" => run_with_output::<Scalar>,
                "rust-simd2" => run_with_output::<Simd>,
                _ => return Err(format!("unknown backend {backend}").into()),
            };
            let mut file = args.get(3).map(File::create_new).transpose()?.map(BufWriter::new);
            let writer = file.as_mut().map(|f| f as &mut dyn Write);
            let result = evaluate(id, paths, writer)?;
            println!("{result}");
        }
        _ => return Err("usage: rust-aad-bench BACKEND ID PATHS [NEW_OUTPUT.f64le] | --describe ID | --fingerprint ID PATHS | --export ID PATHS NEW_DIRECTORY".into()),
    }
    Ok(())
}
