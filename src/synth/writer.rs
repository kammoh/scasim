//! The FST and VCD writers of the generator. Both take the value changes in time order.

use super::SynthError;
use fst_writer::{
    FstFileType, FstInfo, FstScopeType, FstSignalId, FstSignalType, FstVarDirection, FstVarType,
    open_fst,
};
use std::io::Write;
use std::path::Path;

/// The FST buffer is flushed to disk (at the start of a segment) when it is larger than this.
const FLUSH_BYTES: usize = 32 << 20;

/// One signal of the design, in the order of declaration.
pub(crate) struct Decl {
    /// The scope names from the root.
    pub scope: Vec<String>,
    pub name: String,
    pub width: u32,
}

/// A destination of value changes. All signals start at 0 at time 0.
pub(crate) trait Sink {
    /// Called before the first time step of a segment. The writer may flush here.
    fn segment_start(&mut self) -> Result<(), SynthError>;
    /// Starts the time step `time` (strictly after the previous one).
    fn time(&mut self, time: u64) -> Result<(), SynthError>;
    /// Sets the value of signal `signal` (the bits of `value`, MSB first) in the current step.
    fn change(&mut self, signal: usize, value: u64) -> Result<(), SynthError>;
}

/// Writes `value` as `width` characters `0` and `1`, MSB first.
fn bits(buffer: &mut Vec<u8>, width: u32, value: u64) {
    buffer.clear();
    buffer.extend(
        (0..width)
            .rev()
            .map(|bit| b'0' + ((value >> bit) & 1) as u8),
    );
}

/// Calls `up` for every scope that must close and `down` for every scope that must open to go
/// from the scope stack `open` to `target`.
fn move_scope(
    open: &mut Vec<String>,
    target: &[String],
    mut up: impl FnMut() -> Result<(), SynthError>,
    mut down: impl FnMut(&str) -> Result<(), SynthError>,
) -> Result<(), SynthError> {
    let common = open.iter().zip(target).take_while(|(a, b)| a == b).count();
    while open.len() > common {
        up()?;
        open.pop();
    }
    for name in &target[common..] {
        down(name)?;
        open.push(name.clone());
    }
    Ok(())
}

type FstBody = fst_writer::FstBodyWriter<std::io::BufWriter<std::fs::File>>;

pub(crate) struct FstSink {
    body: FstBody,
    ids: Vec<FstSignalId>,
    widths: Vec<u32>,
    buffer: Vec<u8>,
}

fn fst_error(error: fst_writer::FstWriteError) -> SynthError {
    SynthError::Fst(error.to_string())
}

impl FstSink {
    pub fn create(path: &Path, decls: &[Decl]) -> Result<FstSink, SynthError> {
        let info = FstInfo {
            start_time: 0,
            timescale_exponent: -12,
            version: "scasim synth".into(),
            date: "2026-10-09".into(),
            file_type: FstFileType::Verilog,
        };
        let mut header = open_fst(path, &info).map_err(fst_error)?;
        let mut open: Vec<String> = Vec::new();
        let mut ids = Vec::new();
        for decl in decls {
            // The two closures need the header at different times, so they share a cell.
            let header = std::cell::RefCell::new(&mut header);
            move_scope(
                &mut open,
                &decl.scope,
                || header.borrow_mut().up_scope().map_err(fst_error),
                |name| {
                    header
                        .borrow_mut()
                        .scope(name, "", FstScopeType::Module)
                        .map_err(fst_error)
                },
            )?;
            let id = header
                .borrow_mut()
                .var(
                    &decl.name,
                    FstSignalType::bit_vec(decl.width),
                    FstVarType::Wire,
                    FstVarDirection::Implicit,
                    None,
                )
                .map_err(fst_error)?;
            ids.push(id);
        }
        for _ in &open {
            header.up_scope().map_err(fst_error)?;
        }
        let mut body = header.finish().map_err(fst_error)?;
        let mut buffer = Vec::new();
        for (decl, id) in decls.iter().zip(&ids) {
            bits(&mut buffer, decl.width, 0);
            body.signal_change(*id, &buffer).map_err(fst_error)?;
        }
        Ok(FstSink {
            body,
            ids,
            widths: decls.iter().map(|d| d.width).collect(),
            buffer,
        })
    }

    pub fn finish(self) -> Result<(), SynthError> {
        self.body.finish().map_err(fst_error)
    }
}

impl Sink for FstSink {
    fn segment_start(&mut self) -> Result<(), SynthError> {
        if self.body.size() > FLUSH_BYTES {
            self.body.flush().map_err(fst_error)?;
        }
        Ok(())
    }

    fn time(&mut self, time: u64) -> Result<(), SynthError> {
        self.body.time_change(time).map_err(fst_error)
    }

    fn change(&mut self, signal: usize, value: u64) -> Result<(), SynthError> {
        bits(&mut self.buffer, self.widths[signal], value);
        self.body
            .signal_change(self.ids[signal], &self.buffer)
            .map_err(fst_error)
    }
}

/// Checks that the FST file has the time table that the writer wrote: `0` and then `steps`
/// strictly increasing times ending at `last`. `fst-writer` 0.3.1 writes a time table that
/// nobody can read in some cases (the compressed and the raw table have the same length).
pub(crate) fn fst_time_table_ok(path: &Path, steps: u64, last: u64) -> bool {
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    let Ok(reader) = fst_reader::FstReader::open_and_read_time_table(std::io::BufReader::new(file))
    else {
        return false;
    };
    match reader.get_time_table() {
        Some(table) => {
            table.len() as u64 == steps + 1
                && table[0] == 0
                && table.last() == Some(&last)
                && table.windows(2).all(|w| w[0] < w[1])
        }
        None => false,
    }
}

pub(crate) struct VcdSink {
    out: std::io::BufWriter<std::fs::File>,
    widths: Vec<u32>,
    buffer: Vec<u8>,
}

/// The VCD identifier code of signal `n`: printable characters, `!` to `~`.
fn vcd_id(mut n: usize) -> String {
    let mut id = String::new();
    loop {
        id.push(char::from(b'!' + (n % 94) as u8));
        n /= 94;
        if n == 0 {
            return id;
        }
    }
}

impl VcdSink {
    pub fn create(path: &Path, decls: &[Decl]) -> Result<VcdSink, SynthError> {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(out, "$timescale 1 ps $end")?;
        let mut open: Vec<String> = Vec::new();
        for (i, decl) in decls.iter().enumerate() {
            let text = std::cell::RefCell::new(&mut out);
            move_scope(
                &mut open,
                &decl.scope,
                || Ok(writeln!(text.borrow_mut(), "$upscope $end")?),
                |name| Ok(writeln!(text.borrow_mut(), "$scope module {name} $end")?),
            )?;
            writeln!(
                out,
                "$var wire {} {} {} $end",
                decl.width,
                vcd_id(i),
                decl.name
            )?;
        }
        for _ in &open {
            writeln!(out, "$upscope $end")?;
        }
        writeln!(out, "$enddefinitions $end\n#0\n$dumpvars")?;
        let mut sink = VcdSink {
            out,
            widths: decls.iter().map(|d| d.width).collect(),
            buffer: Vec::new(),
        };
        for signal in 0..decls.len() {
            sink.change(signal, 0)?;
        }
        writeln!(sink.out, "$end")?;
        Ok(sink)
    }

    pub fn finish(mut self) -> Result<(), SynthError> {
        Ok(self.out.flush()?)
    }
}

impl Sink for VcdSink {
    fn segment_start(&mut self) -> Result<(), SynthError> {
        Ok(())
    }

    fn time(&mut self, time: u64) -> Result<(), SynthError> {
        Ok(writeln!(self.out, "#{time}")?)
    }

    fn change(&mut self, signal: usize, value: u64) -> Result<(), SynthError> {
        let width = self.widths[signal];
        bits(&mut self.buffer, width, value);
        let id = vcd_id(signal);
        if width == 1 {
            self.out.write_all(&self.buffer)?;
        } else {
            self.out.write_all(b"b")?;
            self.out.write_all(&self.buffer)?;
            self.out.write_all(b" ")?;
        }
        self.out.write_all(id.as_bytes())?;
        Ok(self.out.write_all(b"\n")?)
    }
}
