//! Left-first execution of conditional graphs. Each shared expression has one body.
use super::{AadError, NodeId, Op, Workspace};

#[derive(Clone, Copy)]
enum Instruction {
    Compute(u16),
    SharedLeaf(u16),
    IfZero { condition: u16, target: usize },
    Jump(usize),
    Call { node: u16, target: usize },
    Return(u16),
    End,
}

#[derive(Default)]
pub(super) struct Program {
    code: Box<[Instruction]>,
    pub call_capacity: usize,
}

enum Emit {
    Visit(usize),
    Body(usize),
    Compute(usize),
    Arms {
        condition: usize,
        positive: usize,
        zero: usize,
    },
    Else {
        branch: usize,
        zero: usize,
    },
    Join(usize),
}

impl Program {
    pub(super) fn compile(nodes: &[Op], roots: &[NodeId]) -> Self {
        // Nodes already have topological IDs, so one descending pass counts
        // reachable uses across all output roots and both arms.
        let mut uses = vec![0usize; nodes.len()];
        for root in roots {
            uses[root.0] += 1;
        }
        for (i, &op) in nodes.iter().enumerate().rev() {
            if uses[i] > 0 {
                for child in op.operands() {
                    uses[child] += 1;
                }
            }
        }
        let mut code = Vec::new();
        let mut tasks = Vec::new();
        for root in roots {
            tasks.push(Emit::Visit(root.0));
            emit(nodes, &uses, &mut tasks, &mut code);
        }
        code.push(Instruction::End);
        let mut entries = vec![0; nodes.len()];
        let mut call_capacity = 0;
        for (i, &op) in nodes.iter().enumerate() {
            if uses[i] > 1 && !matches!(op, Op::Input(_) | Op::Constant(_)) {
                entries[i] = code.len();
                tasks.push(Emit::Body(i));
                emit(nodes, &uses, &mut tasks, &mut code);
                code.push(Instruction::Return(i as u16));
                call_capacity += 1;
            }
        }
        for step in &mut code {
            if let Instruction::Call { node, target } = step {
                *target = entries[usize::from(*node)];
            }
        }
        Self {
            code: code.into_boxed_slice(),
            call_capacity,
        }
    }

    pub(super) fn run(
        &self,
        nodes: &[Op],
        inputs: &[f64],
        w: &mut Workspace,
    ) -> Result<(), AadError> {
        w.stack.clear();
        w.active.clear();
        w.visited.fill(false);
        let mut pc = 0;
        loop {
            let step = self.code[pc];
            pc += 1;
            let node = match step {
                Instruction::Compute(node) => node,
                Instruction::SharedLeaf(node) => {
                    if w.visited[usize::from(node)] {
                        continue;
                    }
                    w.visited[usize::from(node)] = true;
                    node
                }
                Instruction::IfZero { condition, target } => {
                    if w.primal[usize::from(condition)] == 0.0 {
                        pc = target;
                    }
                    continue;
                }
                Instruction::Jump(target) => {
                    pc = target;
                    continue;
                }
                Instruction::Call { node, target } => {
                    if !w.visited[usize::from(node)] {
                        w.stack.push(pc);
                        pc = target;
                    }
                    continue;
                }
                Instruction::Return(node) => {
                    w.visited[usize::from(node)] = true;
                    pc = w.stack.pop().expect("each shared body has a caller");
                    continue;
                }
                Instruction::End => return Ok(()),
            };
            let i = usize::from(node);
            let value = nodes[i].value(inputs, &w.primal);
            if !value.is_finite() {
                return Err(AadError::NonFiniteResult { node: i });
            }
            w.primal[i] = value;
            w.active.push(node);
        }
    }
}

fn emit(nodes: &[Op], uses: &[usize], tasks: &mut Vec<Emit>, code: &mut Vec<Instruction>) {
    while let Some(task) = tasks.pop() {
        match task {
            Emit::Visit(i) if uses[i] > 1 => {
                let node = i as u16;
                code.push(if matches!(nodes[i], Op::Input(_) | Op::Constant(_)) {
                    Instruction::SharedLeaf(node)
                } else {
                    Instruction::Call { node, target: 0 }
                });
            }
            Emit::Visit(i) | Emit::Body(i) => {
                tasks.push(Emit::Compute(i));
                if let Op::Select {
                    condition,
                    positive,
                    zero,
                } = nodes[i]
                {
                    tasks.push(Emit::Arms {
                        condition,
                        positive,
                        zero,
                    });
                    tasks.push(Emit::Visit(condition));
                } else {
                    tasks.extend(nodes[i].operands().rev().map(Emit::Visit));
                }
            }
            Emit::Compute(i) => code.push(Instruction::Compute(i as u16)),
            Emit::Arms {
                condition,
                positive,
                zero,
            } => {
                let branch = code.len();
                code.push(Instruction::IfZero {
                    condition: condition as u16,
                    target: 0,
                });
                tasks.push(Emit::Else { branch, zero });
                tasks.push(Emit::Visit(positive));
            }
            Emit::Else { branch, zero } => {
                let jump = code.len();
                code.push(Instruction::Jump(0));
                let start = code.len();
                let Instruction::IfZero { target, .. } = &mut code[branch] else {
                    unreachable!()
                };
                *target = start;
                tasks.push(Emit::Join(jump));
                tasks.push(Emit::Visit(zero));
            }
            Emit::Join(jump) => code[jump] = Instruction::Jump(code.len()),
        }
    }
}
