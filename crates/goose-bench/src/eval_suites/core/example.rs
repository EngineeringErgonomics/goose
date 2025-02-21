use crate::eval_suites::{BenchAgent, Evaluation, EvaluationMetric};
use crate::register_evaluation;
use crate::work_dir::WorkDir;
use async_trait::async_trait;
// use std::fs;

pub struct ExampleEval {}

impl ExampleEval {
    pub fn new() -> Self {
        ExampleEval {}
    }
}

#[async_trait]
impl Evaluation for ExampleEval {
    async fn run(
        &self,
        mut agent: Box<dyn BenchAgent>,
        _work_dir: &mut WorkDir,
    ) -> anyhow::Result<Vec<EvaluationMetric>> {
        println!("ExampleEval - run");
        // let f = work_dir.fs_get(String::from("./arbitrary_dir/arbitrary_file.txt"))?;
        // let _contents = fs::read_to_string(f)?;
        let metrics = Vec::new();
        let _ = agent.prompt("What can you do?".to_string()).await;
        Ok(metrics)
    }

    fn name(&self) -> &str {
        "example_eval"
    }
}

register_evaluation!("core", ExampleEval);
