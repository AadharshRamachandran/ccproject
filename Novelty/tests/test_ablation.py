import unittest
from Novelty.ablation import ExperimentConfig, PAPER_NAME, configurations, paper_config

class AblationTests(unittest.TestCase):
    def test_paper_reference_has_stable_name(self): self.assertEqual(paper_config().name(),PAPER_NAME)
    def test_novelty_name_exposes_all_factors(self):
        name=ExperimentConfig('hybrid','quantile_bilstm','tuned_adaptive','dtr','inplace','adaptive').name()
        self.assertIn('quantile_bilstm',name); self.assertIn('inplace',name)

    def test_each_novelty_is_an_isolated_experiment(self):
        configs=list(configurations())
        novelties=[config for config in configs if config.name().startswith('novelty_') and not 'stacked' in config.name()]
        self.assertEqual(len(configs), 8)
        self.assertEqual(len(novelties), 4)
        baseline=paper_config()
        for novelty in novelties:
            changed=sum((
                novelty.forecaster != baseline.forecaster,
                novelty.burst != baseline.burst,
                novelty.execution != baseline.execution,
                novelty.monitoring != baseline.monitoring,
            ))
            self.assertEqual(changed, 1)

if __name__=='__main__': unittest.main()
