# Contributing to Ergo Sensor

First off, thank you for considering contributing to Ergo Sensor! It's people like you that make Ergo Sensor such a great tool for workplace safety.

## 🌈 How Can I Contribute?

### Reporting Bugs
- Use the [GitHub Issue Tracker](https://github.com/charrada1993/Ergo_Sensor/issues).
- Describe the bug in detail, including steps to reproduce.
- Attach logs from the `logs/` directory if relevant.

### Suggesting Enhancements
- Open a [new issue](https://github.com/charrada1993/Ergo_Sensor/issues/new) with the tag "enhancement".
- Explain the clinical or technical benefit of the feature.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

## 💻 Technical Standards

- **Python**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- **JavaScript**: Use ES6+ standards.
- **Documentation**: All new features must include an update to the relevant `.md` file in `docs/`.
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) (e.g., `feat:`, `fix:`, `docs:`, `refactor:`).

## 🧪 Development Workflow

```bash
# 1. Install dev dependencies
pip install -r requirements.txt pytest black flake8

# 2. Run tests
pytest tests/

# 3. Format code
black .
```

## 📜 Code of Conduct
This project and everyone participating in it is governed by the [Ergo Sensor Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

**Questions?** Reach out to the maintainer at [GitHub Profile](https://github.com/charrada1993).
