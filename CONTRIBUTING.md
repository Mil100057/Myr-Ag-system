# Contributing to Myr-Ag

Thank you for your interest in contributing to Myr-Ag! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome various types of contributions:

- 🐛 **Bug Reports** - Help us fix issues
- 💡 **Feature Requests** - Suggest new capabilities
- 📝 **Documentation** - Improve guides and docs
- 🔧 **Code Changes** - Fix bugs or add features
- 🧪 **Testing** - Help test and validate
- 🌍 **Localization** - Translate to other languages

### Before You Start

1. **Check existing issues** - Your idea might already be discussed
2. **Read the documentation** - Understand how the system works
3. **Set up development environment** - Follow the setup guide
4. **Join discussions** - Comment on existing issues

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- Basic understanding of RAG systems
- Familiarity with FastAPI, Gradio, or LlamaIndex (helpful)

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Myr-Ag.git
   cd Myr-Ag
   ```

2. **Set up environment**
   ```bash
   make setup
   source venv/bin/activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

4. **Start services**
   ```bash
   make start
   ```

## 📝 Making Changes

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Formatting**: Use Black for code formatting
- **Linting**: Pass Flake8 checks
- **Type hints**: Use type hints where appropriate

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Add support for Markdown documents"

# Bad
git commit -m "fix stuff"
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests if applicable
   - Update documentation

3. **Test your changes**
   ```bash
   make test
   make start
   # Test your changes manually
   ```

4. **Submit a pull request**
   - Provide clear description of changes
   - Link related issues
   - Include screenshots for UI changes

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_specific.py -v

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Test new functionality
- Ensure existing features still work
- Use descriptive test names
- Mock external dependencies

## 📚 Documentation

### What to Document

- **New features** - How to use them
- **API changes** - Updated endpoints
- **Configuration** - New settings
- **Examples** - Usage examples

### Documentation Standards

- Use clear, simple language
- Include code examples
- Keep README updated
- Add docstrings to functions

## 🐛 Reporting Issues

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., macOS 24.6.0]
- Python: [e.g., 3.11.7]
- Myr-Ag Version: [e.g., latest main]

**Additional Context**
Any other information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How should this feature work?

**Alternative Solutions**
Other ways to solve the problem

**Additional Context**
Screenshots, examples, etc.
```

## 🔒 Security

### Reporting Security Issues

**Do not** report security issues in public GitHub issues. Instead:

1. Email: support@precellence.icu
2. Subject: "SECURITY: Myr-Ag Vulnerability"
3. Provide detailed description
4. We'll respond within 48 hours

## 🏆 Recognition

### Contributors

We recognize contributors in several ways:

- **GitHub contributors list**
- **Release notes**
- **Contributor hall of fame** (coming soon)
- **Special thanks in documentation**

### Types of Recognition

- **Code contributions** - Pull requests merged
- **Documentation** - Improved guides and docs
- **Testing** - Bug reports and testing
- **Community** - Helping other users

## ❓ Questions?

If you have questions about contributing:

1. **Check existing issues** - Your question might be answered
2. **Comment on relevant issues** - Join the discussion
3. **Create a new issue** - Use the "Question" label
4. **Join our community** - Links coming soon

## 🙏 Thank You

Thank you for contributing to Myr-Ag! Every contribution, no matter how small, helps make this project better for everyone.

---

**Happy Contributing! 🚀📚**
