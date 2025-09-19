# Contributing to AI Multitool 🤝

Thank you for your interest in contributing to the AI Multitool! This document provides guidelines and information for contributors.

## 🎯 Ways to Contribute

### 🐛 Reporting Bugs
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml)
- Provide detailed reproduction steps
- Include your environment information
- Add screenshots when applicable

### 💡 Suggesting Features
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml)
- Explain the use case and benefits
- Consider implementation complexity
- Discuss with maintainers before large changes

### 🔧 Code Contributions
- Fork the repository
- Create a feature branch
- Make your changes
- Test thoroughly
- Submit a pull request

## 🚀 Development Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (for testing with models)
- Git

### Local Development
1. **Fork and clone:**
```bash
git clone https://github.com/YOUR-USERNAME/-AI-Multitool-Image-Description-Code-Generation
cd -AI-Multitool-Image-Description-Code-Generation
```

2. **Set up environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python app.py
```

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where appropriate
- Add docstrings to functions and classes
- Maximum line length: 100 characters

### Code Quality
- Write clear, readable code
- Add comments for complex logic
- Include error handling
- Write tests for new functionality

### Git Workflow
- Use clear, descriptive commit messages
- Reference issues in commits when applicable
- Keep commits focused and atomic
- Rebase before submitting pull requests

## 🧪 Testing

### Running Tests
```bash
python -m pytest test_code_extraction.py -v
```

### Test Requirements
- Add tests for new functionality
- Ensure existing tests pass
- Test edge cases and error conditions
- Include integration tests where appropriate

## 📋 Pull Request Process

1. **Before You Start**
   - Check existing issues and PRs
   - Discuss major changes in an issue first
   - Ensure your local branch is up to date

2. **Making Changes**
   - Create a descriptive branch name: `feature/add-new-model` or `bugfix/fix-memory-leak`
   - Make focused, logical commits
   - Update documentation as needed
   - Add or update tests

3. **Submitting**
   - Use the [Pull Request template](.github/pull_request_template.md)
   - Provide clear description of changes
   - Reference related issues
   - Ensure CI checks pass

4. **Review Process**
   - Maintainers will review your PR
   - Address feedback promptly
   - Make requested changes
   - Maintain discussion in PR comments

## 🏗️ Architecture Guidelines

### Adding New Features
- Follow existing code organization
- Use the `ModelManager` class for model operations
- Add proper error handling and logging
- Consider performance implications

### UI Changes
- Maintain consistent design language
- Test on different screen sizes
- Consider accessibility
- Add proper labels and descriptions

### Model Integration
- Use existing model loading patterns
- Implement proper resource management
- Add configuration options
- Document model requirements

## 📚 Documentation

### Code Documentation
- Add docstrings to public functions
- Include parameter descriptions
- Document return types
- Provide usage examples

### User Documentation
- Update README for user-facing changes
- Add to DOCUMENTATION.md for technical details
- Include screenshots for UI changes
- Update parameter guides for new options

## 🤔 Getting Help

- **Questions**: Open a discussion or issue
- **Chat**: Comment on existing issues
- **Documentation**: Check DOCUMENTATION.md
- **Examples**: Look at existing code patterns

## 📜 Code of Conduct

### Our Standards
- **Respectful**: Be kind and respectful to others
- **Inclusive**: Welcome diverse perspectives
- **Constructive**: Provide helpful feedback
- **Professional**: Maintain professional communication

### Unacceptable Behavior
- Harassment or discriminatory language
- Spam or off-topic content
- Sharing private information
- Disruptive behavior

## 🏷️ Labels and Issues

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to docs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

### Priority Labels
- `priority: high`: Important issues
- `priority: medium`: Standard issues
- `priority: low`: Nice to have

## 🎉 Recognition

Contributors will be:
- Listed in the repository contributors
- Mentioned in release notes for significant contributions
- Given credit in documentation updates

---

Thank you for contributing to make AI Multitool better! 🚀

*For more information, contact [@MariamMahm0ud](https://github.com/MariamMahm0ud)*