# Contributing to RepliBuild

Thank you for considering contributing to RepliBuild! We welcome contributions from everyone.

## Ways to Contribute

### 🐛 Report Bugs
- Use the [issue tracker](https://github.com/obsidianjulua/RepliBuild.jl/issues)
- Include Julia version, OS, LLVM version, and error messages
- Provide a minimal reproducible example

### 💡 Suggest Features
- Open an issue with the `enhancement` label
- Describe the use case and expected behavior
- Consider if it fits RepliBuild's scope

### 📝 Improve Documentation
- Fix typos or clarify existing docs
- Add examples and tutorials
- Document undocumented features

### 🔧 Submit Code
- Fix bugs, implement features, or optimize performance
- Follow the development workflow below

## Development Workflow

### 1. Fork and Clone
```bash
git clone git@github.com:YOUR-USERNAME/RepliBuild.jl.git
cd RepliBuild.jl
```

### 2. Create a Branch
```bash
git checkout -b fix-issue-123
# or
git checkout -b feature-awesome-thing
```

### 3. Make Changes
- Write clear, documented code
- Follow Julia style conventions
- Add tests for new functionality

### 4. Test Your Changes
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

### 5. Commit
```bash
git add .
git commit -m "Fix: brief description of change

Longer explanation if needed. Reference issues with #123."
```

### 6. Push and Create PR
```bash
git push origin fix-issue-123
```
Then open a Pull Request on GitHub.

## Code Guidelines

### Style
- Follow [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Use 4 spaces for indentation
- Maximum line length: 92 characters
- Use descriptive variable names

### Documentation
- Document all public functions with docstrings
- Include examples in docstrings
- Update README.md if adding user-facing features

### Testing
- All new features need tests
- Bug fixes should include regression tests
- Tests should be in `test/runtests.jl`
- Aim for high code coverage

### Commits
- Use descriptive commit messages
- Reference issues: `Fix #123: description`
- Commit message format:
  ```
  Category: Brief summary (50 chars)

  Detailed explanation if needed (wrap at 72 chars).

  Fixes #123
  ```

Categories: `Fix`, `Feature`, `Docs`, `Test`, `Refactor`, `Perf`

## Project Structure

```
RepliBuild.jl/
├── src/                    # Source code
│   ├── RepliBuild.jl      # Main module
│   ├── BuildBridge.jl     # Command execution
│   ├── LLVMake.jl         # C++ compiler
│   ├── JuliaWrapItUp.jl   # Binary wrapper
│   └── ...                # Other modules
├── test/                   # Test suite
│   └── runtests.jl        # Main test file
├── docs/                   # Documentation
├── examples/              # Example projects
└── sysimage/              # Precompilation scripts
```

## Areas for Contribution

### High Priority
- **Error messages**: Make them more helpful and actionable
- **Platform support**: Test and fix Windows/macOS issues
- **LLVM compatibility**: Support more LLVM versions
- **Test coverage**: Add more comprehensive tests
- **Documentation**: More examples and tutorials

### Ideas for Features
- **CMake integration**: Better CMake project import
- **Dependency detection**: Auto-detect system libraries
- **Template system**: More project templates
- **Performance**: Optimize compilation pipeline
- **Diagnostics**: Better build diagnostics and logging

### Good First Issues
Look for issues labeled `good first issue` - these are great starting points!

## Module Contributions

If you've created logic for handling specific situations (error handling, library detection, etc.), we encourage you to submit it! Examples:

- Custom type mappings for bindings
- Error recovery strategies
- Platform-specific fixes
- Build optimization techniques
- Integration with other tools

## Getting Help

- **Questions?** Open a [discussion](https://github.com/obsidianjulua/RepliBuild.jl/discussions)
- **Stuck?** Ask in the issue or PR
- **Want to chat?** Reach out to maintainers

## Code Review Process

1. Maintainers will review your PR
2. Address any feedback or questions
3. Once approved, a maintainer will merge
4. Your contribution will be in the next release!

## Recognition

All contributors are recognized in:
- GitHub contributors list
- Release notes
- Optional: CONTRIBUTORS.md file

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for making RepliBuild better! 🚀
